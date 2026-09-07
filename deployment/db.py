import json
import os

import psycopg
from psycopg.rows import dict_row

DSN = os.environ.get("DATABASE_URL", "postgresql://claims:claims@localhost:5432/claims")
LEASE_SECONDS = int(os.environ.get("LEASE_SECONDS", "60"))
MAX_QUEUE_DEPTH = int(os.environ.get("MAX_QUEUE_DEPTH", "10000"))


def connect():
    return psycopg.connect(DSN, row_factory=dict_row, autocommit=True)


def enqueue(conn, idempotency_key: str, payload: dict) -> dict:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO jobs (idempotency_key, payload) VALUES (%s, %s)
            ON CONFLICT (idempotency_key) DO UPDATE SET updated_at = jobs.updated_at
            RETURNING id, status, (xmax = 0) AS inserted
        """, (idempotency_key, json.dumps(payload)))
        return cur.fetchone()


def get_job(conn, job_id: int) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, status, result, model_version, attempts, last_error,
                   created_at, updated_at
            FROM jobs WHERE id = %s
        """, (job_id,))
        return cur.fetchone()


def queue_depth(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM jobs WHERE status = 'queued'")
        return cur.fetchone()["n"]


def claim(conn) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE jobs SET
                status = 'running',
                attempts = attempts + 1,
                lease_id = gen_random_uuid(),
                locked_until = now() + make_interval(secs => %s),
                updated_at = now()
            WHERE id = (
                SELECT id FROM jobs
                WHERE run_after <= now()
                  AND (status = 'queued'
                       OR (status = 'running' AND locked_until < now()))
                ORDER BY run_after
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id, lease_id, payload, attempts, max_attempts
        """, (LEASE_SECONDS,))
        return cur.fetchone()


def heartbeat(conn, job_id: int, lease_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE jobs SET locked_until = now() + make_interval(secs => %s), updated_at = now()
            WHERE id = %s AND lease_id = %s
        """, (LEASE_SECONDS, job_id, lease_id))
        return cur.rowcount == 1


def succeed(conn, job_id: int, lease_id: str, result: dict, model_version: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE jobs SET status = 'succeeded', result = %s, model_version = %s,
                            lease_id = NULL, locked_until = NULL, updated_at = now()
            WHERE id = %s AND lease_id = %s
        """, (json.dumps(result), model_version, job_id, lease_id))
        return cur.rowcount == 1


def fail(conn, job_id: int, lease_id: str, error: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE jobs SET
                status = CASE WHEN attempts >= max_attempts THEN 'dead_letter'::job_status
                              ELSE 'queued'::job_status END,
                run_after = now() + make_interval(secs => least(300, power(2, attempts)::int)),
                last_error = %s, lease_id = NULL, locked_until = NULL, updated_at = now()
            WHERE id = %s AND lease_id = %s
        """, (error[:2000], job_id, lease_id))
        return cur.rowcount == 1
