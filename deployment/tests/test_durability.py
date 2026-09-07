import os
import random
import signal
import subprocess
import sys
import time
import uuid

import pytest

from deployment import db

N_JOBS = int(os.environ.get("CHAOS_JOBS", "2000"))
N_WORKERS = int(os.environ.get("CHAOS_WORKERS", "3"))
KILL_EVERY = float(os.environ.get("CHAOS_KILL_EVERY", "2.0"))


@pytest.fixture
def clean_db():
    conn = db.connect()
    with conn.cursor() as cur:
        cur.execute("TRUNCATE jobs RESTART IDENTITY")
    yield conn
    conn.close()


def start_worker():
    return subprocess.Popen([sys.executable, "-m", "deployment.worker"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_no_jobs_lost_under_random_kills(clean_db):
    conn = clean_db
    keys = [str(uuid.uuid4()) for _ in range(N_JOBS)]

    accepted = []
    for k in keys:
        row = db.enqueue(conn, k, dict(sentences=["The unemployment rate fell to 3.5%."]))
        accepted.append(row["id"])
    assert len(set(accepted)) == N_JOBS

    workers = [start_worker() for _ in range(N_WORKERS)]
    deadline = time.time() + 120
    try:
        while time.time() < deadline:
            time.sleep(KILL_EVERY)
            victim = random.randrange(len(workers))
            workers[victim].send_signal(signal.SIGKILL)
            workers[victim] = start_worker()
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) AS n FROM jobs "
                            "WHERE status IN ('succeeded','dead_letter')")
                if cur.fetchone()["n"] == N_JOBS:
                    break
    finally:
        for w in workers:
            w.send_signal(signal.SIGKILL)

    with conn.cursor() as cur:
        cur.execute("SELECT status, count(*) AS n FROM jobs GROUP BY status")
        counts = {r["status"]: r["n"] for r in cur.fetchall()}
        cur.execute("SELECT count(*) AS n FROM jobs WHERE status = 'succeeded' "
                    "AND result IS NULL")
        missing_results = cur.fetchone()["n"]

    terminal = counts.get("succeeded", 0) + counts.get("dead_letter", 0)
    print(f"\n{N_JOBS} submitted -> {counts}")
    assert terminal == N_JOBS, f"{N_JOBS - terminal} jobs never reached a terminal state"
    assert missing_results == 0, "a job is marked succeeded with no result"


def test_replay_creates_no_duplicates(clean_db):
    conn = clean_db
    key = str(uuid.uuid4())
    payload = dict(sentences=["The unemployment rate fell to 3.5%."])

    first = db.enqueue(conn, key, payload)
    assert first["inserted"] is True

    for _ in range(5):
        again = db.enqueue(conn, key, payload)
        assert again["inserted"] is False
        assert again["id"] == first["id"]

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM jobs")
        assert cur.fetchone()["n"] == 1


def test_expired_lease_is_reclaimed(clean_db):
    conn = clean_db
    db.enqueue(conn, str(uuid.uuid4()), dict(sentences=["x"]))

    first = db.claim(conn)
    assert first is not None
    assert db.claim(conn) is None

    with conn.cursor() as cur:
        cur.execute("UPDATE jobs SET locked_until = now() - interval '1 second' "
                    "WHERE id = %s", (first["id"],))

    second = db.claim(conn)
    assert second is not None
    assert second["id"] == first["id"]
    assert second["lease_id"] != first["lease_id"]
    assert second["attempts"] == first["attempts"] + 1


def test_stale_lease_cannot_write(clean_db):
    conn = clean_db
    db.enqueue(conn, str(uuid.uuid4()), dict(sentences=["x"]))

    first = db.claim(conn)
    with conn.cursor() as cur:
        cur.execute("UPDATE jobs SET locked_until = now() - interval '1 second' "
                    "WHERE id = %s", (first["id"],))
    second = db.claim(conn)

    assert db.succeed(conn, first["id"], first["lease_id"], {"r": 1}, "v") is False
    assert db.succeed(conn, second["id"], second["lease_id"], {"r": 2}, "v") is True

    row = db.get_job(conn, first["id"])
    assert row["result"] == {"r": 2}


def test_poison_job_dead_letters(clean_db):
    conn = clean_db
    db.enqueue(conn, str(uuid.uuid4()), dict(sentences=["x"]))
    with conn.cursor() as cur:
        cur.execute("UPDATE jobs SET max_attempts = 2")

    for _ in range(3):
        job = db.claim(conn)
        if job is None:
            break
        db.fail(conn, job["id"], job["lease_id"], "boom")
        with conn.cursor() as cur:
            cur.execute("UPDATE jobs SET run_after = now() WHERE id = %s", (job["id"],))

    with conn.cursor() as cur:
        cur.execute("SELECT status, last_error FROM jobs")
        row = cur.fetchone()
    assert row["status"] == "dead_letter"
    assert "boom" in row["last_error"]
