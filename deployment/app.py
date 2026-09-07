import asyncio
import contextlib
import os
import time

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from deployment import db
from deployment.batcher import Batcher
from deployment.model import MAX_BATCH, MAX_CHARS, Detector

API_KEY = os.environ.get("API_KEY")
DRAIN_SECONDS = float(os.environ.get("DRAIN_SECONDS", "5"))

state: dict = dict(detector=None, conn=None, batcher=None, ready=False, inflight=0)
counters = dict(detect=0, jobs=0, replays=0, positive=0, sentences=0)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    state["detector"] = Detector()
    state["conn"] = db.connect()
    state["batcher"] = Batcher(state["detector"].predict)
    await state["batcher"].start()
    state["ready"] = True
    yield
    state["ready"] = False
    await asyncio.sleep(DRAIN_SECONDS)
    while state["inflight"] > 0:
        await asyncio.sleep(0.05)
    await state["batcher"].stop()
    state["conn"].close()


app = FastAPI(title="claim-detection", lifespan=lifespan)


class DetectRequest(BaseModel):
    sentences: list[str] = Field(min_length=1)


def check_key(key: str | None) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(401, "invalid api key")


def validate(sentences: list[str]) -> None:
    if len(sentences) > MAX_BATCH:
        raise HTTPException(413, f"at most {MAX_BATCH} sentences per request")
    if any(len(s) > MAX_CHARS for s in sentences):
        raise HTTPException(413, f"sentences must be under {MAX_CHARS} characters")


@app.post("/v1/detect")
async def detect(req: DetectRequest, idempotency_key: str = Header(...),
                 x_api_key: str | None = Header(None)):
    check_key(x_api_key)
    validate(req.sentences)
    conn, detector = state["conn"], state["detector"]

    row = db.enqueue_inline(conn, idempotency_key, dict(sentences=req.sentences))
    if not row["inserted"]:
        if row["status"] == "succeeded":
            counters["replays"] += 1
            return dict(**row["result"], model_version=row["model_version"], replayed=True)
        raise HTTPException(409, f"job {row['id']} already in flight")

    state["inflight"] += 1
    started = time.perf_counter()
    try:
        results = await state["batcher"].submit(req.sentences)
        payload = dict(results=results, n=len(results))
        db.succeed(conn, row["id"], row["lease_id"], payload, detector.version)
    except Exception as exc:
        db.fail(conn, row["id"], row["lease_id"], repr(exc))
        raise
    finally:
        state["inflight"] -= 1

    counters["detect"] += 1
    counters["sentences"] += len(results)
    counters["positive"] += sum(r["is_claim"] for r in results)
    return dict(results=results, model_version=detector.version,
                threshold=detector.threshold, job_id=row["id"],
                latency_ms=round((time.perf_counter() - started) * 1000, 2))


@app.post("/v1/jobs", status_code=202)
async def submit(req: DetectRequest, response: Response,
                 idempotency_key: str = Header(...), x_api_key: str | None = Header(None)):
    check_key(x_api_key)
    validate(req.sentences)
    conn = state["conn"]
    if db.queue_depth(conn) >= db.MAX_QUEUE_DEPTH:
        raise HTTPException(429, "queue full", headers={"Retry-After": "5"})
    row = db.enqueue(conn, idempotency_key, dict(sentences=req.sentences))
    if not row["inserted"]:
        response.status_code = 200
        counters["replays"] += 1
    counters["jobs"] += 1
    return dict(job_id=row["id"], status=row["status"])


@app.get("/v1/jobs/{job_id}")
async def job_status(job_id: int, x_api_key: str | None = Header(None)):
    check_key(x_api_key)
    row = db.get_job(state["conn"], job_id)
    if row is None:
        raise HTTPException(404, "no such job")
    return row


@app.get("/healthz")
async def healthz():
    return dict(status="ok")


@app.get("/readyz")
async def readyz():
    if not state["ready"]:
        raise HTTPException(503, "draining")
    try:
        db.queue_depth(state["conn"])
    except Exception:
        raise HTTPException(503, "database unreachable")
    return dict(status="ready", model_version=state["detector"].version)


@app.get("/metrics")
async def metrics():
    conn = state["conn"]
    rate = counters["positive"] / counters["sentences"] if counters["sentences"] else 0.0
    b = state["batcher"].stats
    avg_batch = b["sentences"] / b["batches"] if b["batches"] else 0.0
    acct = db.accounting(conn)
    lines = [
        f'claim_requests_total{{endpoint="detect"}} {counters["detect"]}',
        f'claim_requests_total{{endpoint="jobs"}} {counters["jobs"]}',
        f"claim_replays_total {counters['replays']}",
        f"claim_sentences_total {counters['sentences']}",
        f"claim_positive_rate {rate:.4f}",
        f"claim_batch_avg_size {avg_batch:.2f}",
        f"claim_batch_max_size {b['max_seen']}",
        f"claim_queue_depth {acct['queued']}",
        f"claim_jobs_accepted {acct['accepted']}",
        f"claim_jobs_terminal {acct['terminal']}",
        f"claim_jobs_unaccounted {acct['unaccounted']}",
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain")
