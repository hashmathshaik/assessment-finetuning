import asyncio
import contextlib
import hmac
import os
import time

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from deployment.batcher import Batcher
from deployment.model import MAX_CHARS, MAX_SENTENCES, Detector
from deployment.queue import SYNC_SUBJECT, Queue

API_KEY = os.environ.get("API_KEY")
ALLOW_ANONYMOUS = os.environ.get("ALLOW_ANONYMOUS", "").lower() in ("1", "true", "yes")
DRAIN_SECONDS = float(os.environ.get("DRAIN_SECONDS", "5"))
MAX_STREAM_DEPTH = int(os.environ.get("MAX_STREAM_DEPTH", "50000"))
MAX_KEY_CHARS = int(os.environ.get("MAX_KEY_CHARS", "200"))

state: dict = dict(detector=None, queue=None, batcher=None, ready=False, inflight=0)
counters = dict(detect=0, jobs=0, replays=0, positive=0, sentences=0, rejected=0)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY and not ALLOW_ANONYMOUS:
        raise RuntimeError(
            "API_KEY is not set. Set it, or set ALLOW_ANONYMOUS=1 to serve without auth. "
            "Refusing to start with auth silently disabled.")
    if ALLOW_ANONYMOUS:
        print("WARNING: serving without authentication (ALLOW_ANONYMOUS)", flush=True)
    state["detector"] = Detector()
    state["queue"] = await Queue().connect()
    state["batcher"] = Batcher(state["detector"].predict)
    await state["batcher"].start()
    await state["batcher"].submit(["warmup"])
    state["ready"] = True
    yield
    state["ready"] = False
    deadline = time.monotonic() + DRAIN_SECONDS
    while time.monotonic() < deadline or state["inflight"] > 0:
        await asyncio.sleep(0.05)
    await state["batcher"].stop()
    await state["queue"].close()


app = FastAPI(title="claim-detection", lifespan=lifespan)


class DetectRequest(BaseModel):
    sentences: list[str] = Field(min_length=1)


def check_key(key: str | None) -> None:
    if ALLOW_ANONYMOUS:
        return
    if not key or not hmac.compare_digest(key, API_KEY):
        raise HTTPException(401, "invalid api key")


def validate(sentences: list[str], key: str) -> None:
    if not key.strip():
        raise HTTPException(400, "Idempotency-Key must not be empty")
    if len(key) > MAX_KEY_CHARS:
        raise HTTPException(400, f"Idempotency-Key must be under {MAX_KEY_CHARS} characters")
    if len(sentences) > MAX_SENTENCES:
        raise HTTPException(413, f"at most {MAX_SENTENCES} sentences per request")
    if any(len(s) > MAX_CHARS for s in sentences):
        raise HTTPException(413, f"sentences must be under {MAX_CHARS} characters")


@app.post("/v1/detect")
async def detect(req: DetectRequest, idempotency_key: str = Header(...),
                 x_api_key: str | None = Header(None)):
    check_key(x_api_key)
    validate(req.sentences, idempotency_key)
    q, detector = state["queue"], state["detector"]

    cached = await q.get_result(idempotency_key)
    if cached:
        counters["replays"] += 1
        return dict(**cached, replayed=True)

    state["inflight"] += 1
    started = time.perf_counter()
    try:
        await q.publish(idempotency_key, dict(sentences=req.sentences,
                                              key=idempotency_key),
                        subject=SYNC_SUBJECT)
        results = await state["batcher"].submit(req.sentences)
        answer = dict(results=results, model_version=detector.version,
                      threshold=detector.threshold)
        if not await q.put_result(idempotency_key, answer):
            stored = await q.get_result(idempotency_key)
            if stored:
                counters["replays"] += 1
                return dict(**stored, replayed=True)
    finally:
        state["inflight"] -= 1

    counters["detect"] += 1
    counters["sentences"] += len(results)
    counters["positive"] += sum(r["is_claim"] for r in results)
    return dict(**answer, latency_ms=round((time.perf_counter() - started) * 1000, 2))


@app.post("/v1/jobs", status_code=202)
async def submit(req: DetectRequest, response: Response,
                 idempotency_key: str = Header(...), x_api_key: str | None = Header(None)):
    check_key(x_api_key)
    validate(req.sentences, idempotency_key)
    q = state["queue"]

    cached = await q.get_result(idempotency_key)
    if cached:
        counters["replays"] += 1
        response.status_code = 200
        return dict(key=idempotency_key, status="succeeded", **cached)

    info = await q.stream_info()
    if info["messages"] >= MAX_STREAM_DEPTH:
        counters["rejected"] += 1
        raise HTTPException(429, "queue full", headers={"Retry-After": "5"})

    ack = await q.publish(idempotency_key, dict(sentences=req.sentences,
                                                key=idempotency_key))
    if ack["duplicate"]:
        response.status_code = 200
        counters["replays"] += 1
    counters["jobs"] += 1
    return dict(key=idempotency_key, seq=ack["seq"],
                status="duplicate" if ack["duplicate"] else "queued")


@app.get("/v1/jobs/{key}")
async def job_status(key: str, x_api_key: str | None = Header(None)):
    check_key(x_api_key)
    result = await state["queue"].get_result(key)
    if result is None:
        return dict(key=key, status="pending")
    return dict(key=key, status="succeeded", **result)


@app.get("/healthz")
async def healthz():
    return dict(status="ok")


@app.get("/readyz")
async def readyz():
    if not state["ready"]:
        raise HTTPException(503, "draining")
    try:
        await state["queue"].stream_info()
    except Exception:
        raise HTTPException(503, "queue unreachable")
    return dict(status="ready", model_version=state["detector"].version)


@app.get("/metrics")
async def metrics():
    b = state["batcher"].stats
    avg = b["sentences"] / b["batches"] if b["batches"] else 0.0
    rate = counters["positive"] / counters["sentences"] if counters["sentences"] else 0.0
    info = await state["queue"].stream_info()
    lines = [
        f'claim_requests_total{{endpoint="detect"}} {counters["detect"]}',
        f'claim_requests_total{{endpoint="jobs"}} {counters["jobs"]}',
        f"claim_replays_total {counters['replays']}",
        f"claim_rejected_total {counters['rejected']}",
        f"claim_sentences_total {counters['sentences']}",
        f"claim_positive_rate {rate:.4f}",
        f"claim_batch_avg_size {avg:.2f}",
        f"claim_batch_max_size {b['max_seen']}",
        f"claim_stream_messages {info['messages']}",
        f"claim_stream_consumers {info['consumers']}",
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain")
