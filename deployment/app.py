import os
import time

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from deployment import db
from deployment.model import MAX_BATCH, MAX_CHARS, Detector

API_KEY = os.environ.get("API_KEY")

app = FastAPI(title="claim-detection")
detector: Detector | None = None
conn = None
counters = dict(detect=0, jobs=0, positive=0, sentences=0)


@app.on_event("startup")
def startup() -> None:
    global detector, conn
    detector = Detector()
    conn = db.connect()


class DetectRequest(BaseModel):
    sentences: list[str] = Field(min_length=1)


def check_key(key: str | None) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(401, "invalid api key")


def validate(sentences: list[str]) -> None:
    if len(sentences) > MAX_BATCH:
        raise HTTPException(413, f"at most {MAX_BATCH} sentences per request")
    total = sum(len(s) for s in sentences)
    if total > MAX_CHARS * MAX_BATCH or any(len(s) > MAX_CHARS for s in sentences):
        raise HTTPException(413, f"sentences must be under {MAX_CHARS} characters")


@app.post("/v1/detect")
def detect(req: DetectRequest, x_api_key: str | None = Header(None)):
    check_key(x_api_key)
    validate(req.sentences)
    started = time.perf_counter()
    results = detector.predict(req.sentences)
    counters["detect"] += 1
    counters["sentences"] += len(results)
    counters["positive"] += sum(r["is_claim"] for r in results)
    return dict(results=results, model_version=detector.version,
                threshold=detector.threshold,
                latency_ms=round((time.perf_counter() - started) * 1000, 2))


@app.post("/v1/jobs", status_code=202)
def submit(req: DetectRequest, response: Response,
           idempotency_key: str = Header(...), x_api_key: str | None = Header(None)):
    check_key(x_api_key)
    validate(req.sentences)
    if db.queue_depth(conn) >= db.MAX_QUEUE_DEPTH:
        raise HTTPException(429, "queue full", headers={"Retry-After": "5"})
    row = db.enqueue(conn, idempotency_key, dict(sentences=req.sentences))
    if not row["inserted"]:
        response.status_code = 200
    counters["jobs"] += 1
    return dict(job_id=row["id"], status=row["status"])


@app.get("/v1/jobs/{job_id}")
def job_status(job_id: int, x_api_key: str | None = Header(None)):
    check_key(x_api_key)
    row = db.get_job(conn, job_id)
    if row is None:
        raise HTTPException(404, "no such job")
    return row


@app.get("/healthz")
def healthz():
    return dict(status="ok")


@app.get("/readyz")
def readyz():
    if detector is None:
        raise HTTPException(503, "model not loaded")
    try:
        db.queue_depth(conn)
    except Exception:
        raise HTTPException(503, "database unreachable")
    return dict(status="ready", model_version=detector.version)


@app.get("/metrics")
def metrics():
    rate = counters["positive"] / counters["sentences"] if counters["sentences"] else 0.0
    lines = [
        f'claim_requests_total{{endpoint="detect"}} {counters["detect"]}',
        f'claim_requests_total{{endpoint="jobs"}} {counters["jobs"]}',
        f'claim_sentences_total {counters["sentences"]}',
        f"claim_positive_rate {rate:.4f}",
        f"claim_queue_depth {db.queue_depth(conn)}",
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain")
