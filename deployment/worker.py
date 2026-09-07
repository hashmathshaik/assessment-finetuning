import os
import time
import traceback

from deployment import db
from deployment.model import Detector

POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "0.5"))
HEARTBEAT_SECONDS = float(os.environ.get("HEARTBEAT_SECONDS", "15"))


def run_job(detector: Detector, payload: dict) -> dict:
    sentences = payload["sentences"]
    return dict(results=detector.predict(sentences), n=len(sentences))


def main() -> None:
    detector = Detector()
    conn = db.connect()
    print(f"worker up, model {detector.version}", flush=True)

    while True:
        job = db.claim(conn)
        if job is None:
            time.sleep(POLL_SECONDS)
            continue

        job_id, lease = job["id"], job["lease_id"]
        try:
            result = run_job(detector, job["payload"])
            if not db.succeed(conn, job_id, lease, result, detector.version):
                print(f"job {job_id} lease lost, discarding result", flush=True)
        except Exception:
            db.fail(conn, job_id, lease, traceback.format_exc())
            print(f"job {job_id} failed attempt {job['attempts']}", flush=True)


if __name__ == "__main__":
    main()
