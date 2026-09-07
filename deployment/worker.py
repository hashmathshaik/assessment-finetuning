import os
import threading
import time
import traceback

from deployment import db
from deployment.model import Detector

POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "0.5"))
HEARTBEAT_SECONDS = float(os.environ.get("HEARTBEAT_SECONDS", "15"))


def run_job(detector: Detector, payload: dict) -> dict:
    sentences = payload["sentences"]
    return dict(results=detector.predict(sentences), n=len(sentences))


def run_with_heartbeat(conn, detector: Detector, job: dict) -> tuple[dict | None, str | None]:
    box: dict = {}

    def work():
        try:
            box["result"] = run_job(detector, job["payload"])
        except Exception:
            box["error"] = traceback.format_exc()

    thread = threading.Thread(target=work, daemon=True)
    thread.start()

    while thread.is_alive():
        thread.join(HEARTBEAT_SECONDS)
        if thread.is_alive() and not db.heartbeat(conn, job["id"], job["lease_id"]):
            return None, "lease lost"

    return box.get("result"), box.get("error")


def main() -> None:
    detector = Detector()
    conn = db.connect()
    print(f"worker up, model {detector.version}", flush=True)

    while True:
        job = db.claim(conn)
        if job is None:
            time.sleep(POLL_SECONDS)
            continue

        result, error = run_with_heartbeat(conn, detector, job)

        if error == "lease lost":
            print(f"job {job['id']} lease expired mid-flight, dropping", flush=True)
        elif error:
            db.fail(conn, job["id"], job["lease_id"], error)
            print(f"job {job['id']} failed, attempt {job['attempts']}", flush=True)
        elif not db.succeed(conn, job["id"], job["lease_id"], result, detector.version):
            print(f"job {job['id']} lease lost at commit, discarding result", flush=True)


if __name__ == "__main__":
    main()
