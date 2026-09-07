import asyncio
import json
import os

from deployment.model import Detector
from deployment.queue import Queue

BATCH = int(os.environ.get("WORKER_FETCH", "16"))
TIMEOUT = float(os.environ.get("WORKER_TIMEOUT", "2"))


async def main() -> None:
    detector = Detector()
    q = await Queue().connect()
    sub = await q.subscribe()
    print(f"worker up, model {detector.version}", flush=True)

    while True:
        try:
            msgs = await sub.fetch(BATCH, timeout=TIMEOUT)
        except asyncio.TimeoutError:
            continue

        payloads = [json.loads(m.data) for m in msgs]
        flat = [s for p in payloads for s in p["sentences"]]
        results = await asyncio.to_thread(detector.predict, flat)

        offset = 0
        for msg, p in zip(msgs, payloads):
            chunk = results[offset:offset + len(p["sentences"])]
            offset += len(p["sentences"])
            await q.put_result(p["key"], dict(results=chunk,
                                              model_version=detector.version,
                                              threshold=detector.threshold))
            await msg.ack()


if __name__ == "__main__":
    asyncio.run(main())
