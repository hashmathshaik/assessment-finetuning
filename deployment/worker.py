import asyncio
import json
import os
import traceback

from deployment.model import Detector
from deployment.queue import MAX_DELIVER, SYNC_SUBJECT, Queue

FETCH = int(os.environ.get("WORKER_FETCH", "16"))
TIMEOUT = float(os.environ.get("WORKER_TIMEOUT", "2"))
PROGRESS_EVERY = float(os.environ.get("PROGRESS_EVERY", "20"))
RECOVERY_EVERY = float(os.environ.get("RECOVERY_EVERY", "30"))


async def keep_alive(msgs, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=PROGRESS_EVERY)
            return
        except asyncio.TimeoutError:
            for m in msgs:
                try:
                    await m.in_progress()
                except Exception:
                    pass


async def give_up(q: Queue, msg, payload: dict, error: str) -> bool:
    """On the final delivery attempt, record the failure and stop redelivering.

    Without this a message that fails max_deliver times is dropped by JetStream
    with no trace, and /v1/jobs would have returned 202 for work that vanished.
    """
    try:
        delivered = msg.metadata.num_delivered
    except Exception:
        return False
    if delivered < MAX_DELIVER:
        return False
    await q.put_result(payload["key"], dict(status="failed", error=error[-1000:],
                                            attempts=delivered))
    await msg.term()
    print(f"job {payload['key']} dead-lettered after {delivered} attempts", flush=True)
    return True


async def handle(q: Queue, detector: Detector, msgs: list) -> None:
    parsed = []
    for m in msgs:
        try:
            p = json.loads(m.data)
            assert isinstance(p["sentences"], list) and p["key"]
            parsed.append((m, p))
        except Exception:
            await m.term()
            print(f"terminated unparseable message: {m.data[:120]!r}", flush=True)

    todo = []
    for m, p in parsed:
        if await q.get_result(p["key"]) is not None:
            await m.ack()
        else:
            todo.append((m, p))
    if not todo:
        return

    stop = asyncio.Event()
    beat = asyncio.create_task(keep_alive([m for m, _ in todo], stop))
    try:
        flat = [s for _, p in todo for s in p["sentences"]]
        results = await asyncio.to_thread(detector.predict, flat)
    except Exception:
        stop.set(); await beat
        err = traceback.format_exc()
        for m, p in todo:
            if await give_up(q, m, p, err):
                continue
            await m.nak(delay=5)
        print(f"inference failed, redelivering {len(todo)}\n{err}", flush=True)
        return
    stop.set(); await beat

    offset = 0
    for m, p in todo:
        chunk = results[offset:offset + len(p["sentences"])]
        offset += len(p["sentences"])
        try:
            await q.put_result(p["key"], dict(results=chunk, model_version=detector.version,
                                              threshold=detector.threshold))
            await m.ack()
        except Exception:
            err = traceback.format_exc()
            if not await give_up(q, m, p, err):
                await m.nak(delay=5)
            print(f"could not store {p['key']}, will retry", flush=True)


async def recover(q: Queue, detector: Detector) -> None:
    sub = await q.subscribe(durable="sync_recovery", subject=SYNC_SUBJECT, ack_wait=120)
    while True:
        await asyncio.sleep(RECOVERY_EVERY)
        try:
            msgs = await sub.fetch(FETCH, timeout=2)
        except asyncio.TimeoutError:
            continue
        except Exception:
            print(f"recovery fetch failed\n{traceback.format_exc()}", flush=True)
            await asyncio.sleep(5)
            continue
        try:
            await handle(q, detector, msgs)
        except Exception:
            print(f"recovery batch failed\n{traceback.format_exc()}", flush=True)


async def main() -> None:
    detector = Detector()
    q = await Queue().connect()
    sub = await q.subscribe()
    asyncio.create_task(recover(q, detector))
    print(f"worker up, model {detector.version}", flush=True)

    while True:
        try:
            msgs = await sub.fetch(FETCH, timeout=TIMEOUT)
        except asyncio.TimeoutError:
            continue
        except Exception:
            print(f"fetch failed, retrying\n{traceback.format_exc()}", flush=True)
            await asyncio.sleep(1)
            continue
        try:
            await handle(q, detector, msgs)
        except Exception:
            print(f"batch failed\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
