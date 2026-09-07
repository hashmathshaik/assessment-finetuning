import asyncio
import os
import time

MAX_BATCH = int(os.environ.get("BATCH_MAX_SIZE", "32"))
MAX_WAIT_MS = float(os.environ.get("BATCH_MAX_WAIT_MS", "5"))


class Batcher:
    def __init__(self, predict_fn, max_batch: int = MAX_BATCH,
                 max_wait_ms: float = MAX_WAIT_MS):
        self.predict_fn = predict_fn
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self.queue: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.stats = dict(batches=0, sentences=0, max_seen=0)

    async def start(self) -> None:
        self.task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        while not self.queue.empty():
            _, future = self.queue.get_nowait()
            if not future.done():
                future.set_exception(RuntimeError("server shutting down"))

    async def submit(self, sentences: list[str]) -> list[dict]:
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        await self.queue.put((sentences, future))
        return await future

    async def _collect(self) -> list:
        first = await self.queue.get()
        items = [first]
        size = len(first[0])
        deadline = time.perf_counter() + self.max_wait
        while size < self.max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            items.append(item)
            size += len(item[0])
        return items

    async def _loop(self) -> None:
        while True:
            items = await self._collect()
            flat = [s for sentences, _ in items for s in sentences]
            try:
                results = await asyncio.to_thread(self.predict_fn, flat)
            except Exception as exc:
                for _, future in items:
                    if not future.done():
                        future.set_exception(exc)
                continue

            self.stats["batches"] += 1
            self.stats["sentences"] += len(flat)
            self.stats["max_seen"] = max(self.stats["max_seen"], len(flat))

            offset = 0
            for sentences, future in items:
                chunk = results[offset:offset + len(sentences)]
                offset += len(sentences)
                if not future.done():
                    future.set_result(chunk)
