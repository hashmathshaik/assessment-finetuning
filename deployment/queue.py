import json
import os

import nats
from nats.js.api import KeyValueConfig, StreamConfig
from nats.js.errors import KeyNotFoundError

NATS_URL = os.environ.get("NATS_URL", "nats://127.0.0.1:4222")
STREAM = "CLAIMS"
SUBJECT = "claims.jobs"
KV_BUCKET = "claim_results"
REPLICAS = int(os.environ.get("NATS_REPLICAS", "1"))
DEDUPE_WINDOW_S = int(os.environ.get("DEDUPE_WINDOW_S", "7200"))
MAX_DELIVER = int(os.environ.get("MAX_DELIVER", "5"))


class Queue:
    def __init__(self):
        self.nc = None
        self.js = None
        self.kv = None

    async def connect(self):
        self.nc = await nats.connect(NATS_URL, max_reconnect_attempts=-1)
        self.js = self.nc.jetstream()
        await self.js.add_stream(StreamConfig(
            name=STREAM, subjects=[SUBJECT], num_replicas=REPLICAS,
            duplicate_window=DEDUPE_WINDOW_S,
        ))
        self.kv = await self.js.create_key_value(KeyValueConfig(
            bucket=KV_BUCKET, replicas=REPLICAS))
        return self

    async def close(self):
        if self.nc:
            await self.nc.drain()

    async def publish(self, key: str, payload: dict) -> dict:
        ack = await self.js.publish(SUBJECT, json.dumps(payload).encode(),
                                    headers={"Nats-Msg-Id": key})
        return dict(seq=ack.seq, duplicate=bool(ack.duplicate))

    async def get_result(self, key: str) -> dict | None:
        try:
            entry = await self.kv.get(key)
        except KeyNotFoundError:
            return None
        return json.loads(entry.value)

    async def put_result(self, key: str, value: dict) -> None:
        await self.kv.put(key, json.dumps(value).encode())

    async def stream_info(self) -> dict:
        info = await self.js.stream_info(STREAM)
        return dict(messages=info.state.messages, bytes=info.state.bytes,
                    consumers=info.state.consumer_count)

    async def subscribe(self, durable: str = "workers"):
        return await self.js.pull_subscribe(SUBJECT, durable=durable)
