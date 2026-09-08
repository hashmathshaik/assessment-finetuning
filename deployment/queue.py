import json
import os

import asyncio
import hashlib

import nats
from nats.js.api import (ConsumerConfig, KeyValueConfig, RetentionPolicy,
                         StreamConfig)
from nats.js.errors import (BadRequestError, KeyNotFoundError,
                            KeyWrongLastSequenceError)

NATS_URL = os.environ.get("NATS_URL", "nats://127.0.0.1:4222")
STREAM = "CLAIMS"
SUBJECT = "claims.jobs"
SYNC_SUBJECT = "claims.sync"
KV_BUCKET = "claim_results"
REPLICAS = int(os.environ.get("NATS_REPLICAS", "1"))
DEDUPE_WINDOW_S = int(os.environ.get("DEDUPE_WINDOW_S", "7200"))
MAX_DELIVER = int(os.environ.get("MAX_DELIVER", "5"))
ACK_WAIT_S = int(os.environ.get("ACK_WAIT_S", "300"))
MAX_AGE_S = int(os.environ.get("STREAM_MAX_AGE_S", "86400"))


class Queue:
    def __init__(self):
        self.nc = None
        self.js = None
        self.kv = None

    async def connect(self, attempts: int = 30, delay: float = 2.0):
        for i in range(attempts):
            try:
                self.nc = await nats.connect(NATS_URL, max_reconnect_attempts=-1)
                break
            except Exception as exc:
                if i == attempts - 1:
                    raise
                print(f"nats not reachable ({exc}), retry {i + 1}/{attempts}", flush=True)
                await asyncio.sleep(delay)
        self.js = self.nc.jetstream()
        cfg = StreamConfig(
            name=STREAM, subjects=[SUBJECT, SYNC_SUBJECT], num_replicas=REPLICAS,
            duplicate_window=DEDUPE_WINDOW_S, max_age=MAX_AGE_S,
            retention=RetentionPolicy.LIMITS,
        )
        try:
            await self.js.add_stream(cfg)
        except BadRequestError:
            # Stream exists with a different config. Adding is not idempotent, so
            # reconcile instead of refusing to start - otherwise any config change
            # bricks every deploy until someone deletes the stream by hand.
            await self.js.update_stream(cfg)
        try:
            self.kv = await self.js.create_key_value(
                KeyValueConfig(bucket=KV_BUCKET, replicas=REPLICAS))
        except BadRequestError:
            self.kv = await self.js.key_value(KV_BUCKET)
        return self

    async def close(self):
        if self.nc:
            await self.nc.drain()

    async def publish(self, key: str, payload: dict, subject: str = SUBJECT) -> dict:
        ack = await self.js.publish(subject, json.dumps(payload).encode(),
                                    headers={"Nats-Msg-Id": key})
        return dict(seq=ack.seq, duplicate=bool(ack.duplicate))

    @staticmethod
    def kv_key(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    async def get_result(self, key: str) -> dict | None:
        try:
            entry = await self.kv.get(self.kv_key(key))
        except KeyNotFoundError:
            return None
        return json.loads(entry.value)

    async def put_result(self, key: str, value: dict) -> bool:
        try:
            await self.kv.create(self.kv_key(key), json.dumps(value).encode())
            return True
        except (KeyWrongLastSequenceError, Exception) as exc:
            if isinstance(exc, KeyWrongLastSequenceError) or "wrong last sequence" in str(exc):
                return False
            raise

    async def stream_info(self) -> dict:
        info = await self.js.stream_info(STREAM)
        return dict(messages=info.state.messages, bytes=info.state.bytes,
                    consumers=info.state.consumer_count)

    async def backlog(self) -> int:
        try:
            info = await self.js.consumer_info(STREAM, "workers")
            return int(info.num_pending + info.num_ack_pending)
        except Exception:
            return 0

    async def subscribe(self, durable: str = "workers", subject: str = SUBJECT,
                        ack_wait: int = ACK_WAIT_S):
        return await self.js.pull_subscribe(
            subject, durable=durable,
            config=ConsumerConfig(ack_wait=ack_wait, max_deliver=MAX_DELIVER))
