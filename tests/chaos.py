import argparse
import asyncio
import random
import subprocess
import time

import httpx

BASE = "http://127.0.0.1:8080"


async def submit(client, key, sentence):
    for attempt in range(6):
        try:
            r = await client.post(f"{BASE}/v1/jobs", json={"sentences": [sentence]},
                                  headers={"Idempotency-Key": key}, timeout=10)
            if r.status_code in (200, 202):
                return True
            if r.status_code == 429:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            return False
        except Exception:
            await asyncio.sleep(0.3 * (attempt + 1))
    return False


def kill_random_worker():
    out = subprocess.run(["pgrep", "-f", "deployment.worker"],
                         capture_output=True, text=True).stdout.split()
    if not out:
        return None
    pid = random.choice(out)
    subprocess.run(["kill", "-9", pid])
    subprocess.Popen(["/bin/sh", "-c",
                      ".venv/bin/python -m deployment.worker >> logs/worker.log 2>&1 &"])
    return pid


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=2000)
    ap.add_argument("--kill-every", type=float, default=3.0)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()

    keys = [f"chaos-{int(time.time())}-{i}" for i in range(args.n)]
    sentence = "In the UK in 2020 there were 48,000 illegal immigrants in hotels."
    accepted, killed, stop = [], [], False

    async def killer():
        while not stop:
            await asyncio.sleep(args.kill_every)
            pid = kill_random_worker()
            if pid:
                killed.append(pid)

    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        task = asyncio.create_task(killer())
        sem = asyncio.Semaphore(args.concurrency)

        async def one(k):
            async with sem:
                if await submit(client, k, sentence):
                    accepted.append(k)

        t0 = time.perf_counter()
        await asyncio.gather(*[one(k) for k in keys])
        submit_secs = time.perf_counter() - t0
        stop = True
        await task

        print(f"submitted {args.n}, accepted {len(accepted)} in {submit_secs:.1f}s, "
              f"killed {len(killed)} workers")

        print("draining...")
        deadline = time.time() + 300
        while time.time() < deadline:
            done = 0
            for k in accepted:
                r = await client.get(f"{BASE}/v1/jobs/{k}", timeout=10)
                if r.json().get("status") == "succeeded":
                    done += 1
            if done == len(accepted):
                break
            print(f"  {done}/{len(accepted)}")
            await asyncio.sleep(5)

        missing = []
        for k in accepted:
            r = await client.get(f"{BASE}/v1/jobs/{k}", timeout=10)
            if r.json().get("status") != "succeeded":
                missing.append(k)

        print()
        print(f"accepted   {len(accepted)}")
        print(f"completed  {len(accepted) - len(missing)}")
        print(f"lost       {len(missing)}")
        print("PASS" if not missing else f"FAIL: {missing[:5]}")


if __name__ == "__main__":
    asyncio.run(main())
