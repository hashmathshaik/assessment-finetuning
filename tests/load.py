import argparse
import asyncio
import statistics
import time

import httpx

BASE = "http://127.0.0.1:8080"
SENTENCE = "The unemployment rate fell to 3.5 percent last year according to the report."


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--requests", type=int, default=500)
    args = ap.parse_args()

    lat, codes = [], {}
    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency)

    async with httpx.AsyncClient(limits=limits) as client:
        async def one(i):
            async with sem:
                t = time.perf_counter()
                try:
                    r = await client.post(f"{BASE}/v1/detect", json={"sentences": [SENTENCE]},
                                          headers={"Idempotency-Key": f"load-{time.time()}-{i}"},
                                          timeout=30)
                    codes[r.status_code] = codes.get(r.status_code, 0) + 1
                except Exception as e:
                    codes[type(e).__name__] = codes.get(type(e).__name__, 0) + 1
                    return
                lat.append((time.perf_counter() - t) * 1000)

        t0 = time.perf_counter()
        await asyncio.gather(*[one(i) for i in range(args.requests)])
        wall = time.perf_counter() - t0

    lat.sort()
    def pct(p):
        return lat[min(len(lat) - 1, int(len(lat) * p / 100))] if lat else 0
    print(f"concurrency {args.concurrency}  requests {args.requests}")
    print(f"throughput  {args.requests / wall:.1f} req/s   wall {wall:.1f}s")
    print(f"p50 {pct(50):.0f}ms  p95 {pct(95):.0f}ms  p99 {pct(99):.0f}ms  "
          f"mean {statistics.mean(lat):.0f}ms" if lat else "no successful requests")
    print("codes", codes)


if __name__ == "__main__":
    asyncio.run(main())
