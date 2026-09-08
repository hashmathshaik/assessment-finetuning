# Running it

## Local

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m finetuning.train --model bert --max-length 64 --epochs 3
    .venv/bin/python -m finetuning.export
    docker compose up --build

The 418MB weights aren't in git, so the image build needs them at
`artifacts/models/bert-seed42/` first.

nginx on :8080, two api replicas, two workers, one nats node with JetStream on a
named volume.

    curl -X POST localhost:8080/v1/detect \
      -H 'Content-Type: application/json' \
      -H 'Idempotency-Key: demo-1' \
      -d '{"sentences":["The unemployment rate fell to 3.5% last year."]}'

Send the same Idempotency-Key twice and the second call returns the stored
answer with `replayed: true` rather than running the model again.

## Fly

Three apps. nats is separate so redeploying the service never touches the
stream, and the worker is separate so it can scale independently of the api.

### nats

    fly apps create claim-detection-nats
    fly volumes create nats_data --size 1 --app claim-detection-nats --region ewr
    fly deploy --config fly.nats.toml --app claim-detection-nats

### api

    fly apps create claim-detection
    fly secrets set API_KEY=... --app claim-detection
    fly deploy

### Authentication

The service refuses to start with auth silently disabled. Either set `API_KEY`,
or set `ALLOW_ANONYMOUS=1` to serve openly on purpose. `fly.toml` sets
`ALLOW_ANONYMOUS=1` because this is a public demo endpoint — drop that line and
set the secret to lock it down.

Keys are compared with `hmac.compare_digest`, not `==`, so a wrong key takes the
same time to reject regardless of how much of it is right.

### worker

Without this, `/v1/jobs` accepts work that nothing ever picks up. `/v1/detect`
still answers, because the api runs the model itself on that path.

    fly apps create claim-detection-worker
    fly deploy --config fly.worker.toml --app claim-detection-worker

Both build from the same Dockerfile with a different command, so a single build
can't produce two different model versions. They are still two separate `fly
deploy` calls though, and the GitHub workflow only deploys the api — deploy both
after retraining or they will drift.

The api reaches nats at `claim-detection-nats.internal:4222` over Fly's private
network. `fly.nats.toml` deliberately has no `[[services]]` block — that is the
public-edge construct, and adding one would have flyctl allocate a public IP for
an unauthenticated broker.

## Scaling

    fly scale count 4 --app claim-detection
    fly scale count 4 --app claim-detection-worker
    fly status --app claim-detection

`fly.toml` sets a soft limit of 24 concurrent requests per machine. That comes
from the load test — throughput plateaus around 100 req/s and the knee is at 32
concurrent, past which p50 climbs from 300ms to 490ms for 8% more throughput.
Above the soft limit Fly starts another machine instead of letting latency grow.

`min_machines_running = 2` keeps two api machines alive so a deploy or a single
machine failure never leaves zero.

    fly machine stop <id> --app claim-detection

Traffic moves to the remaining machine. Nothing is dropped.

## Verifying it

    python tests/chaos.py -n 500 --kill-every 4
    python tests/load.py --concurrency 32 --requests 240

`chaos.py` submits jobs while killing workers at random and checks every accepted
key reaches a terminal state. It kills and restarts host processes via `pgrep -f
deployment.worker`, so run it against services started directly on the host, not
against compose containers and not against Fly:

    .bin/nats-server -js -sd .natsdata -p 4222 &
    .venv/bin/uvicorn deployment.app:app --port 8080 &
    .venv/bin/python -m deployment.worker &
    .venv/bin/python -m deployment.worker &

`load.py` reports throughput against whatever it is pointed at. The numbers in
the README are against uvicorn directly — through nginx you measure the 50 r/s
per-IP rate limit instead.

## Known gap

One nats node is a single point of failure. While it is down the api cannot
accept anything, because accepting a request *means* writing to it. Running it
with `replicas=3` uses Raft, so losing a node is survivable — that needs three
machines and a volume each.
