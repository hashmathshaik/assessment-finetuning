# Running it

## Local

    docker compose up --build

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
    fly secrets set API_KEY=... --app claim-detection      # optional
    fly deploy

### worker

Without this, `/v1/jobs` accepts work that nothing ever picks up. `/v1/detect`
still answers, because the api runs the model itself on that path.

    fly apps create claim-detection-worker
    fly deploy --config fly.worker.toml --app claim-detection-worker

Both api and worker run the same image with a different command, so they can't
drift onto different model versions.

The api reaches nats over Fly's private network at
`claim-detection-nats.internal:4222`, so the broker has no public listener.

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

`chaos.py` submits jobs while killing workers at random and checks every
accepted key reaches a terminal state. It expects the api on localhost:8080 and
workers started from this repo, so run it against docker compose rather than Fly.

## Known gap

One nats node is a single point of failure. While it is down the api cannot
accept anything, because accepting a request *means* writing to it. Running it
with `replicas=3` uses Raft, so losing a node is survivable — that needs three
machines and a volume each.
