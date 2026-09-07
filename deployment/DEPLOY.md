# Running it

## Local

    docker compose up --build

Serves on http://localhost:8080 behind nginx, two api replicas, two workers,
one nats node with JetStream on a named volume.

    curl -X POST localhost:8080/v1/detect \
      -H 'Content-Type: application/json' \
      -H 'Idempotency-Key: demo-1' \
      -d '{"sentences":["The unemployment rate fell to 3.5% last year."]}'

## Fly

NATS runs as its own app so the stream survives API deploys.

    fly apps create claim-detection-nats
    fly volumes create nats_data --size 1 --app claim-detection-nats
    fly deploy --config fly.nats.toml --app claim-detection-nats

    fly apps create claim-detection
    fly secrets set API_KEY=... --app claim-detection
    fly deploy

The API reaches NATS over Fly's private network at
`claim-detection-nats.internal:4222`, so the broker is never exposed publicly.

## Scaling

    fly scale count 4
    fly status

`min_machines_running = 2` keeps two alive so a deploy or a single machine
failure never leaves zero. Soft limit 24 concurrent requests per machine is the
knee from the load test - above that Fly starts another machine rather than
letting latency climb.

    fly machine stop <id>

Traffic moves to the remaining machines; nothing is dropped.
