CREATE TYPE job_status AS ENUM ('queued', 'running', 'succeeded', 'dead_letter');

CREATE TABLE IF NOT EXISTS jobs (
    id              BIGSERIAL PRIMARY KEY,
    idempotency_key TEXT        NOT NULL UNIQUE,
    status          job_status  NOT NULL DEFAULT 'queued',
    payload         JSONB       NOT NULL,
    result          JSONB,
    model_version   TEXT,
    lease_id        UUID,
    locked_until    TIMESTAMPTZ,
    attempts        INT         NOT NULL DEFAULT 0,
    max_attempts    INT         NOT NULL DEFAULT 5,
    run_after       TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_error      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS jobs_claimable
    ON jobs (run_after)
    WHERE status IN ('queued', 'running');
