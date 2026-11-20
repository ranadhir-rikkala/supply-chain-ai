-- Schema ml: model registry, backtests and forecasts (Teammate B).
CREATE SCHEMA IF NOT EXISTS ml;

CREATE TABLE ml.model_version (
    model_version text PRIMARY KEY,
    status text NOT NULL DEFAULT 'CANDIDATE' CHECK (status IN ('CANDIDATE', 'ACTIVE', 'REJECTED', 'RETIRED')),
    metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
    passed_gate boolean NOT NULL DEFAULT false,
    data_fingerprint text NOT NULL,
    train_start date NOT NULL,
    train_end date NOT NULL,
    test_start date NOT NULL,
    test_end date NOT NULL,
    artifact_key text NOT NULL,
    params jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    activated_at timestamptz,
    CHECK (train_start <= train_end AND train_end < test_start AND test_start <= test_end),
    -- Only a model that passed the quality gate can ever be ACTIVE.
    CHECK (status <> 'ACTIVE' OR passed_gate)
);
-- At most one ACTIVE model; activating another one means retiring the current one first.
CREATE UNIQUE INDEX model_version_single_active ON ml.model_version ((true)) WHERE status = 'ACTIVE';

CREATE TABLE ml.model_status_history (
    id bigserial PRIMARY KEY,
    model_version text NOT NULL REFERENCES ml.model_version ON DELETE CASCADE,
    from_status text,
    to_status text NOT NULL,
    changed_by text NOT NULL,
    reason text,
    changed_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX model_status_history_version ON ml.model_status_history (model_version, changed_at);

CREATE TABLE ml.backtest (
    model_version text NOT NULL REFERENCES ml.model_version ON DELETE CASCADE,
    origin_date date NOT NULL,
    sku text NOT NULL REFERENCES core.dim_product,
    warehouse_id text NOT NULL REFERENCES core.dim_warehouse,
    target_date date NOT NULL,
    horizon smallint NOT NULL CHECK (horizon BETWEEN 1 AND 28),
    actual double precision NOT NULL,
    yhat double precision NOT NULL,
    baseline double precision NOT NULL,
    PRIMARY KEY (model_version, sku, warehouse_id, target_date)
);

CREATE TABLE ml.backtest_metric (
    model_version text NOT NULL REFERENCES ml.model_version ON DELETE CASCADE,
    grain text NOT NULL CHECK (grain IN ('DAILY', 'WEEKLY')),
    segment_type text NOT NULL CHECK (segment_type IN ('OVERALL', 'CATEGORY', 'WAREHOUSE')),
    segment_value text NOT NULL,
    wape double precision,
    mape double precision,
    baseline_wape double precision,
    baseline_mape double precision,
    observations bigint NOT NULL,
    PRIMARY KEY (model_version, grain, segment_type, segment_value)
);

CREATE TABLE ml.forecast_run (
    run_id uuid PRIMARY KEY,
    model_version text REFERENCES ml.model_version,
    origin_date date,
    horizon_days smallint NOT NULL DEFAULT 28 CHECK (horizon_days BETWEEN 1 AND 28),
    status text NOT NULL DEFAULT 'QUEUED' CHECK (status IN ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED')),
    requested_by text NOT NULL,
    requested_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    finished_at timestamptz,
    rows_written bigint,
    error text
);
CREATE INDEX forecast_run_succeeded ON ml.forecast_run (finished_at DESC) WHERE status = 'SUCCEEDED';

CREATE TABLE ml.forecast (
    run_id uuid NOT NULL REFERENCES ml.forecast_run ON DELETE CASCADE,
    sku text NOT NULL REFERENCES core.dim_product,
    warehouse_id text NOT NULL REFERENCES core.dim_warehouse,
    target_date date NOT NULL,
    horizon smallint NOT NULL CHECK (horizon BETWEEN 1 AND 28),
    yhat double precision NOT NULL CHECK (yhat >= 0),
    model_version text NOT NULL REFERENCES ml.model_version,
    PRIMARY KEY (run_id, sku, warehouse_id, target_date)
);

-- The forecast of the latest SUCCEEDED run, which the platform API reads.
CREATE OR REPLACE VIEW ml.v_current_forecast AS
WITH latest_run AS (
    SELECT run_id
    FROM ml.forecast_run
    WHERE status = 'SUCCEEDED'
    ORDER BY finished_at DESC, run_id DESC
    LIMIT 1
)
SELECT f.run_id, f.sku, f.warehouse_id, f.target_date, f.horizon, f.yhat, f.model_version
FROM ml.forecast f
JOIN latest_run l ON l.run_id = f.run_id;
