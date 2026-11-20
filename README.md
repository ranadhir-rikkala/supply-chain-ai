# AI-Powered Supply Chain Optimization

CMPE 272, Fall 2026. Enterprise system that takes in daily ERP feeds, forecasts demand per product and warehouse, flags stockout and overstock risk, and routes reorder and transfer recommendations through role-based approval into purchase orders.

## Dashboard (dashboard/)

React 18 + TypeScript + Vite. Keycloak sign-in with PKCE, role-aware pages for overview, supply map with pending transfers, risks with forecast charts, approvals, orders, audit and admin. Served by Caddy over HTTPS with the API and Keycloak behind the same origin.

```bash
cd dashboard
npm ci
npm run dev:mock      # demo data, pick a demo user; no backend needed
npm run dev           # real mode: proxies /api to :8080 and /auth to :8180
npm run lint && npm test && npm run build
npm run e2e           # Playwright smoke tests (demo data); set E2E_BASE_URL and DEMO_USER_PASSWORD to run against a live stack
docker compose up -d caddy   # https://localhost
```

The mock API in `src/mocks` enforces the same approval rules as the platform API, and `src/api/types.ts` lists the exact response shapes the dashboard expects.

## Data platform (`pipeline/`)

The data platform produces deterministic ERP feeds and loads them through PostgreSQL. Start the dependencies with `cp .env.example .env && make up`, install the package with `make venv`, then run `make generate` and `make load`. `scgen` accepts `--start`, `--end`, and `--seed`; the default seed is 272 and gzip output uses a fixed timestamp for byte-identical reruns. `scetl load-master` validates manifests, schemas, types, dates, keys, and quality gates; rejected files and rows are retained under `data/quarantine/`. Set `DATABASE_URL` to load accepted catalog and fact rows into the Flyway-owned schema. `scetl worker` starts the RabbitMQ consumer with manual acknowledgements and bounded retry/DLQ handling.

The schema is available to Forecasting and the Platform API as soon as `flyway` completes. Contract tables include `core.dim_date`, product/supplier/warehouse dimensions, partitioned sales and inventory facts, purchase-order events/receipts, `ops.etl_*`, and `eval.v_quality_gate_check`.

Run `make test` for deterministic generator, defect-rate, feed-header, validation-line, checksum, quarantine, and rerun tests. Set `TEST_DATABASE_URL` and run `pytest -m integration` after `make up` to verify that every planted defect matches `eval.v_quality_gate_check` in PostgreSQL.
## Forecasting (`pipeline/forecast/`)
One global XGBoost model (Tweedie objective) forecasts 28 days of daily demand for all 4,000 product and warehouse series, one week at a time. Every model is backtested on the last 184 days against a seasonal-naive baseline and must pass a quality gate (weekly WAPE below 25% and below the baseline) before it can become ACTIVE. The design and results are in [ADR 0005](docs/adr/0005-demand-forecasting.md): weekly WAPE 14.9% against 22.5% for the baseline, better in every category and warehouse.

Set up and run locally after the data platform steps above (`make up`, `make generate`, `make load`):
```bash
make venv                                   # installs the dev, forecast and service extras
export DATABASE_URL=postgresql://supplychain:supplychain@localhost:5432/supplychain
scforecast train --requested-by you         # train, backtest, register; prints weekly and daily WAPE
scforecast activate <version> --reason "first release"   # only gate-passing models; an older version is a rollback
scforecast predict --requested-by you       # writes 112,000 rows to ml.forecast
```
On macOS, XGBoost needs OpenMP: `brew install libomp`.

Run as services (set `FORECAST_SERVICE_TOKEN` and `FORECAST_DB_PASSWORD` in `.env` first):
```bash
docker compose up -d --build forecast-api forecast-worker
curl -H "Authorization: Bearer $FORECAST_SERVICE_TOKEN" -H "Content-Type: application/json" \
     -d '{"requested_by":"you"}' http://127.0.0.1:8001/v1/forecast-runs
```
`forecast-api` serves the internal API from the shared contract on `127.0.0.1:8001` (`/health/live`, `/health/ready`, `/v1/models`, `/v1/models/active`, `/v1/models/{version}`, `POST /v1/models/{version}/activation`, `POST /v1/training-runs`, `POST /v1/forecast-runs`, `GET /v1/forecast-runs/{id}`, `/v1/series/{sku}/{warehouse_id}`, `/metrics`, OpenAPI at `/docs`). Every `/v1` call needs the service bearer token; errors are `application/problem+json` with `requestId`, and `X-Request-ID` is echoed. `forecast-worker` consumes `forecast.jobs` on the `ml` exchange, retries after 1, 2 and 4 seconds, then dead-letters to `forecast.jobs.dlq` and marks the forecast run FAILED. Both connect as the least-privilege role `forecast_service`, which reads `core` and writes `ml`. Model files are stored at `models/demand/<version>/model.ubj` and `metadata.json` under `LOCAL_STORE_ROOT` or the S3 bucket.

Tests: `make test` runs the unit tests, including the proof that changing data after a forecast origin cannot change the forecast. With the stack up, `TEST_DATABASE_URL=postgresql://supplychain:supplychain@localhost:5432/supplychain_test TEST_AMQP_URL=amqp://supplychain:supplychain@localhost:5672/ pytest -m integration` also covers the registry, forecast runs, the API with FastAPI TestClient, and the worker's retry and dead-letter path against real Postgres and RabbitMQ.
