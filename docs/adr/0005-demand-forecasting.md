# ADR 0005: One global XGBoost model with a backtest quality gate

- Status: Accepted
- Owner: Teammate B (forecasting)
- Date: 2026-11-20

## Context

The platform needs a 28-day daily demand forecast for every product and warehouse
(500 SKUs x 8 warehouses = 4,000 series). The Platform API turns the forecast into
stockout and overstock risk, so a bad model leads straight to bad purchase orders.
We need a forecast that is clearly better than a naive rule, a way to prove it
before it goes live, and a safe way to roll back.

Constraints:

- Sales are censored on stockout days: units sold understate demand.
- About 1% of feed rows are quarantined, so some days are missing.
- The model must run on a laptop and on one small EC2 instance, without a GPU.
- The Platform API reads forecasts from Postgres and calls our service over HTTP.

## Decision

### Model

One global gradient-boosted tree model (native XGBoost API, `reg:tweedie`,
variance power 1.2, 400 rounds, depth 8) trained across all 4,000 series.
Tweedie suits non-negative, right-skewed counts with some zeros. A single global
model shares seasonal patterns across series and warehouses, which per-series
models with three years of data cannot learn reliably.

Features, all computed from data at least 7 days before the target day:

| Group | Features |
|---|---|
| Lags | units sold 7, 14, 21, 28 and 364 days earlier |
| Rolling | mean over 7, 28 and 91 days, standard deviation over 28 days, all ending 7 days back |
| Seasonality | mean of the same week last year, days from the nearest Black Friday |
| Supply | share of stockout days in the last 28 visible days |
| Calendar | day of week, month, year |
| Commercial | promo flag, price index (price over list price) |
| Segment | category and warehouse as native categorical features |

### Recursive forecast, one week at a time

Because every feature looks at least 7 days back, the model predicts a whole
week from known data. The forecaster then writes that week's predictions into the
history and predicts the next week, four times for 28 days. This keeps one model
for all horizons instead of 28 direct models.

### Training data

Stockout days and missing days are left out of training, because sales on those
days understate demand. They still appear in the features (for example, the
stockout share), so the model can learn how recent stockouts affect sales.

### Hiding the future

At each forecast origin, every value after the origin is hidden: sales, stockout
flags, promotions and prices. Promotions are hidden too because no promotion plan
feed exists. The promo feature still helps in training, where it explains promo
spikes instead of mistaking them for higher base demand. Unit tests change every
value after the origin and assert that the forecast does not change
(`tests/test_forecast_model.py`, `tests/test_forecast_backtest.py`).

### Baseline and backtest

The baseline is seasonal naive: the same weekday 364 days earlier, falling back
one more week when that day is missing. The model is trained on everything before
the last 184 days. The last 184 days are split into non-overlapping 28-day blocks
(six full blocks and one 16-day block). Each block is forecast from its own origin
with only the data known on that day.

Metrics are WAPE and MAPE at daily and weekly grain, overall and per category and
warehouse. Weekly values are sums over complete 7-day weeks within each block.
Stockout and missing days are left out for both the model and the baseline, so
neither is scored against censored sales.

### Quality gate and registry

A model passes the gate when its weekly WAPE is below 25% and below the baseline.
Passing models are registered as `CANDIDATE`, failing ones as `REJECTED`. Only a
passing model can become `ACTIVE`; Postgres enforces this with a check constraint,
and a partial unique index allows only one `ACTIVE` model. Activating an older
passing version is the rollback. Every status change is written to
`ml.model_status_history` with who made it and why.

Artifacts go to object storage at `models/demand/<version>/model.ubj` and
`metadata.json`. The metadata holds the data fingerprint, windows, metrics and a
SHA-256 of the model file, which is checked on every load.

### Serving

`scforecast serve` runs an internal FastAPI service (port 8000, host 8001) with a
service bearer token, `X-Request-ID` propagation, RFC 9457 problem+json errors,
Prometheus `/metrics` and a `psycopg_pool` pool with connection checks. A stopped
database returns 503 with `Retry-After`, never 500, and the service recovers by
itself when the database is back. Forecast and training jobs go through RabbitMQ
(exchange `ml`, queue `forecast.jobs`) with retries at 1, 2 and 4 seconds and a
dead-letter queue. A dead-lettered forecast run is marked `FAILED`.

## Results

Model `20261120010603-a78dd8ce`. It was trained on 2023-01-01 to 2025-06-30 and
backtested on 2025-07-01 to 2025-12-31, using seed 272 data. WAPE in percent;
lower is better.

| Segment | Weekly WAPE | Baseline weekly | Daily WAPE | Baseline daily |
|---|---:|---:|---:|---:|
| **Overall** | **14.9** | **22.5** | **33.3** | **47.6** |
| BEV | 15.4 | 21.9 | 33.7 | 47.1 |
| DAI | 14.8 | 22.2 | 33.3 | 47.6 |
| ELE | 15.0 | 23.1 | 33.2 | 48.1 |
| FRZ | 14.8 | 22.3 | 33.3 | 47.2 |
| HHC | 14.6 | 23.1 | 33.1 | 47.9 |
| OUT | 15.8 | 22.7 | 33.9 | 47.0 |
| PAP | 14.7 | 22.7 | 33.1 | 47.7 |
| PCR | 14.8 | 22.7 | 33.1 | 47.6 |
| PHA | 15.1 | 23.4 | 33.3 | 48.7 |
| SCH | 15.5 | 23.9 | 33.7 | 48.9 |
| SNK | 14.1 | 21.3 | 32.7 | 46.6 |
| WIN | 15.8 | 25.8 | 33.7 | 50.7 |
| WH-01 Los Angeles | 15.4 | 22.5 | 33.7 | 47.4 |
| WH-02 Seattle | 14.9 | 22.7 | 33.2 | 47.6 |
| WH-03 Chicago | 14.4 | 22.0 | 33.0 | 47.5 |
| WH-04 Dallas | 14.8 | 22.2 | 33.2 | 47.6 |
| WH-05 New York | 14.6 | 22.3 | 33.1 | 48.0 |
| WH-06 Miami | 15.3 | 23.0 | 33.5 | 47.5 |
| WH-07 Denver | 15.3 | 22.9 | 33.5 | 47.5 |
| WH-08 Phoenix | 14.9 | 22.5 | 33.4 | 47.4 |

The model passes the gate and beats the baseline in every category and warehouse.
Weekly MAPE is 15.5% against 23.0%. Daily WAPE is much higher than weekly because
day-to-day noise mostly cancels out within a week; the Platform API plans
replenishment over lead times of days to weeks, so the weekly number is the one
that matters for decisions. The forecast is 2.3% below actual sales overall.

Operational numbers on a 10-core laptop: loading the panel takes about 14 s,
training 73 s, the backtest 5 s, and a 28-day forecast run that writes 112,000 rows
about 22 s.

## Alternatives considered

- **Per-series statistical models (ETS, ARIMA).** 4,000 separate fits, weak on
  short series, and no shared learning across categories. Rejected.
- **Direct multi-horizon models (one per horizon day).** Avoids feeding
  predictions back in, but needs 28 models and more training time. Weekly recursion
  with a 7-day feature gap gives one model and keeps feeding back to four steps.
- **Deep learning (DeepAR, TFT).** Needs a GPU to train in reasonable time and is
  harder to explain in a demo. Not worth it at this data size.
- **Retraining on the full history after the gate.** Would add six months of
  data, but the deployed model would not be the one that was evaluated. We keep
  the evaluated model so that the gate result stays honest.

## Consequences

- Forecasts never use future information, and tests prove it.
- A bad model cannot reach production, and a rollback is one API call with an
  audit record.
- Planned promotions are not used at forecast time; adding a promotion plan feed
  would be the next improvement.
- A series with no clean history (for example `SKU-0001` at `WH-01`, whose row
  carries a planted defect every day) still gets a forecast. It comes from the
  category and warehouse patterns rather than its own history.
- The worker image is about 900 MB, mostly numpy, pandas, scipy and XGBoost.
