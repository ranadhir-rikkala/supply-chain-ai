"""Forecast API with FastAPI TestClient: auth, errors, outages and every endpoint."""

import os
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient
from forecast_fixtures import FAST_PARAMS, ensure_dimensions, synthetic_panel
from psycopg_pool import ConnectionPool

from pipeline.forecast.api import Settings, create_app
from pipeline.forecast.jobs import JobPublishError
from pipeline.forecast.registry import GateDecision, TrainingWindows, register_model

TOKEN = "test-service-token-123"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
UNREACHABLE_DATABASE = "postgresql://nobody:nothing@127.0.0.1:1/none?connect_timeout=1"
WINDOWS = TrainingWindows(date(2023, 1, 1), date(2024, 6, 30), date(2024, 7, 1), date(2025, 1, 29))


class RecordingPublisher:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.messages = []

    def publish(self, message, request_id=None):
        if self.fail:
            raise JobPublishError("broker down")
        self.messages.append((message, request_id))


def _client(database_url: str, publisher=None) -> TestClient:
    settings = Settings(database_url=database_url, service_token=TOKEN, pool_timeout_seconds=1)
    pool = ConnectionPool(
        database_url,
        min_size=0,
        max_size=2,
        timeout=1,
        open=False,
        check=ConnectionPool.check_connection,
    )
    return TestClient(create_app(settings, pool, publisher or RecordingPublisher()))


@pytest.fixture
def offline():
    with _client(UNREACHABLE_DATABASE) as client:
        yield client


def test_liveness_needs_no_token_and_echoes_request_id(offline):
    response = offline.get("/health/live", headers={"X-Request-ID": "abc-123"})
    assert response.status_code == 200
    assert response.json() == {"status": "UP"}
    assert response.headers["X-Request-ID"] == "abc-123"


def test_unsafe_request_ids_are_replaced(offline):
    response = offline.get("/health/live", headers={"X-Request-ID": "bad id\n<script>"})
    assert response.headers["X-Request-ID"] != "bad id\n<script>"
    assert len(response.headers["X-Request-ID"]) == 36


@pytest.mark.parametrize("header", [None, "Bearer wrong-token", "Basic abc", TOKEN])
def test_api_rejects_missing_or_wrong_tokens_as_problem_json(offline, header):
    headers = {"Authorization": header} if header else {}
    response = offline.get("/v1/models", headers={**headers, "X-Request-ID": "req-1"})
    assert response.status_code == 401
    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers["WWW-Authenticate"] == "Bearer"
    body = response.json()
    assert body["status"] == 401
    assert body["requestId"] == "req-1"
    assert body["instance"] == "/v1/models"


def test_validation_errors_are_problem_json(offline):
    response = offline.post(
        "/v1/forecast-runs", headers=AUTH, json={"requested_by": "pat", "horizon_days": 40}
    )
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["errors"][0]["location"] == ["body", "horizon_days"]


def test_unknown_fields_and_bad_actors_are_rejected(offline):
    extra = offline.post("/v1/training-runs", headers=AUTH, json={"requested_by": "a", "x": 1})
    bad_actor = offline.post("/v1/training-runs", headers=AUTH, json={"requested_by": "a b;"})
    assert extra.status_code == 422
    assert bad_actor.status_code == 422


def test_database_outage_returns_503_not_500(offline):
    ready = offline.get("/health/ready")
    models = offline.get("/v1/models", headers=AUTH)
    for response in (ready, models):
        assert response.status_code == 503
        assert response.headers["content-type"] == "application/problem+json"
        assert response.headers["Retry-After"] == "5"
    assert ready.json()["checks"] == {"database": "DOWN"}


def test_training_run_is_queued_with_a_job_id(offline):
    publisher = RecordingPublisher()
    with _client(UNREACHABLE_DATABASE, publisher) as client:
        response = client.post(
            "/v1/training-runs",
            headers={**AUTH, "X-Request-ID": "train-1"},
            json={"requested_by": "ada.admin"},
        )
    assert response.status_code == 202
    message, request_id = publisher.messages[0]
    assert response.json()["jobId"] == message["job_id"]
    assert message["job_type"] == "train"
    assert request_id == "train-1"


def test_queue_outage_returns_503(offline):
    with _client(UNREACHABLE_DATABASE, RecordingPublisher(fail=True)) as client:
        response = client.post("/v1/training-runs", headers=AUTH, json={"requested_by": "ada"})
    assert response.status_code == 503
    assert response.json()["type"].endswith("queue-unavailable")


def test_metrics_endpoint_counts_requests_by_route(offline):
    offline.get("/v1/models")
    body = offline.get("/metrics").text
    assert 'forecast_api_requests_total{method="GET",route="/v1/models",status="401"}' in body
    assert "forecast_api_request_duration_seconds_bucket" in body


def test_token_must_be_configured(monkeypatch):
    monkeypatch.setenv("FORECAST_SERVICE_TOKEN", "short")
    with pytest.raises(RuntimeError):
        Settings.from_env()


@pytest.fixture
def database_url():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not configured")
    return url


@pytest.fixture
def seeded(database_url):
    import psycopg

    panel = synthetic_panel()
    with psycopg.connect(database_url) as connection:
        connection.execute("TRUNCATE ml.forecast_run, ml.model_version CASCADE")
        connection.commit()
        ensure_dimensions(connection, panel)
        for version, passed in (("v-good", True), ("v-bad", False)):
            gate = GateDecision(passed, () if passed else ("weekly WAPE too high",))
            summary = {"weekly": {"wape": 0.15 if passed else 0.4, "baseline_wape": 0.22}}
            rows = [
                {
                    "grain": "WEEKLY",
                    "segment_type": "CATEGORY",
                    "segment_value": "BEV",
                    "wape": 0.15,
                    "mape": 0.2,
                    "baseline_wape": 0.22,
                    "baseline_mape": 0.3,
                    "observations": 10,
                }
            ]
            register_model(
                connection,
                version,
                summary,
                rows,
                gate,
                "fp",
                WINDOWS,
                f"models/demand/{version}/model.ubj",
                FAST_PARAMS,
                "tester",
            )
        sku, warehouse = panel.skus[0], panel.warehouses[0]
        connection.execute(
            "DELETE FROM core.fact_sales_daily WHERE sku = %s AND warehouse_id = %s",
            (sku, warehouse),
        )
        for offset in range(10):
            connection.execute(
                "INSERT INTO core.fact_sales_daily (feed_date, sale_id, sku, warehouse_id, "
                "units_sold, unit_price, etl_run_id) VALUES (%s, %s, %s, %s, %s, 2.0, "
                "'00000000-0000-0000-0000-000000000009')",
                (
                    date(2026, 1, 1) + timedelta(days=offset),
                    f"S-API-{offset}",
                    sku,
                    warehouse,
                    offset,
                ),
            )
        connection.execute(
            "INSERT INTO ml.backtest VALUES ('v-good', '2025-12-01', %s, %s, '2025-12-02', 1, "
            "5, 4.5, 6)",
            (sku, warehouse),
        )
        connection.commit()
        yield {"sku": sku, "warehouse": warehouse}
        connection.rollback()
        connection.execute("DELETE FROM core.fact_sales_daily WHERE sale_id LIKE 'S-API-%%'")
        connection.execute("TRUNCATE ml.forecast_run, ml.model_version CASCADE")
        connection.commit()


@pytest.fixture
def online(database_url, seeded):
    publisher = RecordingPublisher()
    with _client(database_url, publisher) as client:
        client.publisher = publisher
        yield client


@pytest.mark.integration
def test_ready_reports_the_active_model(online):
    assert online.get("/health/ready").json()["activeModel"] is None
    online.post(
        "/v1/models/v-good/activation",
        headers=AUTH,
        json={"requested_by": "ada", "reason": "go live"},
    )
    assert online.get("/health/ready").json()["activeModel"] == "v-good"


@pytest.mark.integration
def test_model_list_and_detail(online):
    listed = online.get("/v1/models", headers=AUTH).json()["items"]
    assert {model["model_version"] for model in listed} == {"v-good", "v-bad"}
    detail = online.get("/v1/models/v-good", headers=AUTH).json()
    assert detail["weekly_wape"] == 0.15
    assert detail["metrics"]["by_category"]["weekly"][0]["segment_value"] == "BEV"
    assert detail["status_history"][0]["to_status"] == "CANDIDATE"
    assert online.get("/v1/models/active", headers=AUTH).status_code == 404
    missing = online.get("/v1/models/nope", headers=AUTH)
    assert missing.status_code == 404
    assert missing.headers["content-type"] == "application/problem+json"


@pytest.mark.integration
def test_activation_refuses_models_that_failed_the_gate(online):
    body = {"requested_by": "ada.admin", "reason": "promote"}
    refused = online.post("/v1/models/v-bad/activation", headers=AUTH, json=body)
    assert refused.status_code == 409
    assert refused.json()["type"].endswith("quality-gate-failed")
    accepted = online.post("/v1/models/v-good/activation", headers=AUTH, json=body)
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "ACTIVE"
    assert online.get("/v1/models/active", headers=AUTH).json()["model_version"] == "v-good"


@pytest.mark.integration
def test_forecast_run_is_queued_with_location(online):
    no_model = online.post("/v1/forecast-runs", headers=AUTH, json={"requested_by": "pat"})
    assert no_model.status_code == 409
    online.post(
        "/v1/models/v-good/activation",
        headers=AUTH,
        json={"requested_by": "ada", "reason": "go live"},
    )
    response = online.post(
        "/v1/forecast-runs", headers=AUTH, json={"requested_by": "pat.planner", "horizon_days": 14}
    )
    assert response.status_code == 202
    run = response.json()
    assert response.headers["Location"] == f"/v1/forecast-runs/{run['run_id']}"
    assert run["status"] == "QUEUED"
    assert run["horizon_days"] == 14
    assert online.publisher.messages[0][0]["run_id"] == run["run_id"]
    fetched = online.get(response.headers["Location"], headers=AUTH)
    assert fetched.json()["model_version"] == "v-good"
    unknown = online.get("/v1/forecast-runs/00000000-0000-0000-0000-000000000000", headers=AUTH)
    assert unknown.status_code == 404


@pytest.mark.integration
def test_forecast_run_is_failed_when_it_cannot_be_queued(database_url, seeded):
    with _client(database_url, RecordingPublisher(fail=True)) as client:
        client.post(
            "/v1/models/v-good/activation",
            headers=AUTH,
            json={"requested_by": "ada", "reason": "go live"},
        )
        response = client.post("/v1/forecast-runs", headers=AUTH, json={"requested_by": "pat"})
        assert response.status_code == 503
        import psycopg

        with psycopg.connect(database_url) as connection:
            status = connection.execute("SELECT status FROM ml.forecast_run").fetchone()[0]
        assert status == "FAILED"


@pytest.mark.integration
def test_series_returns_history_backtest_and_forecast(online, seeded):
    online.post(
        "/v1/models/v-good/activation",
        headers=AUTH,
        json={"requested_by": "ada", "reason": "go live"},
    )
    path = f"/v1/series/{seeded['sku']}/{seeded['warehouse']}?history_days=5"
    body = online.get(path, headers=AUTH).json()
    assert [row["units_sold"] for row in body["history"]] == [5, 6, 7, 8, 9]
    assert set(body["history"][0]) == {
        "date",
        "units_sold",
        "promo_flag",
        "stockout_flag",
        "on_hand",
    }
    assert body["backtest"] == [{"date": "2025-12-02", "actual": 5.0, "yhat": 4.5, "baseline": 6.0}]
    assert body["forecast"] == []
    assert online.get("/v1/series/NOPE-1/WH-T0", headers=AUTH).status_code == 404
    assert online.get(f"{path[:-1]}0", headers=AUTH).status_code == 422
