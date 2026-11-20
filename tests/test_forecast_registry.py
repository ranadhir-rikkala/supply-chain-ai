"""Quality gate, registration, activation and rollback."""

import os
from datetime import date

import pytest

from pipeline.forecast.registry import (
    GateDecision,
    ModelNotFound,
    QualityGateFailed,
    TrainingWindows,
    activate_model,
    active_model,
    evaluate_gate,
    get_model,
    list_models,
    register_model,
)

WINDOWS = TrainingWindows(date(2023, 1, 1), date(2025, 6, 30), date(2025, 7, 1), date(2025, 12, 31))
METRIC_ROWS = [
    {
        "grain": "WEEKLY",
        "segment_type": "OVERALL",
        "segment_value": "ALL",
        "wape": 0.2,
        "mape": 0.25,
        "baseline_wape": 0.28,
        "baseline_mape": 0.33,
        "observations": 100,
    }
]


def _summary(wape: float | None, baseline: float | None) -> dict:
    return {"weekly": {"wape": wape, "baseline_wape": baseline}, "daily": {}}


@pytest.mark.parametrize(
    ("wape", "baseline", "passed"),
    [(0.20, 0.28, True), (0.26, 0.40, False), (0.22, 0.21, False), (None, 0.3, False)],
)
def test_quality_gate_needs_weekly_wape_below_25_percent_and_baseline(wape, baseline, passed):
    decision = evaluate_gate(_summary(wape, baseline))
    assert decision.passed is passed
    assert bool(decision.reasons) is not passed


@pytest.fixture
def connection():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")
    import psycopg

    with psycopg.connect(database_url) as connection:
        connection.execute("TRUNCATE ml.model_version CASCADE")
        connection.commit()
        yield connection
        connection.rollback()
        connection.execute("TRUNCATE ml.model_version CASCADE")
        connection.commit()


def _register(connection, version: str, passed: bool) -> str:
    gate = GateDecision(passed, () if passed else ("weekly WAPE too high",))
    status = register_model(
        connection,
        version,
        _summary(0.2 if passed else 0.4, 0.3),
        METRIC_ROWS,
        gate,
        "fingerprint",
        WINDOWS,
        f"models/demand/{version}/model.ubj",
        {"eta": 0.06},
        "tester",
    )
    connection.commit()
    return status


@pytest.mark.integration
def test_registration_status_follows_the_gate(connection):
    assert _register(connection, "v-pass", True) == "CANDIDATE"
    assert _register(connection, "v-fail", False) == "REJECTED"
    model = get_model(connection, "v-fail")
    assert model["metrics"]["gate"]["passed"] is False
    assert model["status_history"][0]["to_status"] == "REJECTED"
    assert model["segment_metrics"][0]["wape"] == pytest.approx(0.2)
    assert {m["model_version"] for m in list_models(connection)} == {"v-pass", "v-fail"}


@pytest.mark.integration
def test_only_gate_passing_models_can_be_activated(connection):
    _register(connection, "v-fail", False)
    with pytest.raises(QualityGateFailed):
        activate_model(connection, "v-fail", "admin", "try it")
    with pytest.raises(ModelNotFound):
        activate_model(connection, "missing", "admin", "try it")
    assert active_model(connection) is None


@pytest.mark.integration
def test_activating_an_older_version_rolls_back_and_is_audited(connection):
    _register(connection, "v1", True)
    _register(connection, "v2", True)
    activate_model(connection, "v1", "admin", "first release")
    result = activate_model(connection, "v2", "admin", "better model")
    assert result["previous"] == "v1"
    assert active_model(connection)["model_version"] == "v2"
    rollback = activate_model(connection, "v1", "admin", "roll back")
    assert rollback["previous"] == "v2"
    assert active_model(connection)["model_version"] == "v1"
    assert get_model(connection, "v2")["status"] == "RETIRED"
    transitions = [
        (row["from_status"], row["to_status"])
        for row in get_model(connection, "v1")["status_history"]
    ]
    assert transitions == [
        (None, "CANDIDATE"),
        ("CANDIDATE", "ACTIVE"),
        ("ACTIVE", "RETIRED"),
        ("RETIRED", "ACTIVE"),
    ]
    assert activate_model(connection, "v1", "admin", "again")["previous"] is None
