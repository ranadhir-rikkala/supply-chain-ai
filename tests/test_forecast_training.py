"""Training pipeline: fingerprint, version name and registration."""

import os
from datetime import UTC, datetime

import pytest
from forecast_fixtures import FAST_PARAMS, ensure_dimensions, synthetic_panel

from pipeline.forecast.artifacts import load_artifacts, metadata_key
from pipeline.forecast.backtest import BACKTEST_DAYS
from pipeline.forecast.registry import get_model
from pipeline.forecast.training import data_fingerprint, new_version, train_and_register
from pipeline.storage import LocalStore


def test_fingerprint_changes_when_the_data_changes():
    panel = synthetic_panel()
    same = synthetic_panel()
    changed = synthetic_panel()
    changed.units[0, 10] += 1
    assert data_fingerprint(panel) == data_fingerprint(same)
    assert data_fingerprint(panel) != data_fingerprint(changed)


def test_version_is_sortable_by_time_and_names_the_data():
    version = new_version("abcdef1234", datetime(2025, 12, 31, 8, 5, 9, tzinfo=UTC))
    assert version == "20251231080509-abcdef12"


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


@pytest.mark.integration
def test_training_registers_metrics_backtest_and_artifacts(connection, tmp_path):
    panel = synthetic_panel()
    ensure_dimensions(connection, panel)
    store = LocalStore(tmp_path)
    outcome = train_and_register(connection, store, "tester", FAST_PARAMS, 60, panel=panel)
    version = outcome["modelVersion"]
    model = get_model(connection, version)
    assert model["status"] == outcome["status"]
    assert model["passed_gate"] is outcome["passedGate"]
    assert model["test_start"] == panel.date_at(panel.day_count - BACKTEST_DAYS)
    assert model["train_end"] < model["test_start"]
    assert {"weekly", "daily", "gate"} <= set(model["metrics"])
    assert {row["segment_type"] for row in model["segment_metrics"]} == {
        "OVERALL",
        "CATEGORY",
        "WAREHOUSE",
    }
    assert store.exists(metadata_key(version))
    _, metadata = load_artifacts(store, version)
    assert metadata["data_fingerprint"] == model["data_fingerprint"]
    count = connection.execute(
        "SELECT count(*) FROM ml.backtest WHERE model_version = %s", (version,)
    ).fetchone()[0]
    assert count == panel.series_count * BACKTEST_DAYS
