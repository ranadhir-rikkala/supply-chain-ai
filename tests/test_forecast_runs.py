"""Forecast runs: creation, execution, failure and the current-forecast view."""

import os
import uuid
from datetime import date, timedelta

import pytest
from forecast_fixtures import FAST_PARAMS, ensure_dimensions, synthetic_panel

from pipeline.forecast.artifacts import save_artifacts
from pipeline.forecast.features import SeriesHistory
from pipeline.forecast.model import train
from pipeline.forecast.registry import (
    GateDecision,
    TrainingWindows,
    activate_model,
    register_model,
)
from pipeline.forecast.runs import (
    NoActiveModel,
    RunNotFound,
    create_run,
    execute_run,
    forecast_panel,
    get_run,
    mark_failed,
)
from pipeline.storage import LocalStore

VERSION = "20251231000000-test0001"


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel()


@pytest.fixture(scope="module")
def model(panel):
    categories = tuple(sorted(set(panel.categories)))
    warehouses = tuple(sorted(set(panel.warehouses)))
    history = SeriesHistory.from_panel(panel, categories, warehouses)
    return train(history, categories, warehouses, panel.day_count - 30, FAST_PARAMS, 30)


def test_forecast_panel_uses_only_data_up_to_the_origin(panel, model):
    origin = panel.date_at(600)
    original = forecast_panel(panel, model, origin, 28)
    tampered = synthetic_panel()
    tampered.units[:, 601:] = 999
    assert original.shape == (panel.series_count, 28)
    assert (forecast_panel(tampered, model, origin, 28) == original).all()


def test_forecast_panel_needs_a_year_of_history(panel, model):
    with pytest.raises(ValueError):
        forecast_panel(panel, model, panel.start + timedelta(days=100), 28)


@pytest.fixture
def connection(panel, model, tmp_path):
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")
    import psycopg

    with psycopg.connect(database_url) as connection:
        connection.execute("TRUNCATE ml.model_version CASCADE")
        connection.execute("TRUNCATE ml.forecast_run CASCADE")
        connection.commit()
        ensure_dimensions(connection, panel)
        yield connection
        connection.rollback()
        connection.execute("TRUNCATE ml.forecast_run CASCADE")
        connection.execute("TRUNCATE ml.model_version CASCADE")
        connection.commit()


def _activate(connection, model, store) -> None:
    key = save_artifacts(store, VERSION, model, {})
    windows = TrainingWindows(
        date(2023, 1, 1), date(2024, 6, 30), date(2024, 7, 1), date(2025, 1, 29)
    )
    register_model(
        connection, VERSION, {}, [], GateDecision(True, ()), "fp", windows, key, {}, "tester"
    )
    connection.commit()
    activate_model(connection, VERSION, "tester", "test")


@pytest.mark.integration
def test_run_needs_an_active_model(connection):
    with pytest.raises(NoActiveModel):
        create_run(connection, "tester")


@pytest.mark.integration
def test_run_writes_one_row_per_series_and_day(connection, panel, model, tmp_path):
    store = LocalStore(tmp_path)
    _activate(connection, model, store)
    run = create_run(connection, "tester")
    assert run["status"] == "QUEUED"
    assert run["model_version"] == VERSION
    finished = execute_run(connection, store, run["run_id"], panel=panel)
    assert finished["status"] == "SUCCEEDED"
    assert finished["origin_date"] == panel.end
    assert finished["rows_written"] == panel.series_count * 28
    current = connection.execute(
        "SELECT count(*), min(horizon), max(horizon), min(target_date) FROM ml.v_current_forecast"
    ).fetchone()
    assert current == (panel.series_count * 28, 1, 28, panel.end + timedelta(days=1))
    again = execute_run(connection, store, run["run_id"], panel=panel)
    assert again["rows_written"] == finished["rows_written"]


@pytest.mark.integration
def test_failed_run_is_recorded_and_not_reported_as_current(connection, model, tmp_path):
    store = LocalStore(tmp_path)
    _activate(connection, model, store)
    run = create_run(connection, "tester", horizon_days=7)
    mark_failed(connection, run["run_id"], "dead-lettered after 4 attempts")
    failed = get_run(connection, run["run_id"])
    assert failed["status"] == "FAILED"
    assert failed["error"] == "dead-lettered after 4 attempts"
    assert connection.execute("SELECT count(*) FROM ml.v_current_forecast").fetchone()[0] == 0
    with pytest.raises(RunNotFound):
        get_run(connection, uuid.uuid4())


@pytest.mark.integration
def test_horizon_is_validated(connection, model, tmp_path):
    _activate(connection, model, LocalStore(tmp_path))
    with pytest.raises(ValueError):
        create_run(connection, "tester", horizon_days=29)
