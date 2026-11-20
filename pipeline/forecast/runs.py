"""Forecast runs: queue, execute and record the outcome."""

from __future__ import annotations

import logging
import uuid
from datetime import date, timedelta

from pipeline.forecast.artifacts import load_artifacts
from pipeline.forecast.features import SeriesHistory
from pipeline.forecast.model import HORIZON_DAYS, recursive_forecast
from pipeline.forecast.panel import SalesPanel, load_panel

logger = logging.getLogger(__name__)

RUN_COLUMNS = (
    "run_id, model_version, origin_date, horizon_days, status, requested_by, requested_at, "
    "started_at, finished_at, rows_written, error"
)
MINIMUM_HISTORY_DAYS = 371


class NoActiveModel(RuntimeError):
    """Raised when a run is requested but no model is ACTIVE."""


class RunNotFound(LookupError):
    """Raised when a run id does not exist."""


def _as_dict(cursor, row) -> dict:
    return {column.name: value for column, value in zip(cursor.description, row, strict=True)}


def create_run(
    connection,
    requested_by: str,
    origin_date: date | None = None,
    horizon_days: int = HORIZON_DAYS,
    run_id: uuid.UUID | None = None,
) -> dict:
    """Record a QUEUED run for the ACTIVE model."""
    if not 1 <= horizon_days <= HORIZON_DAYS:
        raise ValueError(f"horizon_days must be between 1 and {HORIZON_DAYS}")
    with connection.cursor() as cursor:
        cursor.execute("SELECT model_version FROM ml.model_version WHERE status = 'ACTIVE'")
        active = cursor.fetchone()
        if active is None:
            raise NoActiveModel("no ACTIVE model version; activate one first")
        cursor.execute(
            "INSERT INTO ml.forecast_run (run_id, model_version, origin_date, horizon_days, "
            f"requested_by) VALUES (%s, %s, %s, %s, %s) RETURNING {RUN_COLUMNS}",
            (run_id or uuid.uuid4(), active[0], origin_date, horizon_days, requested_by),
        )
        run = _as_dict(cursor, cursor.fetchone())
    connection.commit()
    return run


def get_run(connection, run_id: uuid.UUID) -> dict:
    """A forecast run by id."""
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT {RUN_COLUMNS} FROM ml.forecast_run WHERE run_id = %s", (run_id,))
        row = cursor.fetchone()
        if row is None:
            raise RunNotFound(str(run_id))
        return _as_dict(cursor, row)


def mark_failed(connection, run_id: uuid.UUID, error: str) -> None:
    """Mark a run FAILED unless it already succeeded."""
    connection.rollback()
    connection.execute(
        "UPDATE ml.forecast_run SET status = 'FAILED', finished_at = clock_timestamp(), error = %s "
        "WHERE run_id = %s AND status <> 'SUCCEEDED'",
        (error[:2000], run_id),
    )
    connection.commit()


def _claim(connection, run_id: uuid.UUID) -> dict | None:
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE ml.forecast_run SET status = 'RUNNING', started_at = now(), error = NULL "
            # RUNNING is accepted so a retried message can redo a run that crashed midway.
            "WHERE run_id = %s AND status IN ('QUEUED', 'RUNNING') "
            f"RETURNING {RUN_COLUMNS}",
            (run_id,),
        )
        row = cursor.fetchone()
        run = _as_dict(cursor, row) if row else None
    connection.commit()
    return run


def forecast_panel(panel: SalesPanel, model, origin_date: date, horizon_days: int):
    """Forecast every series from origin_date using only data up to that date."""
    if origin_date < panel.start + timedelta(days=MINIMUM_HISTORY_DAYS):
        raise ValueError(f"origin {origin_date} needs at least a year of history")
    visible = panel.through(origin_date)
    history = SeriesHistory.from_panel(visible, model.categories, model.warehouses)
    return recursive_forecast(model, history, visible.day_count - 1, horizon_days)


def execute_run(connection, store, run_id: uuid.UUID, panel: SalesPanel | None = None) -> dict:
    """Run a queued forecast and write one row per series and day."""
    run = _claim(connection, run_id)
    if run is None:
        existing = get_run(connection, run_id)
        logger.info("run already finished", extra={"run_id": str(run_id)})
        return existing
    model, _ = load_artifacts(store, run["model_version"])
    panel = panel or load_panel(connection, end=run["origin_date"])
    origin = run["origin_date"] or panel.end
    horizon = run["horizon_days"]
    values = forecast_panel(panel, model, origin, horizon)
    with connection.cursor() as cursor:
        # Delete first so a retried run never leaves duplicate rows.
        cursor.execute("DELETE FROM ml.forecast WHERE run_id = %s", (run_id,))
        columns = "run_id, sku, warehouse_id, target_date, horizon, yhat, model_version"
        with cursor.copy(f"COPY ml.forecast ({columns}) FROM STDIN") as copy:
            for series in range(values.shape[0]):
                sku, warehouse = panel.skus[series], panel.warehouses[series]
                for step in range(horizon):
                    copy.write_row(
                        (
                            run_id,
                            sku,
                            warehouse,
                            origin + timedelta(days=step + 1),
                            step + 1,
                            float(values[series, step]),
                            run["model_version"],
                        )
                    )
        rows = values.shape[0] * horizon
        cursor.execute(
            # clock_timestamp() because now() is the start of the transaction, not the end of the run.
            "UPDATE ml.forecast_run SET status = 'SUCCEEDED', finished_at = clock_timestamp(), "
            f"origin_date = %s, rows_written = %s WHERE run_id = %s RETURNING {RUN_COLUMNS}",
            (origin, rows, run_id),
        )
        finished = _as_dict(cursor, cursor.fetchone())
    connection.commit()
    logger.info("forecast run succeeded", extra={"run_id": str(run_id), "rows": rows})
    return finished
