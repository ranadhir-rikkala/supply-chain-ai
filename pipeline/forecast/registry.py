"""Model registry: quality gate, registration, activation and rollback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from psycopg.types.json import Jsonb

from pipeline.forecast.backtest import BacktestResult

GATE_MAX_WEEKLY_WAPE = 0.25
REGISTRY_LOCK_KEY = 272_010


class ModelNotFound(LookupError):
    """Raised when a model version does not exist."""


class QualityGateFailed(RuntimeError):
    """Raised when activating a model that did not pass the quality gate."""


@dataclass(frozen=True)
class GateDecision:
    """Whether a model passed the gate and, if not, why."""

    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TrainingWindows:
    """Training and test date ranges of a model version."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date


def evaluate_gate(summary: dict, max_weekly_wape: float = GATE_MAX_WEEKLY_WAPE) -> GateDecision:
    """Pass when weekly WAPE is below the limit and below the seasonal-naive baseline."""
    weekly = summary.get("weekly", {})
    model_wape = weekly.get("wape")
    baseline_wape = weekly.get("baseline_wape")
    reasons = []
    if model_wape is None:
        reasons.append("weekly WAPE is missing")
    elif model_wape >= max_weekly_wape:
        reasons.append(f"weekly WAPE {model_wape:.4f} is not below {max_weekly_wape:.2f}")
    if model_wape is not None and baseline_wape is not None and model_wape >= baseline_wape:
        reasons.append(
            f"weekly WAPE {model_wape:.4f} is not below the baseline {baseline_wape:.4f}"
        )
    return GateDecision(passed=not reasons, reasons=tuple(reasons))


def _record_status(cursor, version: str, old: str | None, new: str, by: str, reason: str) -> None:
    cursor.execute(
        "INSERT INTO ml.model_status_history "
        "(model_version, from_status, to_status, changed_by, reason) VALUES (%s, %s, %s, %s, %s)",
        (version, old, new, by, reason),
    )


def register_model(
    connection,
    version: str,
    summary: dict,
    metric_rows: list[dict],
    gate: GateDecision,
    fingerprint: str,
    windows: TrainingWindows,
    artifact_key: str,
    params: dict,
    created_by: str,
) -> str:
    """Store a model version with its metrics; failing models are stored as REJECTED."""
    status = "CANDIDATE" if gate.passed else "REJECTED"
    metrics = {**summary, "gate": {"passed": gate.passed, "reasons": list(gate.reasons)}}
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO ml.model_version (model_version, status, metrics, passed_gate, "
            "data_fingerprint, train_start, train_end, test_start, test_end, artifact_key, params) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                version,
                status,
                Jsonb(metrics),
                gate.passed,
                fingerprint,
                windows.train_start,
                windows.train_end,
                windows.test_start,
                windows.test_end,
                artifact_key,
                Jsonb(params),
            ),
        )
        cursor.executemany(
            "INSERT INTO ml.backtest_metric (model_version, grain, segment_type, segment_value, "
            "wape, mape, baseline_wape, baseline_mape, observations) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (
                    version,
                    row["grain"],
                    row["segment_type"],
                    row["segment_value"],
                    row["wape"],
                    row["mape"],
                    row["baseline_wape"],
                    row["baseline_mape"],
                    row["observations"],
                )
                for row in metric_rows
            ],
        )
        reason = "passed quality gate" if gate.passed else "; ".join(gate.reasons)
        _record_status(cursor, version, None, status, created_by, reason)
    return status


def save_backtest(
    connection,
    version: str,
    result: BacktestResult,
    skus: np.ndarray,
    warehouses: np.ndarray,
    start: date,
) -> int:
    """Write the backtest rows of a model version with COPY."""
    observed = ~np.isnan(result.actual)
    columns = (
        "model_version, origin_date, sku, warehouse_id, target_date, horizon, "
        "actual, yhat, baseline"
    )
    with (
        connection.cursor() as cursor,
        cursor.copy(f"COPY ml.backtest ({columns}) FROM STDIN") as copy,
    ):
        for index in np.flatnonzero(observed):
            series = result.series[index]
            copy.write_row(
                (
                    version,
                    start + timedelta(days=int(result.origin_day[index])),
                    skus[series],
                    warehouses[series],
                    start + timedelta(days=int(result.target_day[index])),
                    int(result.horizon[index]),
                    float(result.actual[index]),
                    float(result.yhat[index]),
                    float(result.baseline[index]),
                )
            )
    return int(observed.sum())


def activate_model(connection, version: str, requested_by: str, reason: str) -> dict:
    """Make a gate-passing version ACTIVE and retire the previous one; this is also the rollback."""
    with connection.transaction(), connection.cursor() as cursor:
        # Serialize activations so two concurrent requests cannot fight over the single ACTIVE slot.
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", (REGISTRY_LOCK_KEY,))
        cursor.execute(
            "SELECT status, passed_gate FROM ml.model_version WHERE model_version = %s FOR UPDATE",
            (version,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ModelNotFound(version)
        status, passed_gate = row
        if not passed_gate:
            raise QualityGateFailed(f"model {version} did not pass the quality gate")
        if status == "ACTIVE":
            return {"modelVersion": version, "status": "ACTIVE", "previous": None}
        cursor.execute(
            "SELECT model_version FROM ml.model_version WHERE status = 'ACTIVE' FOR UPDATE"
        )
        previous = cursor.fetchone()
        if previous:
            cursor.execute(
                "UPDATE ml.model_version SET status = 'RETIRED' WHERE model_version = %s",
                (previous[0],),
            )
            _record_status(
                cursor, previous[0], "ACTIVE", "RETIRED", requested_by, f"replaced by {version}"
            )
        cursor.execute(
            "UPDATE ml.model_version SET status = 'ACTIVE', activated_at = now() "
            "WHERE model_version = %s",
            (version,),
        )
        _record_status(cursor, version, status, "ACTIVE", requested_by, reason)
    return {
        "modelVersion": version,
        "status": "ACTIVE",
        "previous": previous[0] if previous else None,
    }


MODEL_COLUMNS = (
    "model_version, status, metrics, passed_gate, data_fingerprint, train_start, train_end, "
    "test_start, test_end, artifact_key, params, created_at, activated_at"
)


def _model_dict(cursor, row) -> dict:
    return {column.name: value for column, value in zip(cursor.description, row, strict=True)}


def list_models(connection) -> list[dict]:
    """All model versions, newest first."""
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT {MODEL_COLUMNS} FROM ml.model_version ORDER BY created_at DESC")
        return [_model_dict(cursor, row) for row in cursor.fetchall()]


def active_model(connection) -> dict | None:
    """The ACTIVE model version, or None."""
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT {MODEL_COLUMNS} FROM ml.model_version WHERE status = 'ACTIVE'")
        row = cursor.fetchone()
        return _model_dict(cursor, row) if row else None


def get_model(connection, version: str) -> dict:
    """A model version with its segment metrics and status history."""
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT {MODEL_COLUMNS} FROM ml.model_version WHERE model_version = %s", (version,)
        )
        row = cursor.fetchone()
        if row is None:
            raise ModelNotFound(version)
        model = _model_dict(cursor, row)
        cursor.execute(
            "SELECT grain, segment_type, segment_value, wape, mape, baseline_wape, baseline_mape, "
            "observations FROM ml.backtest_metric WHERE model_version = %s "
            "ORDER BY grain, segment_type, segment_value",
            (version,),
        )
        model["segment_metrics"] = [_model_dict(cursor, r) for r in cursor.fetchall()]
        cursor.execute(
            "SELECT from_status, to_status, changed_by, reason, changed_at "
            "FROM ml.model_status_history WHERE model_version = %s ORDER BY changed_at, id",
            (version,),
        )
        model["status_history"] = [_model_dict(cursor, r) for r in cursor.fetchall()]
    return model
