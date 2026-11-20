"""Train, backtest, gate and register a model in one step."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np

from pipeline.forecast.artifacts import save_artifacts
from pipeline.forecast.backtest import BACKTEST_DAYS, run_backtest
from pipeline.forecast.features import SeriesHistory
from pipeline.forecast.metrics import segment_metrics, summarize
from pipeline.forecast.model import train
from pipeline.forecast.panel import SalesPanel, load_panel
from pipeline.forecast.registry import (
    TrainingWindows,
    evaluate_gate,
    register_model,
    save_backtest,
)

logger = logging.getLogger(__name__)


def data_fingerprint(panel: SalesPanel) -> str:
    """SHA-256 of the panel's data, so a model records exactly what it was trained on."""
    digest = hashlib.sha256()
    digest.update(f"{panel.start}:{panel.units.shape}".encode())
    for values in (panel.skus, panel.warehouses):
        digest.update("|".join(values).encode())
    digest.update(np.nan_to_num(panel.units, nan=-1.0).tobytes())
    for values in (panel.stockout, panel.promo):
        digest.update(np.packbits(values).tobytes())
    return digest.hexdigest()


def new_version(fingerprint: str, now: datetime | None = None) -> str:
    """Version name: UTC timestamp plus the start of the data fingerprint."""
    moment = now or datetime.now(UTC)
    return f"{moment:%Y%m%d%H%M%S}-{fingerprint[:8]}"


def train_and_register(
    connection,
    store,
    created_by: str,
    params: dict | None = None,
    rounds: int | None = None,
    panel: SalesPanel | None = None,
) -> dict:
    """Train on everything before the backtest window, backtest, gate, store and register."""
    panel = panel or load_panel(connection)
    categories = tuple(sorted(set(panel.categories)))
    warehouses = tuple(sorted(set(panel.warehouses)))
    history = SeriesHistory.from_panel(panel, categories, warehouses)
    test_start = panel.day_count - BACKTEST_DAYS
    options = {} if rounds is None else {"rounds": rounds}
    model = train(history, categories, warehouses, test_start - 1, params, **options)
    result = run_backtest(model, history)
    rows = segment_metrics(result, panel.categories, panel.warehouses)
    summary = summarize(rows)
    gate = evaluate_gate(summary)
    fingerprint = data_fingerprint(panel)
    version = new_version(fingerprint)
    windows = TrainingWindows(
        train_start=panel.start,
        train_end=panel.date_at(test_start - 1),
        test_start=panel.date_at(test_start),
        test_end=panel.end,
    )
    artifact_key = save_artifacts(
        store,
        version,
        model,
        {
            "data_fingerprint": fingerprint,
            "windows": asdict(windows),
            "metrics": summary,
            "gate": {"passed": gate.passed, "reasons": list(gate.reasons)},
            "created_by": created_by,
        },
    )
    status = register_model(
        connection,
        version,
        summary,
        rows,
        gate,
        fingerprint,
        windows,
        artifact_key,
        {**model.params, "rounds": model.rounds},
        created_by,
    )
    save_backtest(connection, version, result, panel.skus, panel.warehouses, panel.start)
    connection.commit()
    weekly = summary["weekly"]
    logger.info(
        "model trained",
        extra={
            "model_version": version,
            "status": status,
            "weekly_wape": weekly["wape"],
            "baseline_weekly_wape": weekly["baseline_wape"],
        },
    )
    return {
        "modelVersion": version,
        "status": status,
        "passedGate": gate.passed,
        "gateReasons": list(gate.reasons),
        "weekly": {"wape": weekly["wape"], "baselineWape": weekly["baseline_wape"]},
        "daily": {
            "wape": summary["daily"]["wape"],
            "baselineWape": summary["daily"]["baseline_wape"],
        },
    }
