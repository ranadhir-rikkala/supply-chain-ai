"""WAPE and MAPE at daily and weekly grain, overall and per category and warehouse."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pipeline.forecast.backtest import WEEK_DAYS, BacktestResult

GRAINS = ("DAILY", "WEEKLY")
SEGMENT_TYPES = ("OVERALL", "CATEGORY", "WAREHOUSE")
OVERALL = "ALL"


def wape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """Weighted absolute percentage error, or None when there were no actual sales."""
    total = float(np.sum(actual))
    if total <= 0:
        return None
    return float(np.sum(np.abs(actual - predicted)) / total)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """Mean absolute percentage error over days with positive actual sales."""
    positive = actual > 0
    if not positive.any():
        return None
    return float(np.mean(np.abs(actual[positive] - predicted[positive]) / actual[positive]))


def _frame(result: BacktestResult, categories: np.ndarray, warehouses: np.ndarray) -> pd.DataFrame:
    usable = result.usable
    series = result.series[usable]
    return pd.DataFrame(
        {
            "series": series,
            "origin_day": result.origin_day[usable],
            "week": (result.horizon[usable] - 1) // WEEK_DAYS,
            "horizon": result.horizon[usable],
            "actual": result.actual[usable].astype(np.float64),
            "yhat": result.yhat[usable].astype(np.float64),
            "baseline": result.baseline[usable].astype(np.float64),
            "CATEGORY": categories[series],
            "WAREHOUSE": warehouses[series],
        }
    )


def _weekly(daily: pd.DataFrame, result: BacktestResult) -> pd.DataFrame:
    block_lengths = pd.Series(result.horizon).groupby(result.origin_day).max()
    complete_weeks = block_lengths // WEEK_DAYS
    # Partial weeks at the end of the short last block would distort weekly WAPE.
    keep = daily["week"] < daily["origin_day"].map(complete_weeks)
    keys = ["series", "origin_day", "week", "CATEGORY", "WAREHOUSE"]
    return daily[keep].groupby(keys, as_index=False)[["actual", "yhat", "baseline"]].sum()


def _row(grain: str, segment_type: str, segment_value: str, frame: pd.DataFrame) -> dict:
    actual = frame["actual"].to_numpy()
    yhat = frame["yhat"].to_numpy()
    baseline = frame["baseline"].to_numpy()
    return {
        "grain": grain,
        "segment_type": segment_type,
        "segment_value": segment_value,
        "wape": wape(actual, yhat),
        "mape": mape(actual, yhat),
        "baseline_wape": wape(actual, baseline),
        "baseline_mape": mape(actual, baseline),
        "observations": len(frame),
    }


def segment_metrics(
    result: BacktestResult, categories: np.ndarray, warehouses: np.ndarray
) -> list[dict]:
    """Metric rows for every grain and segment, with the baseline next to the model."""
    daily = _frame(result, categories, warehouses)
    rows = []
    for grain, frame in (("DAILY", daily), ("WEEKLY", _weekly(daily, result))):
        rows.append(_row(grain, "OVERALL", OVERALL, frame))
        for segment_type in ("CATEGORY", "WAREHOUSE"):
            for value, group in frame.groupby(segment_type, sort=True):
                rows.append(_row(grain, segment_type, str(value), group))
    return rows


def _clean(value: float | None) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(value, 6)


def summarize(rows: list[dict]) -> dict:
    """Nest metric rows into the ml.model_version metrics JSON from the contract."""
    summary: dict = {}
    for row in rows:
        grain = row["grain"].lower()
        values = {
            key: _clean(row[key]) for key in ("wape", "mape", "baseline_wape", "baseline_mape")
        }
        values["observations"] = row["observations"]
        if row["segment_type"] == "OVERALL":
            summary.setdefault(grain, {}).update(values)
        else:
            segment = "by_" + row["segment_type"].lower()
            summary.setdefault(grain, {}).setdefault(segment, {})[row["segment_value"]] = values
    return summary
