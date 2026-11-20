"""WAPE and MAPE formulas and segment metrics."""

import numpy as np
import pytest

from pipeline.forecast.backtest import BacktestResult
from pipeline.forecast.metrics import mape, segment_metrics, summarize, wape

CATEGORIES = np.array(["BEV", "SNK"], dtype=object)
WAREHOUSES = np.array(["WH-01", "WH-01"], dtype=object)


def _result(horizon_days: int = 14, usable: np.ndarray | None = None) -> BacktestResult:
    series = np.repeat([0, 1], horizon_days)
    horizon = np.tile(np.arange(1, horizon_days + 1), 2)
    actual = np.where(series == 0, 10.0, 4.0)
    yhat = np.where(series == 0, 12.0, 4.0)
    yhat[(series == 0) & (horizon % 2 == 0)] = 8.0
    baseline = np.where(series == 0, 5.0, 6.0)
    return BacktestResult(
        series=series,
        origin_day=np.full(series.size, 500),
        target_day=500 + horizon,
        horizon=horizon,
        actual=actual,
        yhat=yhat,
        baseline=baseline,
        usable=np.ones(series.size, dtype=bool) if usable is None else usable,
    )


def _find(rows, grain, segment_type, segment_value):
    return next(
        row
        for row in rows
        if (row["grain"], row["segment_type"], row["segment_value"])
        == (grain, segment_type, segment_value)
    )


def test_wape_and_mape_formulas():
    actual = np.array([10.0, 0.0, 5.0])
    predicted = np.array([8.0, 1.0, 5.0])
    assert wape(actual, predicted) == pytest.approx(3 / 15)
    assert mape(actual, predicted) == pytest.approx((0.2 + 0.0) / 2)
    assert wape(np.zeros(3), predicted) is None
    assert mape(np.zeros(3), predicted) is None


def test_daily_errors_cancel_inside_a_week():
    rows = segment_metrics(_result(), CATEGORIES, WAREHOUSES)
    daily = _find(rows, "DAILY", "OVERALL", "ALL")
    weekly = _find(rows, "WEEKLY", "OVERALL", "ALL")
    assert daily["wape"] == pytest.approx(28 / 196)
    assert weekly["wape"] < 0.05
    assert daily["baseline_wape"] == pytest.approx((14 * 5 + 14 * 2) / 196)
    assert daily["observations"] == 28
    assert weekly["observations"] == 4


def test_metrics_are_split_by_category_and_warehouse():
    rows = segment_metrics(_result(), CATEGORIES, WAREHOUSES)
    assert _find(rows, "DAILY", "CATEGORY", "SNK")["wape"] == 0
    assert _find(rows, "DAILY", "CATEGORY", "BEV")["wape"] == pytest.approx(0.2)
    assert _find(rows, "WEEKLY", "WAREHOUSE", "WH-01")["observations"] == 4
    assert {(r["grain"], r["segment_type"]) for r in rows} == {
        (grain, segment)
        for grain in ("DAILY", "WEEKLY")
        for segment in ("OVERALL", "CATEGORY", "WAREHOUSE")
    }


def test_partial_weeks_are_left_out_of_weekly_metrics():
    rows = segment_metrics(_result(horizon_days=10), CATEGORIES, WAREHOUSES)
    assert _find(rows, "WEEKLY", "OVERALL", "ALL")["observations"] == 2
    assert _find(rows, "DAILY", "OVERALL", "ALL")["observations"] == 20


def test_unusable_days_are_excluded():
    usable = np.ones(28, dtype=bool)
    usable[:14] = False
    rows = segment_metrics(_result(usable=usable), CATEGORIES, WAREHOUSES)
    assert _find(rows, "DAILY", "OVERALL", "ALL")["wape"] == 0
    assert not any(row["segment_value"] == "BEV" for row in rows)


def test_summary_matches_the_model_version_metrics_contract():
    summary = summarize(segment_metrics(_result(), CATEGORIES, WAREHOUSES))
    assert set(summary) == {"daily", "weekly"}
    assert {"wape", "baseline_wape", "mape", "baseline_mape"} <= set(summary["weekly"])
    assert set(summary["weekly"]["by_category"]) == {"BEV", "SNK"}
    assert set(summary["daily"]["by_warehouse"]) == {"WH-01"}
