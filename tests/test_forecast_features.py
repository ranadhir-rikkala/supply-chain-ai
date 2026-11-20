"""Feature values, Black Friday dates and the 7-day visibility gap."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pipeline.forecast.features import (
    FEATURE_NAMES,
    SeriesHistory,
    black_friday,
    build_features,
    days_from_nearest_black_friday,
)
from pipeline.forecast.panel import build_panel

START = date(2023, 1, 1)
DAYS = 800


def _history(units: np.ndarray, stockout: np.ndarray | None = None) -> SeriesHistory:
    series_count = units.shape[0]
    return SeriesHistory(
        start=START,
        units=units.astype(np.float32),
        stockout=np.zeros_like(units, dtype=bool) if stockout is None else stockout,
        promo=np.zeros_like(units, dtype=bool),
        price_index=np.ones_like(units, dtype=np.float32),
        category_codes=np.arange(series_count, dtype=np.float32),
        warehouse_codes=np.zeros(series_count, dtype=np.float32),
    )


def _feature(matrix: np.ndarray, name: str) -> np.ndarray:
    return matrix[:, FEATURE_NAMES.index(name)]


def test_black_friday_is_day_after_fourth_thursday_of_november():
    assert black_friday(2023) == date(2023, 11, 24)
    assert black_friday(2024) == date(2024, 11, 29)
    assert black_friday(2025) == date(2025, 11, 28)
    assert days_from_nearest_black_friday(date(2024, 12, 1)) == 2
    assert days_from_nearest_black_friday(date(2024, 11, 20)) == -9
    assert days_from_nearest_black_friday(date(2025, 1, 10)) == 42


def test_feature_matrix_has_one_row_per_series_and_day():
    history = _history(np.ones((3, DAYS)))
    matrix = build_features(history, np.array([400, 401]))
    assert matrix.shape == (6, len(FEATURE_NAMES))
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(_feature(matrix, "category"), [0, 1, 2, 0, 1, 2])


def test_lags_and_rolling_windows_end_seven_days_back():
    ramp = np.arange(DAYS, dtype=np.float32)[None, :]
    matrix = build_features(_history(ramp), np.array([500]))
    assert _feature(matrix, "lag_7")[0] == 493
    assert _feature(matrix, "lag_28")[0] == 472
    assert _feature(matrix, "lag_364")[0] == 136
    assert _feature(matrix, "rolling_mean_7")[0] == pytest.approx(490)
    assert _feature(matrix, "rolling_mean_28")[0] == pytest.approx(479.5)
    assert _feature(matrix, "rolling_mean_91")[0] == pytest.approx(448)
    assert _feature(matrix, "rolling_std_28")[0] == pytest.approx(
        np.std(np.arange(466, 494)), rel=1e-4
    )
    assert _feature(matrix, "same_week_last_year_mean")[0] == pytest.approx(136)


def test_features_ignore_everything_after_the_visible_cutoff():
    rng = np.random.default_rng(7)
    units = rng.poisson(5, size=(4, DAYS)).astype(np.float32)
    target_day = 600
    original = build_features(_history(units), np.array([target_day]))
    changed_units = units.copy()
    changed_units[:, target_day - 6 :] = 1000
    changed_stockout = np.zeros_like(units, dtype=bool)
    changed_stockout[:, target_day - 6 :] = True
    changed = build_features(_history(changed_units, changed_stockout), np.array([target_day]))
    np.testing.assert_array_equal(original, changed)


def test_missing_days_are_skipped_in_rolling_means():
    units = np.full((1, DAYS), 4.0, dtype=np.float32)
    units[0, 480:490] = np.nan
    matrix = build_features(_history(units), np.array([500]))
    assert _feature(matrix, "rolling_mean_28")[0] == pytest.approx(4.0)
    assert np.isnan(_feature(matrix, "lag_14")[0])


def test_recent_stockout_share_counts_the_last_visible_28_days():
    stockout = np.zeros((1, DAYS), dtype=bool)
    stockout[0, 466:473] = True
    matrix = build_features(_history(np.ones((1, DAYS)), stockout), np.array([500]))
    assert _feature(matrix, "recent_stockout_share")[0] == pytest.approx(0.25)


def test_early_days_have_missing_long_lags_instead_of_errors():
    matrix = build_features(_history(np.ones((1, DAYS))), np.array([3]))
    assert np.isnan(_feature(matrix, "lag_7")[0])
    assert np.isnan(_feature(matrix, "rolling_mean_7")[0])
    assert np.isnan(_feature(matrix, "recent_stockout_share")[0])


def test_calendar_features_describe_the_target_day():
    matrix = build_features(
        _history(np.ones((1, DAYS))), np.array([(date(2024, 11, 29) - START).days])
    )
    assert _feature(matrix, "day_of_week")[0] == 4
    assert _feature(matrix, "month")[0] == 11
    assert _feature(matrix, "year")[0] == 2024
    assert _feature(matrix, "days_from_black_friday")[0] == 0


def test_history_from_panel_pads_future_days_and_encodes_segments():
    panel = build_panel(
        pd.DataFrame({"sku": ["BEV-0001"], "category": ["BEV"], "unit_price": [2.0]}),
        pd.DataFrame({"warehouse_id": ["WH-CHI"]}),
        pd.DataFrame(
            {
                "feed_date": ["2024-01-01"],
                "sku": ["BEV-0001"],
                "warehouse_id": ["WH-CHI"],
                "units_sold": [3],
                "unit_price": [2.0],
            }
        ),
        pd.DataFrame(columns=["feed_date", "sku", "warehouse_id", "on_hand", "stockout_flag"]),
    )
    history = SeriesHistory.from_panel(panel, ("SNK", "BEV"), ("WH-CHI",), extra_days=7)
    assert history.units.shape == (1, 8)
    assert history.units[0, 0] == 3
    assert np.isnan(history.units[0, 1:]).all()
    assert history.category_codes[0] == 1
    assert history.warehouse_codes[0] == 0
