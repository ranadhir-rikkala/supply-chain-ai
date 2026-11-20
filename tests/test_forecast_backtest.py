"""Backtest blocks, seasonal-naive baseline and leakage checks."""

import numpy as np
import pytest
from forecast_fixtures import CATEGORIES, FAST_PARAMS, FAST_ROUNDS, WAREHOUSES, synthetic_history

from pipeline.forecast.backtest import (
    BACKTEST_DAYS,
    backtest_origins,
    run_backtest,
    seasonal_naive,
)
from pipeline.forecast.model import train

DAYS = 760
TEST_START = DAYS - BACKTEST_DAYS


@pytest.fixture(scope="module")
def history():
    return synthetic_history(days=DAYS)


@pytest.fixture(scope="module")
def model(history):
    return train(history, CATEGORIES, WAREHOUSES, TEST_START - 1, FAST_PARAMS, FAST_ROUNDS)


def test_origins_cover_the_last_184_days_in_28_day_blocks():
    origins = backtest_origins(1096)
    assert origins[0] == (1096 - 184 - 1, 28)
    assert [horizon for _, horizon in origins] == [28, 28, 28, 28, 28, 28, 16]
    covered = [origin + 1 + offset for origin, horizon in origins for offset in range(horizon)]
    assert covered == list(range(1096 - 184, 1096))


def test_origins_need_a_year_of_history():
    with pytest.raises(ValueError):
        backtest_origins(400)


def test_seasonal_naive_repeats_the_same_weekday_364_days_earlier(history):
    forecast = seasonal_naive(history, 600, 28)
    np.testing.assert_array_equal(forecast, history.units[:, 601 - 364 : 629 - 364])


def test_seasonal_naive_falls_back_a_week_when_last_year_is_missing():
    history = synthetic_history(days=DAYS)
    history.units[0, 601 - 364] = np.nan
    forecast = seasonal_naive(history, 600, 28)
    assert forecast[0, 0] == history.units[0, 601 - 371]


def test_backtest_rows_cover_every_series_and_test_day(history, model):
    result = run_backtest(model, history)
    series_count = history.units.shape[0]
    assert len(result) == series_count * BACKTEST_DAYS
    assert result.target_day.min() == TEST_START
    assert result.target_day.max() == DAYS - 1
    assert set(np.unique(result.horizon)) == set(range(1, 29))
    np.testing.assert_array_equal(result.actual, history.units[result.series, result.target_day])
    assert not result.usable[history.stockout[result.series, result.target_day]].any()


def test_changing_data_after_an_origin_cannot_change_its_forecast(history, model):
    original = run_backtest(model, history)
    first_origin = backtest_origins(DAYS)[0][0]
    tampered = synthetic_history(days=DAYS)
    tampered.units[:, first_origin + 1 :] *= 7
    tampered.stockout[:, first_origin + 1 :] = ~tampered.stockout[:, first_origin + 1 :]
    tampered.promo[:, first_origin + 1 :] = True
    changed = run_backtest(model, tampered)
    first_block = original.origin_day == first_origin
    np.testing.assert_array_equal(original.yhat[first_block], changed.yhat[first_block])
    np.testing.assert_array_equal(original.baseline[first_block], changed.baseline[first_block])
    assert not np.array_equal(original.actual[first_block], changed.actual[first_block])
