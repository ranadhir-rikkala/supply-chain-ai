"""Recursive XGBoost forecaster, including the proof that future data cannot change a forecast."""

import numpy as np
import pytest
from forecast_fixtures import (
    CATEGORIES,
    FAST_PARAMS,
    FAST_ROUNDS,
    WAREHOUSES,
    synthetic_history,
)

from pipeline.forecast.model import (
    FIRST_TRAINING_DAY,
    HORIZON_DAYS,
    DemandModel,
    recursive_forecast,
    train,
    training_rows,
    visible_history,
)

ORIGIN = 700


@pytest.fixture(scope="module")
def history():
    return synthetic_history()


@pytest.fixture(scope="module")
def model(history):
    return train(history, CATEGORIES, WAREHOUSES, ORIGIN, FAST_PARAMS, FAST_ROUNDS)


def test_training_rows_skip_stockout_and_missing_days(history):
    with_gap = synthetic_history()
    with_gap.units[0, 200] = np.nan
    features, labels = training_rows(with_gap, 0, 300)
    days = 300 - FIRST_TRAINING_DAY + 1
    series_count = with_gap.units.shape[0]
    excluded = with_gap.stockout[:, FIRST_TRAINING_DAY:301].sum() + 1
    assert labels.shape == (days * series_count - excluded,)
    assert features.shape[0] == labels.shape[0]
    assert not np.isnan(labels).any()


def test_forecast_has_one_value_per_series_and_day(history, model):
    forecast = recursive_forecast(model, history, ORIGIN)
    assert forecast.shape == (history.units.shape[0], HORIZON_DAYS)
    assert (forecast >= 0).all()
    assert np.isfinite(forecast).all()


def test_forecast_tracks_demand_better_than_last_year(history, model):
    forecast = recursive_forecast(model, history, ORIGIN)
    actual = history.units[:, ORIGIN + 1 : ORIGIN + 1 + HORIZON_DAYS]
    last_year = history.units[:, ORIGIN + 1 - 364 : ORIGIN + 1 - 364 + HORIZON_DAYS]
    model_error = np.abs(actual - forecast).sum() / actual.sum()
    baseline_error = np.abs(actual - last_year).sum() / actual.sum()
    assert model_error < 0.35
    assert model_error < baseline_error


def test_changing_data_after_the_origin_cannot_change_the_forecast(history, model):
    original = recursive_forecast(model, history, ORIGIN)
    tampered = synthetic_history()
    tampered.units[:, ORIGIN + 1 :] = 10_000
    tampered.stockout[:, ORIGIN + 1 :] = True
    tampered.promo[:, ORIGIN + 1 :] = True
    tampered.price_index[:, ORIGIN + 1 :] = 0.1
    np.testing.assert_array_equal(original, recursive_forecast(model, tampered, ORIGIN))


def test_shorter_horizon_matches_the_start_of_the_full_forecast(history, model):
    full = recursive_forecast(model, history, ORIGIN, 28)
    short = recursive_forecast(model, history, ORIGIN, 10)
    np.testing.assert_allclose(short, full[:, :10])


def test_visible_history_pads_and_hides_the_future(history):
    visible = visible_history(history, 100, 28)
    assert visible.units.shape[1] == 129
    assert np.isnan(visible.units[:, 101:]).all()
    assert not visible.stockout[:, 101:].any()
    np.testing.assert_array_equal(visible.units[:, :101], history.units[:, :101])


def test_model_round_trips_through_ubj_bytes(history, model):
    restored = DemandModel.from_bytes(model.to_bytes(), model.describe())
    np.testing.assert_allclose(
        recursive_forecast(restored, history, ORIGIN), recursive_forecast(model, history, ORIGIN)
    )
    assert restored.describe() == model.describe()
    assert model.to_bytes()[:1] == b"{"


@pytest.mark.parametrize("horizon", [0, 29])
def test_horizon_must_be_between_one_and_twenty_eight(history, model, horizon):
    with pytest.raises(ValueError):
        recursive_forecast(model, history, ORIGIN, horizon)
