"""Global XGBoost demand model with a recursive 28-day forecast."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import xgboost as xgb

from pipeline.forecast.features import (
    CATEGORICAL_FEATURES,
    FEATURE_GAP_DAYS,
    FEATURE_NAMES,
    ROLLING_MEAN_WINDOWS,
    SeriesHistory,
    build_features,
)

HORIZON_DAYS = 28
FIRST_TRAINING_DAY = max(ROLLING_MEAN_WINDOWS) + FEATURE_GAP_DAYS
DEFAULT_ROUNDS = 400
DEFAULT_PARAMS = {
    "objective": "reg:tweedie",
    "tweedie_variance_power": 1.2,
    "eta": 0.06,
    "max_depth": 8,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "tree_method": "hist",
    "max_bin": 256,
    "max_cat_to_onehot": 1,
    "seed": 272,
}
FEATURE_TYPES = ["c" if name in CATEGORICAL_FEATURES else "q" for name in FEATURE_NAMES]


def _matrix(features: np.ndarray, labels: np.ndarray | None = None) -> xgb.DMatrix:
    return xgb.DMatrix(
        features,
        label=labels,
        missing=np.nan,
        feature_names=list(FEATURE_NAMES),
        feature_types=FEATURE_TYPES,
        enable_categorical=True,
    )


@dataclass
class DemandModel:
    """A trained booster with the category and warehouse vocabularies it was trained on."""

    booster: xgb.Booster
    categories: tuple[str, ...]
    warehouses: tuple[str, ...]
    params: dict
    rounds: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Non-negative demand predictions for a feature matrix."""
        if features.shape[0] == 0:
            return np.empty(0, dtype=np.float32)
        return np.clip(self.booster.predict(_matrix(features)), 0.0, None).astype(np.float32)

    def to_bytes(self) -> bytes:
        """Serialize the booster in XGBoost's UBJSON format."""
        return bytes(self.booster.save_raw(raw_format="ubj"))

    def describe(self) -> dict:
        """Metadata needed to rebuild the model from its bytes."""
        return {
            "feature_names": list(FEATURE_NAMES),
            "categories": list(self.categories),
            "warehouses": list(self.warehouses),
            "params": self.params,
            "rounds": self.rounds,
            "horizon_days": HORIZON_DAYS,
        }

    @classmethod
    def from_bytes(cls, data: bytes, description: dict) -> DemandModel:
        """Rebuild a model from UBJSON bytes and its description."""
        booster = xgb.Booster()
        booster.load_model(bytearray(data))
        return cls(
            booster=booster,
            categories=tuple(description["categories"]),
            warehouses=tuple(description["warehouses"]),
            params=dict(description["params"]),
            rounds=int(description["rounds"]),
        )


def training_rows(
    history: SeriesHistory, first_day: int, last_day: int
) -> tuple[np.ndarray, np.ndarray]:
    """Features and labels for the training days, without stockout or missing days."""
    days = np.arange(max(first_day, FIRST_TRAINING_DAY), last_day + 1)
    features = build_features(history, days)
    labels = history.units[:, days].T.reshape(-1)
    # Stockout days understate demand and missing days have no label, so neither trains the model.
    usable = ~np.isnan(labels) & ~history.stockout[:, days].T.reshape(-1)
    return features[usable], labels[usable]


def train(
    history: SeriesHistory,
    categories: tuple[str, ...],
    warehouses: tuple[str, ...],
    last_training_day: int,
    params: dict | None = None,
    rounds: int = DEFAULT_ROUNDS,
) -> DemandModel:
    """Train the global Tweedie model on every day up to last_training_day."""
    chosen = {**DEFAULT_PARAMS, **(params or {})}
    features, labels = training_rows(history, FIRST_TRAINING_DAY, last_training_day)
    if labels.size == 0:
        raise ValueError("no usable training rows before the training cutoff")
    booster = xgb.train(chosen, _matrix(features, labels), num_boost_round=rounds)
    return DemandModel(booster, categories, warehouses, chosen, rounds)


def visible_history(history: SeriesHistory, origin_day: int, horizon: int) -> SeriesHistory:
    """Copy of the history that hides everything after origin_day and adds empty horizon days."""
    visible_days = origin_day + 1
    total_days = visible_days + horizon
    series_count = history.units.shape[0]

    def cut(values: np.ndarray, fill) -> np.ndarray:
        result = np.full((series_count, total_days), fill, dtype=values.dtype)
        # Sales, stockouts, promotions and prices after the origin are all hidden so nothing leaks from the future.
        result[:, :visible_days] = values[:, :visible_days]
        return result

    return replace(
        history,
        units=cut(history.units, np.nan),
        stockout=cut(history.stockout, False),
        promo=cut(history.promo, False),
        price_index=cut(history.price_index, 1.0),
    )


def recursive_forecast(
    model: DemandModel, history: SeriesHistory, origin_day: int, horizon: int = HORIZON_DAYS
) -> np.ndarray:
    """Forecast horizon days after origin_day, one week at a time."""
    if not 1 <= horizon <= HORIZON_DAYS:
        raise ValueError(f"horizon must be between 1 and {HORIZON_DAYS}")
    if origin_day < 0 or origin_day >= history.units.shape[1]:
        raise ValueError("origin day is outside the history")
    working = visible_history(history, origin_day, horizon)
    series_count = working.units.shape[0]
    forecast = np.empty((series_count, horizon), dtype=np.float32)
    for week_start in range(0, horizon, FEATURE_GAP_DAYS):
        offsets = np.arange(week_start, min(week_start + FEATURE_GAP_DAYS, horizon))
        days = origin_day + 1 + offsets
        predictions = model.predict(build_features(working, days))
        predictions = predictions.reshape(len(days), series_count).T
        # This week's predictions become history for next week's lag and rolling features.
        working.units[:, days] = predictions
        forecast[:, offsets] = predictions
    return forecast
