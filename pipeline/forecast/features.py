"""Model features computed from history that ends at least 7 days before the target."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from pipeline.forecast.panel import SalesPanel

FEATURE_GAP_DAYS = 7
LAGS = (7, 14, 21, 28, 364)
ROLLING_MEAN_WINDOWS = (7, 28, 91)
ROLLING_STD_WINDOW = 28
STOCKOUT_SHARE_WINDOW = 28
SAME_WEEK_LAST_YEAR_OFFSET = 364
SAME_WEEK_LAST_YEAR_HALF_WIDTH = 3

FEATURE_NAMES = (
    *(f"lag_{lag}" for lag in LAGS),
    *(f"rolling_mean_{window}" for window in ROLLING_MEAN_WINDOWS),
    f"rolling_std_{ROLLING_STD_WINDOW}",
    "same_week_last_year_mean",
    "recent_stockout_share",
    "day_of_week",
    "month",
    "year",
    "days_from_black_friday",
    "promo_flag",
    "price_index",
    "category",
    "warehouse",
)
CATEGORICAL_FEATURES = ("category", "warehouse")


@dataclass(frozen=True)
class SeriesHistory:
    """Arrays the features are computed from, plus encoded category and warehouse codes."""

    start: date
    units: np.ndarray
    stockout: np.ndarray
    promo: np.ndarray
    price_index: np.ndarray
    category_codes: np.ndarray
    warehouse_codes: np.ndarray

    @classmethod
    def from_panel(
        cls,
        panel: SalesPanel,
        categories: tuple[str, ...],
        warehouses: tuple[str, ...],
        extra_days: int = 0,
    ) -> SeriesHistory:
        """Build a history from a panel, optionally padded with unknown future days."""

        def extend(values: np.ndarray, fill) -> np.ndarray:
            if extra_days == 0:
                return values.copy()
            padding = np.full((values.shape[0], extra_days), fill, dtype=values.dtype)
            return np.concatenate([values, padding], axis=1)

        return cls(
            start=panel.start,
            units=extend(panel.units.astype(np.float32), np.nan),
            stockout=extend(panel.stockout, False),
            promo=extend(panel.promo, False),
            price_index=extend(panel.price_index.astype(np.float32), 1.0),
            category_codes=_codes(panel.categories, categories),
            warehouse_codes=_codes(panel.warehouses, warehouses),
        )


def _codes(values: np.ndarray, vocabulary: tuple[str, ...]) -> np.ndarray:
    lookup = {name: code for code, name in enumerate(vocabulary)}
    return np.array([lookup.get(value, -1) for value in values], dtype=np.float32)


def black_friday(year: int) -> date:
    """Black Friday of a year: the day after the fourth Thursday of November."""
    first_of_november = date(year, 11, 1)
    first_thursday = first_of_november + timedelta(days=(3 - first_of_november.weekday()) % 7)
    return first_thursday + timedelta(days=22)


def days_from_nearest_black_friday(day: date) -> int:
    """Signed days from the closest Black Friday in the previous, same or next year."""
    candidates = (black_friday(day.year - 1), black_friday(day.year), black_friday(day.year + 1))
    return min(((day - candidate).days for candidate in candidates), key=abs)


def _prefix_sums(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Prefix sums give every rolling window in constant time and let missing days be skipped.
    observed = ~np.isnan(values)
    filled = np.where(observed, values, 0.0).astype(np.float64)
    zeros = np.zeros((values.shape[0], 1))
    total = np.concatenate([zeros, np.cumsum(filled, axis=1)], axis=1)
    squares = np.concatenate([zeros, np.cumsum(filled * filled, axis=1)], axis=1)
    count = np.concatenate([zeros, np.cumsum(observed, axis=1)], axis=1)
    return total, squares, count


def _window(prefix: np.ndarray, last_day: int, length: int) -> np.ndarray:
    if last_day < 0:
        return np.zeros(prefix.shape[0])
    first_day = max(0, last_day - length + 1)
    return prefix[:, last_day + 1] - prefix[:, first_day]


def _column(values: np.ndarray, day: int) -> np.ndarray:
    if day < 0:
        return np.full(values.shape[0], np.nan, dtype=np.float32)
    return values[:, day]


def build_features(history: SeriesHistory, days: np.ndarray) -> np.ndarray:
    """Feature matrix for every series on each target day, ordered day by day."""
    series_count = history.units.shape[0]
    total, squares, count = _prefix_sums(history.units)
    stockout_total = np.concatenate(
        [np.zeros((series_count, 1)), np.cumsum(history.stockout, axis=1)], axis=1
    )
    blocks = []
    for day in days:
        day = int(day)
        # Every window ends 7 days back, so a whole week can be predicted from known data.
        last_visible = day - FEATURE_GAP_DAYS
        columns = [_column(history.units, day - lag) for lag in LAGS]
        for window in ROLLING_MEAN_WINDOWS:
            observed = _window(count, last_visible, window)
            with np.errstate(invalid="ignore", divide="ignore"):
                columns.append(
                    np.where(observed > 0, _window(total, last_visible, window) / observed, np.nan)
                )
        observed = _window(count, last_visible, ROLLING_STD_WINDOW)
        window_sum = _window(total, last_visible, ROLLING_STD_WINDOW)
        window_squares = _window(squares, last_visible, ROLLING_STD_WINDOW)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = window_sum / observed
            variance = np.maximum(window_squares / observed - mean * mean, 0.0)
            columns.append(np.where(observed > 1, np.sqrt(variance), np.nan))
        last_year_end = day - SAME_WEEK_LAST_YEAR_OFFSET + SAME_WEEK_LAST_YEAR_HALF_WIDTH
        last_year_length = 2 * SAME_WEEK_LAST_YEAR_HALF_WIDTH + 1
        observed = _window(count, last_year_end, last_year_length)
        with np.errstate(invalid="ignore", divide="ignore"):
            columns.append(
                np.where(
                    observed > 0, _window(total, last_year_end, last_year_length) / observed, np.nan
                )
            )
        stockout_days = _window(stockout_total, last_visible, STOCKOUT_SHARE_WINDOW)
        visible_days = min(STOCKOUT_SHARE_WINDOW, max(0, last_visible + 1))
        columns.append(
            stockout_days / visible_days if visible_days else np.full(series_count, np.nan)
        )
        calendar_day = history.start + timedelta(days=day)
        for value in (
            calendar_day.weekday(),
            calendar_day.month,
            calendar_day.year,
            days_from_nearest_black_friday(calendar_day),
        ):
            columns.append(np.full(series_count, value))
        columns.append(history.promo[:, day].astype(np.float32))
        columns.append(history.price_index[:, day])
        columns.append(history.category_codes)
        columns.append(history.warehouse_codes)
        blocks.append(np.column_stack(columns).astype(np.float32))
    if not blocks:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.concatenate(blocks, axis=0)
