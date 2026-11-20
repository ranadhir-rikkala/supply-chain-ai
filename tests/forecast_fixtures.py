"""Synthetic data shared by the forecasting tests."""

from datetime import date

import numpy as np

from pipeline.forecast.features import SeriesHistory

START = date(2023, 1, 1)
CATEGORIES = ("BEV", "SNK", "WIN", "ELE")
WAREHOUSES = ("WH-01", "WH-02")
WEEKLY = np.array([0.9, 0.9, 0.95, 1.0, 1.1, 1.25, 1.0])


def synthetic_history(series_per_cell: int = 5, days: int = 760, seed: int = 11) -> SeriesHistory:
    """Seasonal synthetic demand with weekly pattern, noise and 3% stockouts."""
    rng = np.random.default_rng(seed)
    cells = [(c, w) for c in range(len(CATEGORIES)) for w in range(len(WAREHOUSES))]
    series = [cell for cell in cells for _ in range(series_per_cell)]
    category_codes = np.array([c for c, _ in series], dtype=np.float32)
    warehouse_codes = np.array([w for _, w in series], dtype=np.float32)
    base = rng.uniform(5, 40, size=len(series))[:, None]
    day_index = np.arange(days)
    weekday = WEEKLY[[(START.weekday() + d) % 7 for d in day_index]]
    phase = (category_codes[:, None] * 90 + day_index[None, :]) * 2 * np.pi / 365.25
    season = 1 + 0.35 * np.cos(phase)
    mean = base * weekday[None, :] * season
    units = rng.gamma(8.0, mean / 8.0).round().astype(np.float32)
    stockout = rng.random(units.shape) < 0.03
    units[stockout] = np.floor(units[stockout] * 0.3)
    return SeriesHistory(
        start=START,
        units=units,
        stockout=stockout,
        promo=np.zeros(units.shape, dtype=bool),
        price_index=np.ones(units.shape, dtype=np.float32),
        category_codes=category_codes,
        warehouse_codes=warehouse_codes,
    )


FAST_PARAMS = {"max_depth": 4, "min_child_weight": 5, "eta": 0.15}
FAST_ROUNDS = 80


TEST_SUPPLIER = "SUP-T01"


def synthetic_panel(series_per_cell: int = 1, days: int = 760):
    """SalesPanel built from synthetic_history with TST- products and WH-T warehouses."""
    from pipeline.forecast.panel import SalesPanel

    history = synthetic_history(series_per_cell=series_per_cell, days=days)
    codes = history.category_codes.astype(int)
    warehouse_codes = history.warehouse_codes.astype(int)
    occurrence: dict[tuple[int, int], int] = {}
    skus = []
    for category, warehouse in zip(codes, warehouse_codes, strict=True):
        index = occurrence.get((category, warehouse), 0)
        occurrence[(category, warehouse)] = index + 1
        skus.append(f"TST-{category}{index:03d}")
    return SalesPanel(
        start=START,
        skus=np.array(skus, dtype=object),
        warehouses=np.array([f"WH-T{w}" for w in warehouse_codes], dtype=object),
        categories=np.array([CATEGORIES[c] for c in codes], dtype=object),
        units=history.units,
        stockout=history.stockout,
        promo=history.promo,
        price_index=history.price_index,
        on_hand=np.full(history.units.shape, 50.0, dtype=np.float32),
    )


def ensure_dimensions(connection, panel) -> None:
    """Insert the test products, warehouses and supplier the panel refers to."""
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO core.dim_supplier VALUES (%s, 'Test', 'US-WEST', 4, 1) "
            "ON CONFLICT DO NOTHING",
            (TEST_SUPPLIER,),
        )
        for warehouse in sorted(set(panel.warehouses)):
            cursor.execute(
                "INSERT INTO core.dim_warehouse VALUES (%s, 'Test', 'Test', 0, 0, 1) "
                "ON CONFLICT DO NOTHING",
                (warehouse,),
            )
        for sku, category in sorted(set(zip(panel.skus, panel.categories, strict=True))):
            cursor.execute(
                "INSERT INTO core.dim_product VALUES (%s, %s, 'Test item', %s, 6, 1.00, 2.00) "
                "ON CONFLICT DO NOTHING",
                (sku, category, TEST_SUPPLIER),
            )
    connection.commit()
