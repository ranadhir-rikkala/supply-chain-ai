"""Panel loader: dense layout, missing days, cut-off and COPY from Postgres."""

import os
import uuid
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pipeline.forecast.panel import build_panel, load_panel


def _frames():
    products = pd.DataFrame(
        {"sku": ["SNK-0002", "BEV-0001"], "category": ["SNK", "BEV"], "unit_price": [4.0, 2.0]}
    )
    warehouses = pd.DataFrame({"warehouse_id": ["WH-SEA", "WH-CHI"]})
    sales = pd.DataFrame(
        {
            "feed_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02"],
            "sku": ["BEV-0001", "BEV-0001", "BEV-0001", "SNK-0002", "UNKNOWN"],
            "warehouse_id": ["WH-CHI", "WH-CHI", "WH-CHI", "WH-SEA", "WH-SEA"],
            "units_sold": [5, 7, 0, 3, 99],
            "unit_price": [2.0, 1.5, 2.0, 4.0, 1.0],
            "promo_flag": [False, True, False, False, False],
        }
    )
    inventory = pd.DataFrame(
        {
            "feed_date": ["2024-01-01", "2024-01-03"],
            "sku": ["BEV-0001", "BEV-0001"],
            "warehouse_id": ["WH-CHI", "WH-CHI"],
            "on_hand": [10, 0],
            "stockout_flag": [False, True],
        }
    )
    return products, warehouses, sales, inventory


def test_build_panel_is_dense_per_series_and_day():
    panel = build_panel(*_frames())
    assert panel.series_count == 4
    assert panel.day_count == 3
    assert panel.start == date(2024, 1, 1)
    assert panel.end == date(2024, 1, 3)
    assert list(panel.warehouses) == ["WH-CHI", "WH-CHI", "WH-SEA", "WH-SEA"]
    assert list(panel.skus) == ["BEV-0001", "SNK-0002", "BEV-0001", "SNK-0002"]
    assert list(panel.categories) == ["BEV", "SNK", "BEV", "SNK"]


def test_build_panel_places_values_and_marks_missing_days():
    panel = build_panel(*_frames())
    beverage = panel.series_index("BEV-0001", "WH-CHI")
    snack = panel.series_index("SNK-0002", "WH-SEA")
    np.testing.assert_array_equal(panel.units[beverage], [5, 7, 0])
    assert panel.units[snack, 0] == 3
    assert np.isnan(panel.units[snack, 1:]).all()
    assert np.isnan(panel.units[panel.series_index("SNK-0002", "WH-CHI")]).all()
    np.testing.assert_array_equal(panel.promo[beverage], [False, True, False])
    np.testing.assert_allclose(panel.price_index[beverage], [1.0, 0.75, 1.0])
    np.testing.assert_array_equal(panel.stockout[beverage], [False, False, True])
    assert panel.on_hand[beverage, 0] == 10
    assert np.isnan(panel.on_hand[beverage, 1])


def test_build_panel_ignores_unknown_series_and_dates_outside_window():
    panel = build_panel(*_frames(), start=date(2024, 1, 2), end=date(2024, 1, 2))
    assert panel.day_count == 1
    assert np.nansum(panel.units) == 7


def test_build_panel_without_promo_column_defaults_to_no_promo():
    products, warehouses, sales, inventory = _frames()
    panel = build_panel(products, warehouses, sales.drop(columns="promo_flag"), inventory)
    assert not panel.promo.any()


def test_through_hides_days_after_cutoff():
    panel = build_panel(*_frames())
    cut = panel.through(date(2024, 1, 2))
    assert cut.day_count == 2
    assert cut.end == date(2024, 1, 2)
    assert cut.units.shape == cut.stockout.shape == cut.on_hand.shape == (4, 2)
    with pytest.raises(ValueError):
        panel.through(date(2024, 1, 5))


def test_series_index_rejects_unknown_series():
    with pytest.raises(KeyError):
        build_panel(*_frames()).series_index("BEV-0001", "WH-XXX")


@pytest.mark.integration
def test_load_panel_reads_facts_with_copy():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")
    import psycopg

    run_id = uuid.uuid4()
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO core.dim_supplier VALUES ('SUP-T01', 'Test', 'US-WEST', 4, 1) "
                "ON CONFLICT DO NOTHING"
            )
            cursor.execute(
                "INSERT INTO core.dim_product VALUES ('TST-0001', 'TST', 'Test item', 'SUP-T01', "
                "6, 1.00, 2.00) ON CONFLICT DO NOTHING"
            )
            cursor.execute(
                "INSERT INTO core.dim_warehouse VALUES ('WH-TST', 'Test', 'Test', 0, 0, 1) "
                "ON CONFLICT DO NOTHING"
            )
            for day, units, stockout in ((30, 4, False), (31, 0, True)):
                cursor.execute(
                    "INSERT INTO core.fact_sales_daily VALUES (%s, %s, 'TST-0001', 'WH-TST', "
                    "%s, 2.00, %s)",
                    (date(2026, 12, day), f"S-TST-{day}", units, run_id),
                )
                cursor.execute(
                    "INSERT INTO core.fact_inventory_daily VALUES (%s, %s, 'TST-0001', 'WH-TST', "
                    "%s, 0, 1, %s, %s)",
                    (date(2026, 12, day), f"I-TST-{day}", units, stockout, run_id),
                )
        try:
            panel = load_panel(connection, date(2026, 12, 30), date(2026, 12, 31))
            index = panel.series_index("TST-0001", "WH-TST")
            np.testing.assert_array_equal(panel.units[index], [4, 0])
            np.testing.assert_array_equal(panel.stockout[index], [False, True])
            assert panel.categories[index] == "TST"
        finally:
            connection.rollback()
