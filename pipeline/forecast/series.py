"""History, backtest and forecast of one series for the series endpoint."""

from __future__ import annotations


class SeriesNotFound(LookupError):
    """Raised when the sku or warehouse is unknown."""


HISTORY_QUERY = """
WITH latest AS (
    SELECT max(feed_date) AS last_day
    FROM core.fact_sales_daily
    WHERE sku = %(sku)s AND warehouse_id = %(warehouse_id)s
)
SELECT s.feed_date, s.units_sold, s.promo_flag, i.stockout_flag, i.on_hand
FROM core.fact_sales_daily s
CROSS JOIN latest
LEFT JOIN core.fact_inventory_daily i
    ON i.feed_date = s.feed_date AND i.sku = s.sku AND i.warehouse_id = s.warehouse_id
WHERE s.sku = %(sku)s AND s.warehouse_id = %(warehouse_id)s
  AND s.feed_date > latest.last_day - %(history_days)s
ORDER BY s.feed_date
"""

BACKTEST_QUERY = """
SELECT b.target_date, b.actual, b.yhat, b.baseline
FROM ml.backtest b
JOIN ml.model_version m ON m.model_version = b.model_version AND m.status = 'ACTIVE'
WHERE b.sku = %(sku)s AND b.warehouse_id = %(warehouse_id)s
ORDER BY b.target_date
"""

FORECAST_QUERY = """
SELECT target_date, yhat, model_version
FROM ml.v_current_forecast
WHERE sku = %(sku)s AND warehouse_id = %(warehouse_id)s
ORDER BY target_date
"""


def fetch_series(connection, sku: str, warehouse_id: str, history_days: int) -> dict:
    """History, backtest of the ACTIVE model and current forecast for one series."""
    parameters = {"sku": sku, "warehouse_id": warehouse_id, "history_days": history_days}
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM core.dim_product WHERE sku = %(sku)s), "
            "EXISTS (SELECT 1 FROM core.dim_warehouse WHERE warehouse_id = %(warehouse_id)s)",
            parameters,
        )
        known_sku, known_warehouse = cursor.fetchone()
        if not (known_sku and known_warehouse):
            raise SeriesNotFound(f"{sku} at {warehouse_id}")
        cursor.execute(HISTORY_QUERY, parameters)
        history = [
            {
                "date": day,
                "units_sold": units,
                "promo_flag": bool(promo),
                "stockout_flag": bool(stockout) if stockout is not None else None,
                "on_hand": on_hand,
            }
            for day, units, promo, stockout, on_hand in cursor.fetchall()
        ]
        cursor.execute(BACKTEST_QUERY, parameters)
        backtest = [
            {"date": day, "actual": actual, "yhat": round(yhat, 3), "baseline": baseline}
            for day, actual, yhat, baseline in cursor.fetchall()
        ]
        cursor.execute(FORECAST_QUERY, parameters)
        forecast = [
            {"date": day, "yhat": round(yhat, 3), "model_version": version}
            for day, yhat, version in cursor.fetchall()
        ]
    return {
        "sku": sku,
        "warehouse_id": warehouse_id,
        "history": history,
        "backtest": backtest,
        "forecast": forecast,
    }
