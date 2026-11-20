"""Load sales and inventory facts into dense per-series, per-day arrays."""

from __future__ import annotations

import io
from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
from psycopg import sql

PROMO_COLUMN_CANDIDATES = ("promo_flag", "promotion")


@dataclass(frozen=True)
class SalesPanel:
    """Dense arrays for every (sku, warehouse) series over a continuous date range."""

    start: date
    skus: np.ndarray
    warehouses: np.ndarray
    categories: np.ndarray
    units: np.ndarray
    stockout: np.ndarray
    promo: np.ndarray
    price_index: np.ndarray
    on_hand: np.ndarray

    @property
    def series_count(self) -> int:
        """Number of (sku, warehouse) series in the panel."""
        return self.units.shape[0]

    @property
    def day_count(self) -> int:
        """Number of days in the panel."""
        return self.units.shape[1]

    @property
    def end(self) -> date:
        """Last date covered by the panel."""
        return self.start + timedelta(days=self.day_count - 1)

    def date_at(self, day_index: int) -> date:
        """Calendar date of a day index."""
        return self.start + timedelta(days=day_index)

    def day_index(self, day: date) -> int:
        """Day index of a calendar date."""
        return (day - self.start).days

    def series_index(self, sku: str, warehouse_id: str) -> int:
        """Row of a (sku, warehouse) series; raises KeyError if it is unknown."""
        matches = np.flatnonzero((self.skus == sku) & (self.warehouses == warehouse_id))
        if matches.size == 0:
            raise KeyError((sku, warehouse_id))
        return int(matches[0])

    def through(self, last_day: date) -> SalesPanel:
        """Copy of the panel that ends on last_day, hiding every later day."""
        days = self.day_index(last_day) + 1
        if days < 1 or days > self.day_count:
            raise ValueError(f"{last_day} is outside the panel {self.start}..{self.end}")
        return replace(
            self,
            units=self.units[:, :days],
            stockout=self.stockout[:, :days],
            promo=self.promo[:, :days],
            price_index=self.price_index[:, :days],
            on_hand=self.on_hand[:, :days],
        )


def build_panel(
    products: pd.DataFrame,
    warehouses: pd.DataFrame,
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    start: date | None = None,
    end: date | None = None,
) -> SalesPanel:
    """Build a SalesPanel from product, warehouse, sales and inventory frames."""
    products = products.sort_values("sku").reset_index(drop=True)
    warehouse_ids = np.sort(warehouses["warehouse_id"].to_numpy(dtype=object))
    skus = products["sku"].to_numpy(dtype=object)
    sales_dates = pd.to_datetime(sales["feed_date"])
    if start is None:
        if sales.empty:
            raise ValueError("no sales rows to build a panel from")
        start = sales_dates.min().date()
    if end is None:
        if sales.empty:
            raise ValueError("no sales rows to build a panel from")
        end = sales_dates.max().date()
    day_count = (end - start).days + 1
    series_count = len(skus) * len(warehouse_ids)
    shape = (series_count, day_count)

    def locate(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sku_codes = pd.Index(skus).get_indexer(frame["sku"])
        warehouse_codes = pd.Index(warehouse_ids).get_indexer(frame["warehouse_id"])
        day_codes = (pd.to_datetime(frame["feed_date"]) - pd.Timestamp(start)).dt.days.to_numpy()
        valid = (
            (sku_codes >= 0) & (warehouse_codes >= 0) & (day_codes >= 0) & (day_codes < day_count)
        )
        rows = warehouse_codes.astype(np.int64) * len(skus) + sku_codes
        return valid, rows[valid], day_codes[valid]

    # Days without a sales row stay NaN, not 0, so quarantined rows are not read as zero demand.
    units = np.full(shape, np.nan, dtype=np.float32)
    promo = np.zeros(shape, dtype=bool)
    price_index = np.ones(shape, dtype=np.float32)
    valid, rows, days = locate(sales)
    selected_sales = sales[valid]
    units[rows, days] = selected_sales["units_sold"].to_numpy(dtype=np.float32)
    if "promo_flag" in selected_sales:
        promo[rows, days] = selected_sales["promo_flag"].fillna(False).to_numpy(dtype=bool)
    list_price = products.set_index("sku")["unit_price"].astype(float)
    reference_price = selected_sales["sku"].map(list_price).to_numpy(dtype=np.float64)
    observed_price = selected_sales["unit_price"].to_numpy(dtype=np.float64)
    ratio = np.where(reference_price > 0, observed_price / reference_price, 1.0)
    price_index[rows, days] = np.nan_to_num(ratio, nan=1.0).astype(np.float32)

    stockout = np.zeros(shape, dtype=bool)
    on_hand = np.full(shape, np.nan, dtype=np.float32)
    valid, rows, days = locate(inventory)
    selected_inventory = inventory[valid]
    stockout[rows, days] = selected_inventory["stockout_flag"].to_numpy(dtype=bool)
    on_hand[rows, days] = selected_inventory["on_hand"].to_numpy(dtype=np.float32)

    return SalesPanel(
        start=start,
        skus=np.tile(skus, len(warehouse_ids)),
        warehouses=np.repeat(warehouse_ids, len(skus)),
        categories=np.tile(products["category"].to_numpy(dtype=object), len(warehouse_ids)),
        units=units,
        stockout=stockout,
        promo=promo,
        price_index=price_index,
        on_hand=on_hand,
    )


def _copy_frame(cursor, query: sql.Composable) -> pd.DataFrame:
    buffer = io.BytesIO()
    statement = sql.SQL("COPY ({}) TO STDOUT WITH (FORMAT csv, HEADER true)").format(query)
    with cursor.copy(statement) as copy:
        for chunk in copy:
            buffer.write(chunk)
    buffer.seek(0)
    return pd.read_csv(buffer)


def _promo_column(cursor) -> str | None:
    cursor.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'core' AND table_name = 'fact_sales_daily' "
        "AND column_name = ANY(%s) ORDER BY column_name DESC LIMIT 1",
        (list(PROMO_COLUMN_CANDIDATES),),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def _date_filter(column: str, start: date | None, end: date | None) -> sql.Composable:
    conditions = [sql.SQL("true")]
    if start is not None:
        conditions.append(sql.SQL("{} >= {}").format(sql.Identifier(column), sql.Literal(start)))
    if end is not None:
        conditions.append(sql.SQL("{} <= {}").format(sql.Identifier(column), sql.Literal(end)))
    return sql.SQL(" AND ").join(conditions)


def load_panel(connection, start: date | None = None, end: date | None = None) -> SalesPanel:
    """Read the facts from Postgres with COPY and return a SalesPanel."""
    with connection.cursor() as cursor:
        products = _copy_frame(
            cursor, sql.SQL("SELECT sku, category, unit_price FROM core.dim_product")
        )
        warehouses = _copy_frame(cursor, sql.SQL("SELECT warehouse_id FROM core.dim_warehouse"))
        promo_column = _promo_column(cursor)
        promo_select = (
            sql.SQL("{}::boolean::int AS promo_flag").format(sql.Identifier(promo_column))
            if promo_column
            else sql.SQL("0 AS promo_flag")
        )
        sales = _copy_frame(
            cursor,
            sql.SQL(
                "SELECT feed_date, sku, warehouse_id, units_sold, unit_price, {} "
                "FROM core.fact_sales_daily WHERE {}"
            ).format(promo_select, _date_filter("feed_date", start, end)),
        )
        inventory = _copy_frame(
            cursor,
            sql.SQL(
                # Booleans are cast to integers because pandas reads Postgres 't' and 'f' as text.
                "SELECT feed_date, sku, warehouse_id, on_hand, stockout_flag::int AS stockout_flag "
                "FROM core.fact_inventory_daily WHERE {}"
            ).format(_date_filter("feed_date", start, end)),
        )
    return build_panel(products, warehouses, sales, inventory, start, end)
