from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import random
from collections import deque
from datetime import date, timedelta
from pathlib import Path

from .storage import make_store

SEED = 272
CATEGORIES = ["BEV", "SNK", "DAI", "FRZ", "PCR", "HHC", "PAP", "ELE", "PHA", "OUT", "WIN", "SCH"]
PRODUCT_WORDS = {
    "BEV": ["Sparkling Water Lime 12x355ml", "Cold Brew Coffee 6x250ml"],
    "SNK": ["Sea Salt Crisps 12x150g", "Honey Granola Bites 8x200g"],
    "DAI": ["Greek Yogurt Vanilla 6x150g", "Sharp Cheddar Block 400g"],
    "FRZ": ["Wild Berry Frozen Mix 1kg", "Margherita Pizza 2x400g"],
    "PCR": ["Laundry Pods Fresh 42ct", "Dishwasher Tablets 30ct"],
    "HHC": ["Cotton Bath Towel 2pk", "Bamboo Storage Basket Medium"],
    "PAP": ["Premium Paper Towels 6pk", "Recycled Copy Paper 500ct"],
    "ELE": ["USB-C Charging Cable 2m", "Wireless Desk Charger 15W"],
    "PHA": ["Daily Vitamin C 100ct", "First Aid Travel Kit 35pc"],
    "OUT": ["Trail Hydration Pack 2L", "Insulated Camp Mug 16oz"],
    "WIN": ["Fleece Winter Blanket Queen", "Thermal Hand Warmers 10pk"],
    "SCH": ["Graph Paper Notebook 3pk", "Colored Pencil Set 24ct"],
}
WAREHOUSES = [
    ("Los Angeles", 34.0522, -118.2437, 1.25, 0.85),
    ("Seattle", 47.6062, -122.3321, 1.10, 1.20),
    ("Chicago", 41.8781, -87.6298, 0.95, 1.25),
    ("Dallas", 32.7767, -96.7970, 1.20, 0.95),
    ("New York", 40.7128, -74.0060, 1.00, 1.10),
    ("Miami", 25.7617, -80.1918, 1.35, 0.80),
    ("Denver", 39.7392, -104.9903, 0.90, 1.30),
    ("Phoenix", 33.4484, -112.0740, 1.40, 0.75),
]
SUPPLIER_REGIONS = [
    ("US-WEST", 3, 6),
    ("US-CENTRAL", 3, 6),
    ("US-EAST", 3, 6),
    ("MEXICO", 8, 14),
    ("CHINA", 30, 45),
]

CATEGORY_BASE_DEMAND = {
    "BEV": 30.0,
    "SNK": 24.0,
    "DAI": 20.0,
    "FRZ": 14.0,
    "PCR": 12.0,
    "HHC": 8.0,
    "PAP": 16.0,
    "ELE": 6.0,
    "PHA": 10.0,
    "OUT": 7.0,
    "WIN": 6.0,
    "SCH": 9.0,
}
CATEGORY_SEASON = {
    "BEV": (0.30, 196, "summer"),
    "SNK": (0.08, 200, None),
    "DAI": (0.05, 30, None),
    "FRZ": (0.25, 205, "summer"),
    "PCR": (0.04, 30, None),
    "HHC": (0.10, 340, None),
    "PAP": (0.05, 20, None),
    "ELE": (0.08, 350, None),
    "PHA": (0.22, 15, "winter"),
    "OUT": (0.50, 175, "summer"),
    "WIN": (0.65, 10, "winter"),
    "SCH": (0.10, 240, None),
}
WEEKDAY_FACTORS = (0.92, 0.90, 0.93, 0.98, 1.08, 1.18, 1.01)
YEARLY_GROWTH = 0.06
DEMAND_SHAPE = 6.0
PROMO_WEEKLY_PROBABILITY = 0.06
PROMO_LIFT = 1.30
PROMO_DISCOUNT = 0.15
SAFETY_DAYS = 4
REVIEW_DAYS = 7
LATE_DELIVERY_PROBABILITY = 0.12
DISRUPTION_DAYS = {
    "SUPPLIER_DELAY": 14,
    "PORT_CLOSURE": 10,
    "HEAT_WAVE": 7,
    "COLD_SNAP": 7,
    "PANIC_BUYING": 5,
    "VIRAL_PRODUCT": 21,
}
SUPPLY_DELAY_DAYS = {"SUPPLIER_DELAY": 10, "PORT_CLOSURE": 14}
DEMAND_SHOCKS = {
    "HEAT_WAVE": ({"BEV", "FRZ", "OUT"}, 1.8),
    "COLD_SNAP": ({"WIN", "PHA"}, 1.8),
    "PANIC_BUYING": ({"PAP", "PCR", "PHA", "DAI"}, 2.2),
}
VIRAL_LIFT = 3.5
WEST_COAST_CITIES = ("Los Angeles", "Seattle")
OCEAN_REGION = "CHINA"
PO_EVENT_FIELDS = [
    "event_id",
    "po_id",
    "event_type",
    "event_time",
    "supplier_id",
    "warehouse_id",
    "sku",
    "quantity",
    "eta",
]
RECEIPT_FIELDS = ["receipt_id", "po_id", "receipt_date", "quantity", "warehouse_id"]


def _csv_bytes(rows: list[dict], fields: list[str], gz: bool = True) -> bytes:
    raw = io.StringIO(newline="")
    writer = csv.DictWriter(raw, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    data = raw.getvalue().encode()
    return gzip.compress(data, compresslevel=6, mtime=0) if gz else data


def catalog(rng: random.Random):
    suppliers = []
    for category in CATEGORIES:
        for region, lo, hi in SUPPLIER_REGIONS:
            suppliers.append(
                {
                    "supplier_id": f"SUP-{len(suppliers) + 1:03d}",
                    "supplier_name": f"Brookline {category} {region} Supply",
                    "category": category,
                    "region": region,
                    "lead_time_mean": round((lo + hi) / 2, 2),
                    "lead_time_sd": round((hi - lo) / 4, 2),
                }
            )
    products = []
    for i in range(500):
        category = CATEGORIES[i % len(CATEGORIES)]
        template = PRODUCT_WORDS[category][i % len(PRODUCT_WORDS[category])]
        supplier = suppliers[CATEGORIES.index(category) * 5 + (i // len(CATEGORIES)) % 5]
        unit_cost = round(rng.uniform(2, 120), 2)
        products.append(
            {
                "sku": f"SKU-{i + 1:04d}",
                "category": category,
                "product_name": f"Brookline {template} {i + 1:04d}",
                "supplier_id": supplier["supplier_id"],
                "case_pack": rng.choice([6, 12, 24]),
                "unit_cost": f"{unit_cost:.2f}",
                "unit_price": f"{unit_cost * rng.uniform(1.25, 2.4):.2f}",
            }
        )
    warehouses = [
        {
            "warehouse_id": f"WH-{i + 1:02d}",
            "warehouse_name": f"{city} Distribution Center",
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "summer_scale": summer,
            "winter_scale": winter,
            "climate_scale": (summer + winter) / 2,
        }
        for i, (city, lat, lon, summer, winter) in enumerate(WAREHOUSES)
    ]
    return products, suppliers, warehouses


def _disruptions(start: date, end: date) -> list[dict]:
    total_days = max(1, (end - start).days)
    kinds = [
        "SUPPLIER_DELAY",
        "PORT_CLOSURE",
        "HEAT_WAVE",
        "COLD_SNAP",
        "PANIC_BUYING",
        "VIRAL_PRODUCT",
    ]
    rows = []
    for i in range(40):
        offset = (
            total_days - 180 + ((i - 24) * 179 // 15)
            if i >= 24 and total_days >= 180
            else i * total_days // 40
        )
        event_date = start + timedelta(days=max(0, offset))
        kind = kinds[i % len(kinds)]
        rows.append(
            {
                "disruption_id": f"DIS-{i + 1:03d}",
                "disruption_type": kind,
                "start_date": event_date.isoformat(),
                "end_date": (event_date + timedelta(days=DISRUPTION_DAYS[kind] - 1)).isoformat(),
                "description": f"Planted {kind.lower().replace('_', ' ')} event",
            }
        )
    return rows


def black_friday(year: int) -> date:
    first_of_november = date(year, 11, 1)
    first_thursday = first_of_november + timedelta(days=(3 - first_of_november.weekday()) % 7)
    return first_thursday + timedelta(days=22)


def demand_multiplier(day: date, category: str, warehouse: dict) -> float:
    amplitude, peak_day, climate = CATEGORY_SEASON[category]
    if climate == "summer":
        amplitude *= warehouse["summer_scale"]
    elif climate == "winter":
        amplitude *= warehouse["winter_scale"]
    day_of_year = day.timetuple().tm_yday
    value = 1.0 + amplitude * math.cos(2 * math.pi * (day_of_year - peak_day) / 365.25)
    value *= (1.0 + YEARLY_GROWTH) ** ((day - date(2023, 1, 1)).days / 365.25)
    value *= WEEKDAY_FACTORS[day.weekday()]
    days_after_black_friday = (day - black_friday(day.year)).days
    if 0 <= days_after_black_friday <= 3:
        value *= 2.5 if category == "ELE" else 1.4
    if day.month == 12 and day.day <= 24:
        value *= 1.35 if category in ("ELE", "HHC", "WIN") else 1.12
    if category == "SCH" and date(day.year, 7, 25) <= day <= date(day.year, 9, 7):
        value *= 2.2
    if category == "ELE" and date(day.year, 9, 12) <= day <= date(day.year, 9, 30):
        value *= 1.6
    return value


def _assign_disruption_targets(
    disruptions: list[dict],
    products: list[dict],
    suppliers: list[dict],
    warehouses: list[dict],
    rng,
) -> None:
    ocean_suppliers = [s for s in suppliers if s["region"] == OCEAN_REGION]
    west_coast = [w for w in warehouses if w["city"] in WEST_COAST_CITIES]
    for disruption in disruptions:
        kind = disruption["disruption_type"]
        supplier_pool = ocean_suppliers if kind == "PORT_CLOSURE" else suppliers
        warehouse_pool = west_coast if kind == "PORT_CLOSURE" else warehouses
        disruption.update(
            {
                "supplier_id": supplier_pool[rng.randrange(len(supplier_pool))]["supplier_id"],
                "sku": products[rng.randrange(len(products))]["sku"],
                "warehouse_id": warehouse_pool[rng.randrange(len(warehouse_pool))]["warehouse_id"],
            }
        )


def _demand_shock(product: dict, warehouse_id: str, active: list[dict]) -> float:
    shock = 1.0
    for disruption in active:
        kind = disruption["disruption_type"]
        if kind == "VIRAL_PRODUCT" and disruption["sku"] == product["sku"]:
            shock *= VIRAL_LIFT
        elif kind in DEMAND_SHOCKS and disruption["warehouse_id"] == warehouse_id:
            categories, lift = DEMAND_SHOCKS[kind]
            if product["category"] in categories:
                shock *= lift
    return shock


def _supply_delay(supplier: dict, warehouse_id: str, active: list[dict]) -> int:
    delay = 0
    for disruption in active:
        kind = disruption["disruption_type"]
        if kind not in SUPPLY_DELAY_DAYS:
            continue
        if kind == "SUPPLIER_DELAY" and disruption["supplier_id"] == supplier["supplier_id"]:
            delay = max(delay, SUPPLY_DELAY_DAYS[kind])
        if (
            kind == "PORT_CLOSURE"
            and supplier["region"] == OCEAN_REGION
            and disruption["warehouse_id"] == warehouse_id
        ):
            delay = max(delay, SUPPLY_DELAY_DAYS[kind])
    return delay


def _affected_by(disruption: dict, order: dict, suppliers_by_id: dict[str, dict]) -> bool:
    if disruption["disruption_type"] == "SUPPLIER_DELAY":
        return order["supplier_id"] == disruption["supplier_id"]
    if disruption["disruption_type"] == "PORT_CLOSURE":
        return (
            suppliers_by_id[order["supplier_id"]]["region"] == OCEAN_REGION
            and order["warehouse_id"] == disruption["warehouse_id"]
        )
    return False


def _plant_row_defects(rows: list[dict], kind: str, file_key: str, planted: list[dict]) -> None:
    key = "sale_id" if kind == "sales" else "inventory_id"
    quantity_field = "units_sold" if kind == "sales" else "on_hand"
    for index in range(0, len(rows), 100):
        row = rows[index]
        mode = (index // 100) % 5
        if mode == 0:
            row["sku"] = ""
            reason, defect = "MISSING_FIELD:sku", "missing_sku"
        elif mode == 1:
            row[quantity_field] = "-7"
            reason, defect = f"NEGATIVE_VALUE:{quantity_field}", "negative_quantity"
        elif mode == 2:
            row["warehouse_id"] = "WH-UNKNOWN"
            reason, defect = "UNKNOWN_WAREHOUSE", "unknown_warehouse"
        elif mode == 3:
            row[quantity_field] = "not-a-number"
            reason, defect = f"BAD_TYPE:{quantity_field}", "non_numeric"
        else:
            row[key] = rows[index - 1][key]
            reason, defect = "DUPLICATE_KEY", "duplicate_row"
        planted.append(
            {
                "defect_id": f"DEF-{len(planted) + 1:06d}",
                "file_key": file_key,
                "line_number": index + 2,
                "defect_type": defect,
                "expected_reason": reason,
            }
        )


def generate(root: str | Path, start: date, end: date, seed: int = SEED) -> dict:
    rng = random.Random(seed)
    store = make_store(root)
    products, suppliers, warehouses = catalog(rng)
    store.put("catalog/products.csv", _csv_bytes(products, list(products[0]), False))
    store.put("catalog/suppliers.csv", _csv_bytes(suppliers, list(suppliers[0]), False))
    store.put("catalog/warehouses.csv", _csv_bytes(warehouses, list(warehouses[0]), False))
    disruptions = _disruptions(start, end)
    _assign_disruption_targets(disruptions, products, suppliers, warehouses, rng)
    suppliers_by_id = {supplier["supplier_id"]: supplier for supplier in suppliers}
    warehouse_size = {w["warehouse_id"]: rng.uniform(0.7, 1.3) for w in warehouses}
    cells = []
    for warehouse in warehouses:
        for product in products:
            base = (
                CATEGORY_BASE_DEMAND[product["category"]]
                * rng.lognormvariate(0.0, 0.5)
                * warehouse_size[warehouse["warehouse_id"]]
            )
            supplier = suppliers_by_id[product["supplier_id"]]
            cover_days = supplier["lead_time_mean"] + SAFETY_DAYS + REVIEW_DAYS
            cells.append(
                {
                    "warehouse": warehouse,
                    "product": product,
                    "supplier": supplier,
                    "base": base,
                    "on_hand": int(base * cover_days),
                    "open_orders": [],
                    "recent_sales": deque(maxlen=28),
                    "promo": False,
                }
            )
    planted, manifest = [], []
    open_orders_by_id: dict[str, dict] = {}
    po_sequence = 0
    day = start
    day_index = 0
    total_days = (end - start).days + 1
    while day <= end:
        datestr = day.isoformat()
        sales, inventory, events, receipts = [], [], [], []
        active = [d for d in disruptions if d["start_date"] <= datestr <= d["end_date"]]
        for disruption in disruptions:
            if disruption["start_date"] != datestr or disruption["disruption_type"] not in (
                SUPPLY_DELAY_DAYS
            ):
                continue
            delay = SUPPLY_DELAY_DAYS[disruption["disruption_type"]]
            for order in open_orders_by_id.values():
                if order["arrival"] > day and _affected_by(disruption, order, suppliers_by_id):
                    order["arrival"] += timedelta(days=delay)
                    events.append(
                        {
                            "event_id": f"ETA-{order['po_id']}-{datestr}",
                            "po_id": order["po_id"],
                            "event_type": "ETA_UPDATED",
                            "event_time": f"{datestr}T12:00:00Z",
                            "supplier_id": order["supplier_id"],
                            "warehouse_id": order["warehouse_id"],
                            "sku": order["sku"],
                            "quantity": order["quantity"],
                            "eta": order["arrival"].isoformat(),
                        }
                    )
        for cell in cells:
            warehouse, product, supplier = cell["warehouse"], cell["product"], cell["supplier"]
            still_open = []
            for order in cell["open_orders"]:
                if order["arrival"] <= day:
                    cell["on_hand"] += order["quantity"]
                    open_orders_by_id.pop(order["po_id"], None)
                    receipts.append(
                        {
                            "receipt_id": f"RC-{order['po_id']}",
                            "po_id": order["po_id"],
                            "receipt_date": datestr,
                            "quantity": order["quantity"],
                            "warehouse_id": order["warehouse_id"],
                        }
                    )
                else:
                    still_open.append(order)
            cell["open_orders"] = still_open
            if day.weekday() == 0 or day_index == 0:
                cell["promo"] = rng.random() < PROMO_WEEKLY_PROBABILITY
            mean_demand = (
                cell["base"]
                * demand_multiplier(day, product["category"], warehouse)
                * (PROMO_LIFT if cell["promo"] else 1.0)
                * _demand_shock(product, warehouse["warehouse_id"], active)
            )
            potential_demand = round(rng.gammavariate(DEMAND_SHAPE, mean_demand / DEMAND_SHAPE))
            sold = min(potential_demand, cell["on_hand"])
            stockout = sold < potential_demand
            cell["on_hand"] -= sold
            cell["recent_sales"].append(sold)
            moving_average = (
                sum(cell["recent_sales"]) / len(cell["recent_sales"])
                if len(cell["recent_sales"]) >= 7
                else cell["base"]
            )
            lead_mean = supplier["lead_time_mean"]
            reorder_point = max(1, math.ceil(moving_average * (lead_mean + SAFETY_DAYS)))
            order_up_to = moving_average * (lead_mean + SAFETY_DAYS + REVIEW_DAYS)
            on_order = sum(order["quantity"] for order in cell["open_orders"])
            if cell["on_hand"] + on_order <= reorder_point:
                case_pack = product["case_pack"]
                needed = max(case_pack, order_up_to - cell["on_hand"] - on_order)
                quantity = math.ceil(needed / case_pack) * case_pack
                lead_time = max(1, round(rng.gauss(lead_mean, supplier["lead_time_sd"])))
                if rng.random() < LATE_DELIVERY_PROBABILITY:
                    lead_time += rng.randint(2, 10)
                lead_time += _supply_delay(supplier, warehouse["warehouse_id"], active)
                po_sequence += 1
                order = {
                    "po_id": f"PO-{po_sequence:07d}",
                    "supplier_id": supplier["supplier_id"],
                    "warehouse_id": warehouse["warehouse_id"],
                    "sku": product["sku"],
                    "quantity": quantity,
                    "arrival": day + timedelta(days=lead_time),
                }
                cell["open_orders"].append(order)
                open_orders_by_id[order["po_id"]] = order
                on_order += quantity
                events.append(
                    {
                        "event_id": f"EV-{order['po_id']}",
                        "po_id": order["po_id"],
                        "event_type": "CREATED",
                        "event_time": f"{datestr}T08:00:00Z",
                        "supplier_id": order["supplier_id"],
                        "warehouse_id": order["warehouse_id"],
                        "sku": order["sku"],
                        "quantity": quantity,
                        "eta": order["arrival"].isoformat(),
                    }
                )
            unit_price = float(product["unit_price"])
            if cell["promo"]:
                unit_price *= 1 - PROMO_DISCOUNT
            sales.append(
                {
                    "feed_date": datestr,
                    "sale_id": f"S-{datestr}-{warehouse['warehouse_id']}-{product['sku']}",
                    "sku": product["sku"],
                    "warehouse_id": warehouse["warehouse_id"],
                    "units_sold": sold,
                    "unit_price": f"{unit_price:.2f}",
                    "promotion": int(cell["promo"]),
                }
            )
            inventory.append(
                {
                    "feed_date": datestr,
                    "inventory_id": f"I-{datestr}-{warehouse['warehouse_id']}-{product['sku']}",
                    "sku": product["sku"],
                    "warehouse_id": warehouse["warehouse_id"],
                    "on_hand": cell["on_hand"],
                    "on_order": on_order,
                    "reorder_point": reorder_point,
                    "stockout_flag": stockout,
                }
            )
        rows_by_kind = [
            ("sales", sales, list(sales[0])),
            ("inventory", inventory, list(inventory[0])),
            ("po_events", events, PO_EVENT_FIELDS),
            ("receipts", receipts, RECEIPT_FIELDS),
        ]
        if total_days > 2 and day_index in (total_days // 3, (2 * total_days) // 3):
            renamed_inventory = []
            for row in inventory:
                renamed = dict(row)
                renamed["onhand"] = renamed.pop("on_hand")
                renamed_inventory.append(renamed)
            inventory = renamed_inventory
            rows_by_kind[1] = (
                "inventory",
                inventory,
                [
                    "feed_date",
                    "inventory_id",
                    "sku",
                    "warehouse_id",
                    "onhand",
                    "on_order",
                    "reorder_point",
                    "stockout_flag",
                ],
            )
            planted.append(
                {
                    "defect_id": f"DEF-SCHEMA-{day_index:06d}",
                    "file_key": f"raw/inventory/{datestr}.csv.gz",
                    "line_number": 1,
                    "defect_type": "renamed_column",
                    "expected_reason": "FILE_SCHEMA_MISMATCH",
                }
            )
        for name, rows, fields in rows_by_kind:
            key = f"raw/{name}/{datestr}.csv.gz"
            if name == "sales" or (name == "inventory" and "on_hand" in fields):
                _plant_row_defects(rows, name, key, planted)
            data = _csv_bytes(rows, fields)
            store.put(key, data)
            manifest.append(
                {
                    "file_key": key,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "row_count": len(rows),
                    "feed_date": datestr,
                }
            )
        day += timedelta(days=1)
        day_index += 1
    store.put(
        "raw/manifest.csv",
        _csv_bytes(manifest, ["file_key", "sha256", "row_count", "feed_date"], False),
    )
    store.put(
        "eval/erp/planted_defects.csv.gz",
        _csv_bytes(
            planted, ["defect_id", "file_key", "line_number", "defect_type", "expected_reason"]
        ),
    )
    store.put(
        "eval/erp/disruptions.csv",
        _csv_bytes(
            disruptions,
            [
                "disruption_id",
                "disruption_type",
                "start_date",
                "end_date",
                "description",
                "supplier_id",
                "sku",
                "warehouse_id",
            ],
            False,
        ),
    )
    return {
        "files": len(manifest),
        "rows": sum(int(x["row_count"]) for x in manifest),
        "seed": seed,
        "planted_defects": len(planted),
        "disruptions": len(disruptions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic ERP feeds")
    parser.add_argument("--output", default="data")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.start, args.end, args.seed)))


if __name__ == "__main__":
    main()
