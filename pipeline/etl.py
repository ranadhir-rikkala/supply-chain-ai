from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from .messaging import FeedConsumer
from .validation import SPECS, feed_type, validate

KEY_FIELDS = {
    "sales": "sale_id",
    "inventory": "inventory_id",
    "po_events": "event_id",
    "receipts": "receipt_id",
}


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def _known_catalog(root: Path) -> dict[str, set[str]]:
    known = {"sku": set(), "warehouse_id": set(), "supplier_id": set()}
    for filename, key in (
        ("products.csv", "sku"),
        ("warehouses.csv", "warehouse_id"),
        ("suppliers.csv", "supplier_id"),
    ):
        with (root / "catalog" / filename).open(newline="") as handle:
            known[key].update(row[key] for row in csv.DictReader(handle))
    return known


def _expected_date(key: str, manifest_row: dict) -> str | None:
    if manifest_row.get("feed_date"):
        return manifest_row["feed_date"]
    return Path(key).stem


def check_file(
    path: Path,
    root: Path,
    manifest: dict[str, dict],
    known: dict[str, set[str]],
    expected_date: str | None = None,
) -> tuple[list[dict], list[dict]]:
    key = str(path.relative_to(root)).replace("\\", "/")
    kind = feed_type(key)
    expected = manifest.get(key)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if not expected or actual != expected.get("sha256"):
        return [], [{"line_number": 0, "reason_code": "FILE_CHECKSUM_MISMATCH", "raw": {}}]
    fields, rows = _read_csv(path)
    errors: list[dict] = []
    good: list[dict] = []
    if not kind or fields != SPECS[kind][0]:
        errors.append(
            {"line_number": 1, "reason_code": "FILE_SCHEMA_MISMATCH", "raw": {"header": fields}}
        )
        return [], errors
    if str(len(rows)) != str(expected.get("row_count", len(rows))):
        errors.append(
            {
                "line_number": 0,
                "reason_code": "FILE_ROW_COUNT_MISMATCH",
                "raw": {"actual": len(rows), "expected": expected.get("row_count")},
            }
        )
        return [], errors
    seen: set[str] = set()
    key_field = KEY_FIELDS[kind]
    for line_number, row in enumerate(rows, 2):
        reason = validate(row, kind, known)
        if (
            not reason
            and kind in {"sales", "inventory"}
            and expected_date
            and row.get("feed_date") != expected_date
        ):
            reason = "FEED_DATE_MISMATCH"
        if not reason and kind == "po_events" and row.get("event_time", "")[:10] != expected_date:
            reason = "FEED_DATE_MISMATCH"
        if not reason and row.get(key_field) in seen:
            reason = "DUPLICATE_KEY"
        seen.add(row.get(key_field, ""))
        if reason:
            errors.append({"line_number": line_number, "reason_code": reason, "raw": row})
        else:
            good.append(row)
    if rows and len(errors) / len(rows) > 0.05:
        errors.insert(
            0,
            {
                "line_number": 0,
                "reason_code": "QUALITY_GATE_REJECTED",
                "raw": {"bad_rows": len(errors), "row_count": len(rows)},
            },
        )
        good = []
    return good, errors


def _load_with_copy(db, kind: str, rows: list[dict], run_id: str) -> None:
    if not rows:
        return
    if kind == "sales":
        columns = [
            "feed_date",
            "sale_id",
            "sku",
            "warehouse_id",
            "units_sold",
            "unit_price",
            "promo_flag",
            "etl_run_id",
        ]
        values = [
            (
                r["feed_date"],
                r["sale_id"],
                r["sku"],
                r["warehouse_id"],
                int(r["units_sold"]),
                r["unit_price"],
                int(r["promotion"]) == 1,
                run_id,
            )
            for r in rows
        ]
        target, temp = "core.fact_sales_daily", "tmp_sales_daily"
        types = "feed_date date, sale_id text, sku text, warehouse_id text, units_sold int, unit_price numeric, promo_flag boolean, etl_run_id uuid"
    elif kind == "inventory":
        columns = [
            "feed_date",
            "inventory_id",
            "sku",
            "warehouse_id",
            "on_hand",
            "on_order",
            "reorder_point",
            "stockout_flag",
            "etl_run_id",
        ]
        values = [
            (
                r["feed_date"],
                r["inventory_id"],
                r["sku"],
                r["warehouse_id"],
                int(r["on_hand"]),
                int(r["on_order"]),
                int(r["reorder_point"]),
                r["stockout_flag"] in {"True", "1"},
                run_id,
            )
            for r in rows
        ]
        target, temp = "core.fact_inventory_daily", "tmp_inventory_daily"
        types = "feed_date date, inventory_id text, sku text, warehouse_id text, on_hand int, on_order int, reorder_point int, stockout_flag boolean, etl_run_id uuid"
    elif kind == "po_events":
        columns = [
            "event_id",
            "po_id",
            "event_type",
            "event_time",
            "supplier_id",
            "warehouse_id",
            "sku",
            "quantity",
            "eta",
            "etl_run_id",
        ]
        values = [
            (
                r["event_id"],
                r["po_id"],
                r["event_type"],
                r["event_time"],
                r["supplier_id"],
                r["warehouse_id"],
                r["sku"],
                int(r["quantity"]),
                r["eta"] or None,
                run_id,
            )
            for r in rows
        ]
        target, temp = "core.fact_po_event", "tmp_po_event"
        types = "event_id text, po_id text, event_type text, event_time timestamptz, supplier_id text, warehouse_id text, sku text, quantity int, eta date, etl_run_id uuid"
    else:
        columns = ["receipt_id", "po_id", "receipt_date", "quantity", "warehouse_id", "etl_run_id"]
        values = [
            (
                r["receipt_id"],
                r["po_id"],
                r["receipt_date"],
                int(r["quantity"]),
                r["warehouse_id"],
                run_id,
            )
            for r in rows
        ]
        target, temp = "core.fact_receipt", "tmp_receipt"
        types = "receipt_id text, po_id text, receipt_date date, quantity int, warehouse_id text, etl_run_id uuid"
    with db.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {temp}")
        cur.execute(f"CREATE TEMP TABLE {temp} ({types}) ON COMMIT DROP")
        with cur.copy(f"COPY {temp} ({','.join(columns)}) FROM STDIN") as copy:
            for value in values:
                copy.write_row(value)
        cur.execute(
            f"INSERT INTO {target} ({','.join(columns)}) SELECT {','.join(columns)} FROM {temp} ON CONFLICT DO NOTHING"
        )


def _db_catalog(db, root: Path) -> None:
    for filename, table, columns in [
        (
            "suppliers.csv",
            "core.dim_supplier",
            ["supplier_id", "supplier_name", "region", "lead_time_mean", "lead_time_sd"],
        ),
        (
            "warehouses.csv",
            "core.dim_warehouse",
            ["warehouse_id", "warehouse_name", "city", "latitude", "longitude", "climate_scale"],
        ),
        (
            "products.csv",
            "core.dim_product",
            [
                "sku",
                "category",
                "product_name",
                "supplier_id",
                "case_pack",
                "unit_cost",
                "unit_price",
            ],
        ),
    ]:
        with (root / "catalog" / filename).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        placeholders = ",".join(["%s"] * len(columns))
        db.cursor().executemany(
            f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
            [tuple(row[column] for column in columns) for row in rows],
        )


def _load_one(
    path: Path,
    root: Path,
    manifest: dict[str, dict],
    known: dict[str, set[str]],
    run_id: str,
    db=None,
) -> tuple[int, int, str]:
    key = str(path.relative_to(root)).replace("\\", "/")
    expected = manifest.get(key, {})
    good, bad = check_file(path, root, manifest, known, _expected_date(key, expected))
    if db:
        db.execute(
            "INSERT INTO ops.etl_file (file_key, sha256, feed_date, row_count, good_count, bad_count, status, etl_run_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            (
                key,
                expected.get("sha256", ""),
                expected.get("feed_date"),
                expected.get("row_count", 0),
                len(good),
                len(bad),
                "QUARANTINED" if bad else "LOADED",
                run_id,
            ),
        )
        for row in bad:
            db.execute(
                "INSERT INTO ops.quarantine_row (file_key,line_number,raw_row,reason_code,etl_run_id) VALUES (%s,%s,%s::jsonb,%s,%s)",
                (key, row["line_number"], json.dumps(row["raw"]), row["reason_code"], run_id),
            )
        if good:
            _load_with_copy(db, feed_type(key), good, run_id)
    return len(good), bad, key


def load_master(
    root: str,
    database_url: str | None = None,
    date_filter: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    pending: bool = False,
) -> dict:
    root_path = Path(root)
    with (root_path / "raw/manifest.csv").open(newline="") as handle:
        manifest = {row["file_key"]: row for row in csv.DictReader(handle)}
    known = _known_catalog(root_path)
    processed_path = root_path / ".etl_file.jsonl"
    processed = set()
    if processed_path.exists():
        with processed_path.open() as marker_file:
            processed = {
                json.loads(line)["file_key"] + "|" + json.loads(line)["sha256"]
                for line in marker_file
            }
    run_id = str(uuid.uuid4())
    results = {"run_id": run_id, "files": 0, "skipped": 0, "good_rows": 0, "bad_rows": 0}
    db = None
    if database_url or os.getenv("DATABASE_URL"):
        import psycopg

        db = psycopg.connect(database_url or os.environ["DATABASE_URL"])
        db.execute(
            "INSERT INTO ops.etl_run (etl_run_id, source) VALUES (%s,%s)", (run_id, str(root_path))
        )
        _db_catalog(db, root_path)
        db.commit()
    try:
        for path in sorted((root_path / "raw").glob("*/*.csv.gz")):
            key = str(path.relative_to(root_path)).replace("\\", "/")
            expected = manifest.get(key, {})
            feed_date = expected.get("feed_date")
            if date_filter and feed_date != date_filter:
                continue
            if date_from and feed_date < date_from:
                continue
            if date_to and feed_date > date_to:
                continue
            marker = key + "|" + expected.get("sha256", "")
            if marker in processed:
                results["skipped"] += 1
                continue
            if (
                db
                and db.execute(
                    "SELECT 1 FROM ops.etl_file WHERE file_key=%s AND sha256=%s",
                    (key, expected.get("sha256", "")),
                ).fetchone()
            ):
                results["skipped"] += 1
                continue
            good, bad_rows, _ = _load_one(path, root_path, manifest, known, run_id, db)
            if db:
                db.commit()
            results["files"] += 1
            results["good_rows"] += good
            results["bad_rows"] += len(bad_rows)
            if bad_rows:
                qpath = root_path / "quarantine/files" / key.removeprefix("raw/")
                qpath.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, qpath)
                with (root_path / "quarantine/rows.jsonl").open("a") as handle:
                    for row in bad_rows:
                        handle.write(
                            json.dumps({"file_key": key, "etl_run_id": run_id, **row}) + "\n"
                        )
            else:
                archive = root_path / "raw/archive" / key.removeprefix("raw/")
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, archive)
            with processed_path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "file_key": key,
                            "sha256": expected.get("sha256", ""),
                            "status": "QUARANTINED" if bad_rows else "LOADED",
                        }
                    )
                    + "\n"
                )
        if db:
            db.execute(
                "UPDATE ops.etl_run SET finished_at=now(), status='SUCCEEDED' WHERE etl_run_id=%s",
                (run_id,),
            )
            db.commit()
    except Exception:
        if db:
            db.rollback()
            db.execute(
                "UPDATE ops.etl_run SET finished_at=now(), status='FAILED' WHERE etl_run_id=%s",
                (run_id,),
            )
            db.commit()
        raise
    finally:
        if db:
            db.close()
    return results


def load_eval(root: str, database_url: str | None = None) -> dict:
    root_path = Path(root)
    with gzip.open(root_path / "eval/erp/planted_defects.csv.gz", "rt") as handle:
        defects = list(csv.DictReader(handle))
    disruptions = list(csv.DictReader((root_path / "eval/erp/disruptions.csv").open()))
    if database_url or os.getenv("DATABASE_URL"):
        import psycopg

        with psycopg.connect(database_url or os.environ["DATABASE_URL"]) as db:
            db.cursor().executemany(
                "INSERT INTO eval.planted_defect (defect_id,file_key,line_number,defect_type,expected_reason) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                [
                    (
                        x["defect_id"],
                        x["file_key"],
                        int(x["line_number"]),
                        x["defect_type"],
                        x["expected_reason"],
                    )
                    for x in defects
                ],
            )
            db.cursor().executemany(
                "INSERT INTO eval.disruption_truth (disruption_id,disruption_type,start_date,end_date,description,supplier_id,sku,warehouse_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                [
                    (
                        x["disruption_id"],
                        x["disruption_type"],
                        x["start_date"],
                        x["end_date"],
                        x["description"],
                        x["supplier_id"],
                        x["sku"],
                        x["warehouse_id"],
                    )
                    for x in disruptions
                ],
            )
    return {"planted_defects": len(defects), "disruptions": len(disruptions)}


def enqueue(root: str) -> dict:
    import pika

    root_path = Path(root)
    with (root_path / "raw/manifest.csv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    connection = pika.BlockingConnection(
        pika.URLParameters(
            os.getenv("RABBITMQ_URL", "amqp://supplychain:supplychain@localhost:5672/")
        )
    )
    channel = connection.channel()
    channel.exchange_declare(exchange="erp", exchange_type="topic", durable=True)
    count = 0
    for item in manifest:
        kind = feed_type(item["file_key"])
        channel.basic_publish(
            "erp",
            f"feed.{kind}",
            json.dumps({"file_key": item["file_key"], "feed_date": item["feed_date"]}).encode(),
            pika.BasicProperties(delivery_mode=2),
        )
        count += 1
    connection.close()
    return {"status": "queued", "messages": count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate, quarantine, and load ERP feeds")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("load-master", "load"):
        command_parser = sub.add_parser(command)
        command_parser.add_argument("--root", default="data")
        command_parser.add_argument("--date")
        command_parser.add_argument("--from", dest="date_from")
        command_parser.add_argument("--to", dest="date_to")
        command_parser.add_argument("--pending", action="store_true")
    eval_parser = sub.add_parser("load-eval")
    eval_parser.add_argument("--root", default="data")
    enqueue_parser = sub.add_parser("enqueue")
    enqueue_parser.add_argument("--root", default="data")
    sub.add_parser("status")
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--root", default="data")
    args = parser.parse_args()
    if args.command in {"load-master", "load"}:
        print(
            json.dumps(
                load_master(
                    args.root,
                    date_filter=args.date,
                    date_from=args.date_from,
                    date_to=args.date_to,
                    pending=args.pending,
                )
            )
        )
    elif args.command == "load-eval":
        print(json.dumps(load_eval(args.root)))
    elif args.command == "status":
        print(json.dumps({"status": "ready"}))
    elif args.command == "enqueue":
        print(json.dumps(enqueue(args.root)))
    else:
        FeedConsumer(
            lambda message: load_master(args.root, date_filter=message.get("feed_date"))
        ).run()


if __name__ == "__main__":
    main()
