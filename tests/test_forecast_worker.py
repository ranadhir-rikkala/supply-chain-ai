"""Forecast worker: job dispatch, retries and the dead-letter path on an isolated queue."""

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import date, timedelta

import pytest
from forecast_fixtures import FAST_PARAMS, ensure_dimensions, synthetic_panel

from pipeline.forecast.artifacts import save_artifacts
from pipeline.forecast.features import SeriesHistory
from pipeline.forecast.jobs import JobTopology, RabbitJobPublisher
from pipeline.forecast.model import train
from pipeline.forecast.registry import GateDecision, TrainingWindows, activate_model, register_model
from pipeline.forecast.runs import create_run
from pipeline.forecast.worker import UnknownJobType, handle_dead_letter, handle_job
from pipeline.storage import LocalStore


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_forecast_job_runs_the_requested_run(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "pipeline.forecast.runs.execute_run",
        lambda connection, store, run_id: calls.append(run_id) or {"status": "SUCCEEDED"},
    )
    run_id = uuid.uuid4()
    result = handle_job(
        {"job_type": "forecast", "run_id": str(run_id)}, FakeConnection, lambda: "store"
    )
    assert result == {"status": "SUCCEEDED"}
    assert calls == [run_id]


def test_training_job_trains_and_registers(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "pipeline.forecast.training.train_and_register",
        lambda connection, store, requested_by: calls.append(requested_by) or {"status": "ok"},
    )
    handle_job({"job_type": "train", "requested_by": "ada"}, FakeConnection, lambda: "store")
    assert calls == ["ada"]


def test_unknown_job_type_fails_so_it_is_retried_then_dead_lettered():
    with pytest.raises(UnknownJobType):
        handle_job({"job_type": "dance"}, FakeConnection, lambda: "store")


def test_dead_lettered_forecast_marks_the_run_failed(monkeypatch):
    failed = []
    monkeypatch.setattr(
        "pipeline.forecast.runs.mark_failed",
        lambda connection, run_id, error: failed.append((run_id, error)),
    )
    run_id = uuid.uuid4()
    handle_dead_letter(
        {"job_type": "forecast", "run_id": str(run_id)}, "WORKER:boom", FakeConnection
    )
    handle_dead_letter({"job_type": "train"}, "WORKER:boom", FakeConnection)
    handle_dead_letter(None, "POISON:bad json", FakeConnection)
    assert failed == [(run_id, "dead-lettered: WORKER:boom")]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextmanager
def _worker_process(database_url: str, amqp_url: str, store_root, topology: JobTopology):
    env = {
        **os.environ,
        "DATABASE_URL": database_url,
        "AMQP_URL": amqp_url,
        "LOCAL_STORE_ROOT": str(store_root),
        "STORE": "local",
        "METRICS_PORT": str(_free_port()),
        "FORECAST_JOBS_EXCHANGE": topology.exchange,
        "FORECAST_JOBS_QUEUE": topology.queue,
    }
    process = subprocess.Popen(
        [sys.executable, "-c", "from pipeline.forecast.cli import main; main(['worker'])"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        yield process
    finally:
        process.terminate()
        process.wait(timeout=20)


def _wait_for_status(connection, run_id, statuses, timeout_seconds=60) -> str:
    deadline = time.monotonic() + timeout_seconds
    status = None
    while time.monotonic() < deadline:
        connection.rollback()
        status = connection.execute(
            "SELECT status FROM ml.forecast_run WHERE run_id = %s", (run_id,)
        ).fetchone()[0]
        if status in statuses:
            return status
        time.sleep(0.5)
    return status


def _publish(amqp_url: str, topology: JobTopology, message: dict) -> None:
    RabbitJobPublisher(amqp_url, topology=topology).publish(message, "worker-test")


@pytest.fixture
def isolated_topology():
    amqp_url = os.getenv("TEST_AMQP_URL")
    if not amqp_url:
        pytest.skip("TEST_AMQP_URL is not configured")
    import pika

    suffix = uuid.uuid4().hex[:12]
    topology = JobTopology(exchange=f"ml.test.{suffix}", queue=f"forecast.jobs.test.{suffix}")
    yield topology
    connection = pika.BlockingConnection(pika.URLParameters(amqp_url))
    try:
        channel = connection.channel()
        for queue in (topology.queue, topology.retry_queue, topology.dead_letter_queue):
            channel.queue_delete(queue=queue)
        channel.exchange_delete(exchange=topology.exchange)
    finally:
        connection.close()


def _queue_depth(amqp_url: str, queue: str) -> int:
    import pika

    connection = pika.BlockingConnection(pika.URLParameters(amqp_url))
    try:
        return connection.channel().queue_declare(queue=queue, passive=True).method.message_count
    finally:
        connection.close()


@pytest.fixture
def services(tmp_path):
    database_url = os.getenv("TEST_DATABASE_URL")
    amqp_url = os.getenv("TEST_AMQP_URL")
    if not (database_url and amqp_url):
        pytest.skip("TEST_DATABASE_URL and TEST_AMQP_URL are not configured")
    import psycopg

    panel = synthetic_panel(days=420)
    with psycopg.connect(database_url) as connection:
        connection.execute("TRUNCATE ml.forecast_run, ml.model_version CASCADE")
        connection.execute("DELETE FROM core.fact_sales_daily WHERE sale_id LIKE 'S-WRK-%%'")
        connection.commit()
        ensure_dimensions(connection, panel)
        with (
            connection.cursor() as cursor,
            cursor.copy(
                "COPY core.fact_sales_daily (feed_date, sale_id, sku, warehouse_id, units_sold, "
                "unit_price, etl_run_id) FROM STDIN"
            ) as copy,
        ):
            for series in range(panel.series_count):
                for day in range(panel.day_count):
                    copy.write_row(
                        (
                            panel.start + timedelta(days=day),
                            f"S-WRK-{series}-{day}",
                            panel.skus[series],
                            panel.warehouses[series],
                            int(panel.units[series, day]),
                            2.0,
                            "00000000-0000-0000-0000-00000000000a",
                        )
                    )
        categories = tuple(sorted(set(panel.categories)))
        warehouses = tuple(sorted(set(panel.warehouses)))
        history = SeriesHistory.from_panel(panel, categories, warehouses)
        model = train(history, categories, warehouses, panel.day_count - 1, FAST_PARAMS, 20)
        version = "20251231000000-worker01"
        key = save_artifacts(LocalStore(tmp_path), version, model, {})
        windows = TrainingWindows(
            date(2023, 1, 1), date(2023, 6, 30), date(2023, 7, 1), date(2024, 2, 24)
        )
        register_model(
            connection, version, {}, [], GateDecision(True, ()), "fp", windows, key, {}, "tester"
        )
        connection.commit()
        activate_model(connection, version, "tester", "worker test")
        yield {
            "connection": connection,
            "amqp_url": amqp_url,
            "database_url": database_url,
            "store_root": tmp_path,
            "panel": panel,
            "version": version,
        }
        connection.rollback()
        connection.execute("DELETE FROM core.fact_sales_daily WHERE sale_id LIKE 'S-WRK-%%'")
        connection.execute("TRUNCATE ml.forecast_run, ml.model_version CASCADE")
        connection.commit()


@pytest.mark.integration
def test_worker_runs_a_queued_forecast(services, isolated_topology):
    connection = services["connection"]
    run = create_run(connection, "pat", origin_date=date(2024, 2, 24), horizon_days=7)
    with _worker_process(
        services["database_url"], services["amqp_url"], services["store_root"], isolated_topology
    ):
        _publish(
            services["amqp_url"],
            isolated_topology,
            {"job_type": "forecast", "run_id": str(run["run_id"])},
        )
        status = _wait_for_status(connection, run["run_id"], {"SUCCEEDED", "FAILED"})
    assert status == "SUCCEEDED"
    panel = services["panel"]
    pairs = [list(panel.skus), list(panel.warehouses)]
    written = connection.execute(
        "SELECT count(*) FROM ml.forecast f "
        "JOIN unnest(%s::text[], %s::text[]) AS s(sku, warehouse_id) USING (sku, warehouse_id) "
        "WHERE run_id = %s",
        (*pairs, run["run_id"]),
    ).fetchone()[0]
    assert written == services["panel"].series_count * 7


@pytest.mark.integration
def test_failing_forecast_is_dead_lettered_and_marked_failed(services, isolated_topology):
    connection = services["connection"]
    run = create_run(connection, "pat", origin_date=date(2023, 3, 1), horizon_days=7)
    with _worker_process(
        services["database_url"], services["amqp_url"], services["store_root"], isolated_topology
    ):
        _publish(
            services["amqp_url"],
            isolated_topology,
            {"job_type": "forecast", "run_id": str(run["run_id"])},
        )
        status = _wait_for_status(connection, run["run_id"], {"FAILED", "SUCCEEDED"})
    assert status == "FAILED"
    error = connection.execute(
        "SELECT error FROM ml.forecast_run WHERE run_id = %s", (run["run_id"],)
    ).fetchone()[0]
    assert error.startswith("dead-lettered: WORKER:")
    assert json.dumps(error)
    assert _queue_depth(services["amqp_url"], isolated_topology.dead_letter_queue) == 1
