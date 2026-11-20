"""RabbitMQ worker that runs forecast and training jobs."""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable

from prometheus_client import Counter, Histogram

from pipeline.forecast.jobs import FORECAST_JOB, TRAINING_JOB, JobTopology

logger = logging.getLogger(__name__)

JOBS_TOTAL = Counter(
    "forecast_worker_jobs_total", "Jobs handled by the forecast worker", ["job_type", "outcome"]
)
JOB_SECONDS = Histogram(
    "forecast_worker_job_duration_seconds",
    "Time spent on a forecast or training job",
    ["job_type"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)
DEAD_LETTERED = Counter(
    "forecast_worker_dead_lettered_total", "Jobs that ended in the dead-letter queue", ["job_type"]
)


class UnknownJobType(ValueError):
    """Raised for a job_type the worker does not handle."""


def _connect():
    import psycopg

    return psycopg.connect(os.environ["DATABASE_URL"])


def _store():
    from pipeline.forecast.artifacts import default_store

    return default_store()


def handle_job(message: dict, connect: Callable = _connect, store_factory: Callable = _store):
    """Run one forecast or training job; any exception makes the consumer retry it."""
    job_type = message.get("job_type")
    if job_type not in (FORECAST_JOB, TRAINING_JOB):
        raise UnknownJobType(f"unknown job_type {job_type!r}")
    started = time.perf_counter()
    logger.info("job started", extra={"job_type": job_type, "job_id": message.get("job_id")})
    try:
        with connect() as connection:
            if job_type == FORECAST_JOB:
                from pipeline.forecast.runs import execute_run

                result = execute_run(connection, store_factory(), uuid.UUID(message["run_id"]))
            else:
                from pipeline.forecast.training import train_and_register

                result = train_and_register(
                    connection, store_factory(), message.get("requested_by", "worker")
                )
    except Exception:
        JOBS_TOTAL.labels(job_type, "failed").inc()
        logger.exception(
            "job failed", extra={"job_type": job_type, "job_id": message.get("job_id")}
        )
        raise
    JOBS_TOTAL.labels(job_type, "succeeded").inc()
    JOB_SECONDS.labels(job_type).observe(time.perf_counter() - started)
    logger.info("job finished", extra={"job_type": job_type, "job_id": message.get("job_id")})
    return result


def handle_dead_letter(message: dict | None, reason: str, connect: Callable = _connect) -> None:
    """Record a dead-lettered job and mark its forecast run FAILED."""
    job_type = (message or {}).get("job_type", "unknown")
    DEAD_LETTERED.labels(str(job_type)).inc()
    logger.error("job dead-lettered", extra={"job_type": job_type, "reason": reason})
    if job_type != FORECAST_JOB or not message.get("run_id"):
        return
    from pipeline.forecast.runs import mark_failed

    with connect() as connection:
        mark_failed(connection, uuid.UUID(message["run_id"]), f"dead-lettered: {reason}")


def run_worker() -> None:
    """Consume jobs until SIGTERM, using the shared consumer with retries and a dead-letter queue."""
    from pipeline.messaging import FeedConsumer

    topology = JobTopology.from_env()
    consumer = FeedConsumer(
        handle_job,
        exchange=topology.exchange,
        queue=topology.queue,
        routing_key=topology.routing_key,
        required_field="job_type",
        on_dead_letter=handle_dead_letter,
    )
    logger.info(
        "forecast worker consuming",
        extra={"exchange": topology.exchange, "queue": topology.queue},
    )
    consumer.run()
