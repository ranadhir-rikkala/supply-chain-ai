"""RabbitMQ topology and publisher for forecast and training jobs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

EXCHANGE = "ml"
QUEUE = "forecast.jobs"
ROUTING_KEY = "forecast"
FORECAST_JOB = "forecast"
TRAINING_JOB = "train"
JOB_TYPES = (FORECAST_JOB, TRAINING_JOB)


class JobPublishError(RuntimeError):
    """Raised when a job cannot be handed to RabbitMQ."""


def amqp_url() -> str:
    """Broker URL from AMQP_URL, falling back to RABBITMQ_URL."""
    return os.getenv("AMQP_URL") or os.getenv(
        "RABBITMQ_URL", "amqp://supplychain:supplychain@localhost:5672/"
    )


@dataclass(frozen=True)
class JobTopology:
    """Exchange and queue names; tests override them to get their own queues."""

    exchange: str = EXCHANGE
    queue: str = QUEUE
    routing_key: str = ROUTING_KEY

    @property
    def retry_queue(self) -> str:
        """Queue that holds a failed job until its retry delay expires."""
        return f"{self.queue}.retry"

    @property
    def dead_letter_queue(self) -> str:
        """Queue for jobs that failed every attempt."""
        return f"{self.queue}.dlq"

    @classmethod
    def from_env(cls) -> JobTopology:
        """Topology from FORECAST_JOBS_EXCHANGE and FORECAST_JOBS_QUEUE, defaulting to ml and forecast.jobs."""
        return cls(
            exchange=os.getenv("FORECAST_JOBS_EXCHANGE", EXCHANGE),
            queue=os.getenv("FORECAST_JOBS_QUEUE", QUEUE),
        )


def declare_topology(channel, topology: JobTopology | None = None) -> None:
    """Declare the exchange, main queue, retry queue and dead-letter queue."""
    topology = topology or JobTopology.from_env()
    channel.exchange_declare(exchange=topology.exchange, exchange_type="topic", durable=True)
    channel.queue_declare(queue=topology.queue, durable=True)
    channel.queue_bind(
        queue=topology.queue,
        exchange=topology.exchange,
        routing_key=f"{topology.routing_key}.#",
    )
    channel.queue_declare(
        queue=topology.retry_queue,
        durable=True,
        arguments={
            "x-dead-letter-exchange": topology.exchange,
            "x-dead-letter-routing-key": topology.routing_key,
        },
    )
    channel.queue_declare(queue=topology.dead_letter_queue, durable=True)


class RabbitJobPublisher:
    """Publishes a job per call on a short-lived connection."""

    def __init__(
        self,
        url: str | None = None,
        timeout_seconds: float = 5.0,
        topology: JobTopology | None = None,
    ):
        self.url = url or amqp_url()
        self.timeout_seconds = timeout_seconds
        self.topology = topology or JobTopology.from_env()

    def publish(self, message: dict, request_id: str | None = None) -> None:
        """Publish a persistent job message; raises JobPublishError if the broker refuses it."""
        import pika

        parameters = pika.URLParameters(self.url)
        parameters.socket_timeout = self.timeout_seconds
        parameters.blocked_connection_timeout = self.timeout_seconds
        parameters.connection_attempts = 1
        try:
            connection = pika.BlockingConnection(parameters)
            try:
                channel = connection.channel()
                # Publisher confirms make the broker accept the message before the API answers 202.
                channel.confirm_delivery()
                # Declaring on publish keeps jobs even before any worker has started.
                declare_topology(channel, self.topology)
                channel.basic_publish(
                    self.topology.exchange,
                    f"{self.topology.routing_key}.{message['job_type']}",
                    json.dumps(message, default=str).encode(),
                    pika.BasicProperties(
                        delivery_mode=2,
                        content_type="application/json",
                        correlation_id=request_id,
                        headers={"x-request-id": request_id} if request_id else None,
                    ),
                    mandatory=True,
                )
            finally:
                connection.close()
        except (pika.exceptions.AMQPError, OSError) as error:
            raise JobPublishError(
                f"could not publish {message['job_type']} job: {error}"
            ) from error
