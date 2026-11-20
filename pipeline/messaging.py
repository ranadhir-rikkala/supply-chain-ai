from __future__ import annotations

import json
import logging
import os
import signal
from collections.abc import Callable

try:
    from prometheus_client import Counter, Gauge, start_http_server
except ImportError:  # Optional at import time; production installs the declared dependency.

    class _Metric:
        def inc(self, amount: float = 1) -> None:
            return None

        def set(self, value: float) -> None:
            return None

    def Counter(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _Metric()

    def Gauge(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _Metric()

    def start_http_server(*args, **kwargs):  # type: ignore[no-untyped-def]
        return None


MESSAGES_PROCESSED = Counter("scdata_messages_processed_total", "Messages successfully processed")
MESSAGES_RETRIED = Counter("scdata_messages_retried_total", "Messages sent to retry")
MESSAGES_FAILED = Counter("scdata_messages_failed_total", "Messages sent to the dead-letter queue")
MESSAGES_POISON = Counter("scdata_messages_poison_total", "Malformed messages sent directly to DLQ")
CONSUMER_READY = Gauge("scdata_consumer_ready", "Whether the consumer is connected")


logger = logging.getLogger(__name__)


class PoisonMessage(ValueError):
    """A message that cannot safely be retried, such as invalid JSON."""


class FeedConsumer:
    """Reusable RabbitMQ consumer for ETL and Forecasting workers.

    The retry queue uses per-message expiration. RabbitMQ dead-letters expired
    messages back to ``etl.feed``; retry delays are 1, 2, and 4 seconds. The
    fourth failure is routed to the durable dead-letter queue.
    """

    def __init__(
        self,
        callback: Callable[[dict], None],
        url: str | None = None,
        metrics_port: int | None = None,
        *,
        exchange: str = "erp",
        queue: str = "etl.feed",
        routing_key: str = "feed",
        required_field: str = "file_key",
        max_attempts: int = 4,
        on_dead_letter: Callable[[dict | None, str], None] | None = None,
    ):
        import pika

        self.pika = pika
        self.callback = callback
        self.stopping = False
        self.queue = queue
        self.retry_queue = f"{queue}.retry"
        self.dead_letter_queue = f"{queue}.dlq"
        self.required_field = required_field
        self.max_attempts = max_attempts
        self.on_dead_letter = on_dead_letter
        self.connection = pika.BlockingConnection(
            pika.URLParameters(
                url
                or os.getenv("AMQP_URL")
                or os.getenv("RABBITMQ_URL", "amqp://supplychain:supplychain@localhost:5672/")
            )
        )
        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange=exchange, exchange_type="topic", durable=True)
        self.channel.queue_declare(queue=queue, durable=True)
        self.channel.queue_bind(queue=queue, exchange=exchange, routing_key=f"{routing_key}.#")
        self.channel.queue_declare(
            queue=self.retry_queue,
            durable=True,
            arguments={
                "x-dead-letter-exchange": exchange,
                "x-dead-letter-routing-key": routing_key,
            },
        )
        self.channel.queue_declare(queue=self.dead_letter_queue, durable=True)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=queue, on_message_callback=self._consume, auto_ack=False)
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)
        start_http_server(metrics_port or int(os.getenv("METRICS_PORT", "9100")))
        CONSUMER_READY.set(1)

    def _shutdown(self, signum, frame) -> None:
        del signum, frame
        self.stopping = True
        if self.connection.is_open:
            self.connection.add_callback_threadsafe(self.channel.stop_consuming)

    def _publish_dlq(self, body: bytes, headers: dict, reason: str, message: dict | None) -> None:
        headers = {**headers, "x-failure-reason": reason}
        self.channel.basic_publish(
            "",
            self.dead_letter_queue,
            body,
            self.pika.BasicProperties(delivery_mode=2, headers=headers),
        )
        MESSAGES_FAILED.inc()
        if self.on_dead_letter:
            try:
                self.on_dead_letter(message, reason)
            except Exception:
                logger.exception("dead-letter callback failed")

    def _consume(self, channel, method, properties, body) -> None:
        headers = properties.headers or {}
        try:
            message = json.loads(body)
            if not isinstance(message, dict) or not message.get(self.required_field):
                raise PoisonMessage(f"message must be an object with {self.required_field}")
        except (json.JSONDecodeError, UnicodeDecodeError, PoisonMessage, TypeError) as error:
            channel.basic_ack(method.delivery_tag)
            self._publish_dlq(body, headers, f"POISON:{error}", None)
            MESSAGES_POISON.inc()
            return
        try:
            self.callback(message)
            channel.basic_ack(method.delivery_tag)
            MESSAGES_PROCESSED.inc()
        except Exception as error:  # noqa: BLE001 - all worker failures are retryable until the attempt limit
            attempt = int(headers.get("x-attempts", 0)) + 1
            channel.basic_ack(method.delivery_tag)
            if attempt >= self.max_attempts:
                self._publish_dlq(
                    body, {**headers, "x-attempts": attempt}, f"WORKER:{error}", message
                )
            else:
                delay_ms = 1000 * (2 ** (attempt - 1))
                retry_headers = {**headers, "x-attempts": attempt}
                channel.basic_publish(
                    "",
                    self.retry_queue,
                    body,
                    self.pika.BasicProperties(
                        delivery_mode=2, expiration=str(delay_ms), headers=retry_headers
                    ),
                )
                MESSAGES_RETRIED.inc()

    def run(self) -> None:
        try:
            self.channel.start_consuming()
        finally:
            CONSUMER_READY.set(0)
            if self.connection.is_open:
                self.connection.close()
