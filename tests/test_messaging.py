import json
from types import SimpleNamespace

import pytest

from pipeline.messaging import FeedConsumer


class RecordingChannel:
    def __init__(self):
        self.acks = []
        self.published = []

    def basic_ack(self, tag):
        self.acks.append(tag)

    def basic_publish(self, exchange, routing_key, body, properties):
        self.published.append((exchange, routing_key, body, properties))


class Properties(SimpleNamespace):
    pass


def _consumer(callback, dead_letters=None):
    import pika

    consumer = object.__new__(FeedConsumer)
    consumer.pika = pika
    consumer.callback = callback
    consumer.queue = "forecast.jobs"
    consumer.retry_queue = "forecast.jobs.retry"
    consumer.dead_letter_queue = "forecast.jobs.dlq"
    consumer.required_field = "job_type"
    consumer.max_attempts = 4
    consumer.on_dead_letter = (
        (lambda message, reason: dead_letters.append((message, reason)))
        if dead_letters is not None
        else None
    )
    consumer.channel = RecordingChannel()
    return consumer


def _deliver(consumer, message, attempts=None):
    body = message if isinstance(message, bytes) else json.dumps(message).encode()
    headers = {"x-attempts": attempts} if attempts is not None else None
    consumer._consume(
        consumer.channel, SimpleNamespace(delivery_tag=7), Properties(headers=headers), body
    )


def test_successful_message_is_acknowledged_once():
    handled = []
    consumer = _consumer(handled.append)
    _deliver(consumer, {"job_type": "forecast"})
    assert handled == [{"job_type": "forecast"}]
    assert consumer.channel.acks == [7]
    assert consumer.channel.published == []


def test_failure_goes_to_the_retry_queue_directly_with_backoff():
    def fail(message):
        raise RuntimeError("database down")

    consumer = _consumer(fail)
    _deliver(consumer, {"job_type": "forecast"}, attempts=1)
    exchange, routing_key, _, properties = consumer.channel.published[0]
    assert (exchange, routing_key) == ("", "forecast.jobs.retry")
    assert properties.expiration == "2000"
    assert properties.headers["x-attempts"] == 2


@pytest.mark.parametrize("message", [b"not json", {"other": 1}])
def test_poison_message_goes_straight_to_the_dead_letter_queue(message):
    dead = []
    consumer = _consumer(lambda message: None, dead)
    _deliver(consumer, message)
    assert consumer.channel.published[0][:2] == ("", "forecast.jobs.dlq")
    assert dead[0][0] is None
    assert dead[0][1].startswith("POISON:")


def test_fourth_failure_is_dead_lettered_and_reported():
    def fail(message):
        raise RuntimeError("model missing")

    dead = []
    consumer = _consumer(fail, dead)
    _deliver(consumer, {"job_type": "forecast", "run_id": "r1"}, attempts=3)
    exchange, routing_key, _, properties = consumer.channel.published[0]
    assert (exchange, routing_key) == ("", "forecast.jobs.dlq")
    assert properties.headers["x-failure-reason"] == "WORKER:model missing"
    assert dead == [({"job_type": "forecast", "run_id": "r1"}, "WORKER:model missing")]
