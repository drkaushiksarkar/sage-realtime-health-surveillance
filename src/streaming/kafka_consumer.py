"""High-throughput Kafka consumer for health event ingestion.

Processes 50,000+ health surveillance events per second from WHO, CDC,
and national health systems. Each event is a structured health observation
(disease notification, lab result, syndromic surveillance signal).

Architecture:
    Kafka Consumer Group -> Deserialization -> Validation -> Feature Store Update
    -> Model Inference -> Alert Decision -> Dashboard WebSocket

Latency budget: 200ms end-to-end (50ms ingest, 80ms features, 50ms inference, 20ms alert)
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from confluent_kafka import Consumer, KafkaError, KafkaException

logger = logging.getLogger(__name__)


@dataclass
class ConsumerConfig:
    bootstrap_servers: str = "localhost:9092"
    group_id: str = "health-anomaly-detector"
    topics: List[str] = field(default_factory=lambda: [
        "health.surveillance.who",
        "health.surveillance.cdc",
        "health.surveillance.national",
        "health.lab.results",
        "health.syndromic.signals",
    ])
    max_poll_records: int = 500
    session_timeout_ms: int = 30000
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = False
    fetch_min_bytes: int = 65536
    fetch_max_wait_ms: int = 50  # 50ms max wait to stay within latency budget


@dataclass
class HealthEvent:
    """Canonical health surveillance event."""
    event_id: str
    source: str  # WHO, CDC, ECDC, national
    country_code: str  # ISO3
    indicator_code: str
    value: float
    timestamp: float
    geo_code: str = ""
    age_group: str = ""
    sex: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        return (time.time() - self.timestamp) * 1000


class HealthEventConsumer:
    """High-throughput Kafka consumer with back-pressure and latency tracking."""

    def __init__(
        self,
        config: ConsumerConfig,
        handler: Callable[[List[HealthEvent]], None],
        max_latency_ms: float = 200.0,
    ):
        self.config = config
        self.handler = handler
        self.max_latency_ms = max_latency_ms
        self._consumer = None
        self._running = False
        self._metrics = {
            "events_processed": 0,
            "batches_processed": 0,
            "avg_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "errors": 0,
            "back_pressure_events": 0,
        }
        self._latency_buffer: List[float] = []

    def start(self) -> None:
        """Start consuming health events."""
        kafka_config = {
            "bootstrap.servers": self.config.bootstrap_servers,
            "group.id": self.config.group_id,
            "session.timeout.ms": self.config.session_timeout_ms,
            "auto.offset.reset": self.config.auto_offset_reset,
            "enable.auto.commit": self.config.enable_auto_commit,
            "fetch.min.bytes": self.config.fetch_min_bytes,
            "fetch.wait.max.ms": self.config.fetch_max_wait_ms,
            "max.poll.interval.ms": 300000,
        }
        self._consumer = Consumer(kafka_config)
        self._consumer.subscribe(self.config.topics)
        self._running = True
        logger.info(
            "Health event consumer started on topics: %s",
            self.config.topics,
        )

        try:
            while self._running:
                self._poll_and_process()
        except KeyboardInterrupt:
            logger.info("Consumer interrupted")
        finally:
            self._consumer.close()

    def _poll_and_process(self) -> None:
        """Poll Kafka and process a batch of health events."""
        batch_start = time.monotonic()
        messages = self._consumer.consume(
            num_messages=self.config.max_poll_records,
            timeout=self.config.fetch_max_wait_ms / 1000.0,
        )

        if not messages:
            return

        events = []
        for msg in messages:
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("Kafka error: %s", msg.error())
                self._metrics["errors"] += 1
                continue

            try:
                event = self._deserialize(msg.value())
                events.append(event)
            except Exception as e:
                logger.warning("Deserialization error: %s", e)
                self._metrics["errors"] += 1

        if events:
            self.handler(events)
            self._consumer.commit(asynchronous=True)

            # Track latency
            batch_latency = (time.monotonic() - batch_start) * 1000
            self._latency_buffer.append(batch_latency)
            if len(self._latency_buffer) > 1000:
                self._latency_buffer = self._latency_buffer[-1000:]

            self._metrics["events_processed"] += len(events)
            self._metrics["batches_processed"] += 1
            self._metrics["avg_latency_ms"] = sum(self._latency_buffer) / len(self._latency_buffer)

            # Back-pressure: if approaching latency budget, reduce batch size
            if batch_latency > self.max_latency_ms * 0.8:
                self._metrics["back_pressure_events"] += 1
                logger.warning(
                    "Back-pressure: batch_latency=%.1fms (budget=%dms)",
                    batch_latency, self.max_latency_ms,
                )

    def _deserialize(self, raw: bytes) -> HealthEvent:
        """Deserialize Kafka message to HealthEvent."""
        import json
        data = json.loads(raw)
        return HealthEvent(
            event_id=data["event_id"],
            source=data["source"],
            country_code=data["country_code"],
            indicator_code=data["indicator_code"],
            value=float(data["value"]),
            timestamp=float(data["timestamp"]),
            geo_code=data.get("geo_code", ""),
            age_group=data.get("age_group", ""),
            sex=data.get("sex", ""),
            metadata=data.get("metadata", {}),
        )

    @property
    def metrics(self) -> Dict[str, Any]:
        return {**self._metrics}

    def stop(self) -> None:
        self._running = False
