"""Alert engine for health anomaly notifications."""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HealthAlert:
    alert_id: str
    severity: AlertSeverity
    country_code: str
    indicator_code: str
    anomaly_score: float
    value: float
    expected_range: tuple
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "country_code": self.country_code,
            "indicator_code": self.indicator_code,
            "anomaly_score": self.anomaly_score,
            "value": self.value,
            "expected_range": list(self.expected_range),
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


class AlertEngine:
    """Generate and manage health anomaly alerts with deduplication."""

    SEVERITY_THRESHOLDS = {
        AlertSeverity.CRITICAL: 0.95,
        AlertSeverity.HIGH: 0.85,
        AlertSeverity.WARNING: 0.70,
        AlertSeverity.INFO: 0.50,
    }

    def __init__(self, cooldown_minutes: int = 60, max_alerts_per_hour: int = 100):
        self.cooldown_minutes = cooldown_minutes
        self.max_alerts_per_hour = max_alerts_per_hour
        self._recent_alerts: Dict[str, datetime] = {}
        self._alert_count = 0

    def evaluate(
        self,
        prediction: Dict[str, Any],
        country_code: str,
        indicator_code: str,
        value: float,
    ) -> Optional[HealthAlert]:
        """Evaluate prediction and generate alert if warranted."""
        if not prediction.get("is_anomaly", False):
            return None

        score = prediction["anomaly_score"]
        severity = self._classify_severity(score)

        # Dedup: skip if same (country, indicator) alerted recently
        key = f"{country_code}:{indicator_code}"
        if key in self._recent_alerts:
            elapsed = (datetime.utcnow() - self._recent_alerts[key]).total_seconds() / 60
            if elapsed < self.cooldown_minutes:
                return None

        alert = HealthAlert(
            alert_id=f"HA-{self._alert_count:08d}",
            severity=severity,
            country_code=country_code,
            indicator_code=indicator_code,
            anomaly_score=score,
            value=value,
            expected_range=(0, 0),  # populated from feature store
            context=prediction.get("component_scores", {}),
        )

        self._recent_alerts[key] = datetime.utcnow()
        self._alert_count += 1
        logger.info("Alert %s: %s %s/%s score=%.3f",
                     alert.alert_id, severity.value, country_code, indicator_code, score)
        return alert

    def _classify_severity(self, score: float) -> AlertSeverity:
        for severity, threshold in self.SEVERITY_THRESHOLDS.items():
            if score >= threshold:
                return severity
        return AlertSeverity.INFO
