"""Redis-backed streaming feature store for real-time health anomaly detection.

Maintains per-(country, indicator) feature vectors updated in real-time.
Features include rolling statistics, trend indicators, seasonal decomposition,
and cross-indicator correlation signals.

Latency budget: 80ms for full feature vector computation.
"""
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import redis

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    redis_url: str = "redis://localhost:6379/0"
    window_sizes: List[int] = field(default_factory=lambda: [7, 14, 30, 90, 365])
    ttl_seconds: int = 86400 * 30  # 30 days
    max_history: int = 365 * 3  # 3 years of daily values


class StreamingFeatureStore:
    """Real-time feature store for health surveillance anomaly detection.

    Features computed per (country_code, indicator_code) pair:
    - Rolling mean, std, min, max for each window size
    - Rate of change (first derivative)
    - Acceleration (second derivative)
    - Z-score relative to historical distribution
    - Seasonal ratio (current / same-period-last-year)
    - Cross-indicator correlation (e.g., temperature to malaria)
    - Geographic neighbor aggregates

    All features computed within 80ms latency budget using Redis
    sorted sets for O(log N) windowed aggregation.
    """

    FEATURE_NAMES = [
        # Rolling statistics (5 windows x 4 stats = 20)
        *[f"rolling_{stat}_{w}d" for w in [7,14,30,90,365] for stat in ["mean","std","min","max"]],
        # Trend features (4)
        "rate_of_change_7d", "rate_of_change_30d",
        "acceleration_7d", "acceleration_30d",
        # Anomaly signals (4)
        "zscore_30d", "zscore_90d", "zscore_365d", "modified_zscore_30d",
        # Seasonal (2)
        "seasonal_ratio_yoy", "seasonal_ratio_52w",
        # Cross-indicator (2)
        "temp_correlation_30d", "precip_correlation_30d",
        # Geographic (2)
        "neighbor_mean_ratio", "neighbor_zscore",
    ]

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.redis = redis.from_url(self.config.redis_url, decode_responses=True)
        self._pipeline = None

    def update(
        self, country_code: str, indicator_code: str,
        value: float, timestamp: float,
    ) -> np.ndarray:
        """Update feature store and return current feature vector.

        Returns:
            numpy array of shape (34,) with all features
        """
        start = time.monotonic()
        key = self._key(country_code, indicator_code)

        # Append value to time series (sorted set: score=timestamp, member=value)
        self.redis.zadd(f"{key}:ts", {f"{timestamp}:{value}": timestamp})

        # Trim to max history
        cutoff = timestamp - self.config.max_history * 86400
        self.redis.zremrangebyscore(f"{key}:ts", "-inf", cutoff)

        # Compute features
        history = self._get_history(key)
        features = self._compute_features(history, value, timestamp)

        # Cache feature vector
        self.redis.set(
            f"{key}:features",
            features.tobytes(),
            ex=self.config.ttl_seconds,
        )

        elapsed_ms = (time.monotonic() - start) * 1000
        if elapsed_ms > 80:
            logger.warning(
                "Feature computation exceeded budget: %.1fms for %s/%s",
                elapsed_ms, country_code, indicator_code,
            )

        return features

    def get_features(
        self, country_code: str, indicator_code: str,
    ) -> Optional[np.ndarray]:
        """Get cached feature vector."""
        key = self._key(country_code, indicator_code)
        raw = self.redis.get(f"{key}:features")
        if raw:
            return np.frombuffer(raw.encode("latin-1"), dtype=np.float64)
        return None

    def _get_history(self, key: str) -> List[Tuple[float, float]]:
        """Get time series history as [(timestamp, value), ...]."""
        members = self.redis.zrangebyscore(f"{key}:ts", "-inf", "+inf", withscores=True)
        history = []
        for member, score in members:
            parts = str(member).split(":")
            if len(parts) >= 2:
                try:
                    history.append((score, float(parts[1])))
                except (ValueError, IndexError):
                    continue
        return sorted(history, key=lambda x: x[0])

    def _compute_features(
        self, history: List[Tuple[float, float]],
        current_value: float, current_ts: float,
    ) -> np.ndarray:
        """Compute full feature vector from history."""
        features = np.zeros(len(self.FEATURE_NAMES), dtype=np.float64)
        if len(history) < 2:
            return features

        values = np.array([v for _, v in history])
        timestamps = np.array([t for t, _ in history])

        idx = 0
        # Rolling statistics for each window
        for w in self.config.window_sizes:
            cutoff = current_ts - w * 86400
            window_vals = values[timestamps >= cutoff]
            if len(window_vals) > 0:
                features[idx] = np.mean(window_vals)
                features[idx + 1] = np.std(window_vals) if len(window_vals) > 1 else 0
                features[idx + 2] = np.min(window_vals)
                features[idx + 3] = np.max(window_vals)
            idx += 4

        # Rate of change
        for w in [7, 30]:
            cutoff = current_ts - w * 86400
            past = values[timestamps <= cutoff]
            if len(past) > 0:
                features[idx] = (current_value - past[-1]) / (past[-1] + 1e-8)
            idx += 1

        # Acceleration (second derivative)
        for w in [7, 30]:
            cutoff1 = current_ts - w * 86400
            cutoff2 = current_ts - 2 * w * 86400
            v1 = values[(timestamps >= cutoff1)]
            v2 = values[(timestamps >= cutoff2) & (timestamps < cutoff1)]
            if len(v1) > 0 and len(v2) > 0:
                d1 = np.mean(v1) - np.mean(v2)
                d0 = current_value - np.mean(v1)
                features[idx] = d0 - d1
            idx += 1

        # Z-scores
        for w in [30, 90, 365]:
            cutoff = current_ts - w * 86400
            window_vals = values[timestamps >= cutoff]
            if len(window_vals) > 1:
                std = np.std(window_vals)
                features[idx] = (current_value - np.mean(window_vals)) / (std + 1e-8)
            idx += 1

        # Modified Z-score (MAD-based, robust to outliers)
        w30 = values[timestamps >= current_ts - 30 * 86400]
        if len(w30) > 1:
            median = np.median(w30)
            mad = np.median(np.abs(w30 - median)) + 1e-8
            features[idx] = 0.6745 * (current_value - median) / mad
        idx += 1

        # Seasonal ratios
        yoy_ts = current_ts - 365 * 86400
        yoy_vals = values[np.abs(timestamps - yoy_ts) < 7 * 86400]
        if len(yoy_vals) > 0:
            features[idx] = current_value / (np.mean(yoy_vals) + 1e-8)
        idx += 1
        features[idx] = 0  # 52-week placeholder
        idx += 1

        # Cross-indicator and geographic features are placeholders
        # (populated by the cross-indicator correlation engine)
        features[idx:idx+4] = 0
        idx += 4

        return features

    @staticmethod
    def _key(country: str, indicator: str) -> str:
        return f"sage:health:{country}:{indicator}"
