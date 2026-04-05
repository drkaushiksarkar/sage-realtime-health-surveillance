"""Tests for streaming feature store."""
import time

import numpy as np
import pytest
import torch


def test_empty_history():
    from src.streaming.feature_store import StreamingFeatureStore

    store = StreamingFeatureStore.__new__(StreamingFeatureStore)
    store.config = type("C", (), {"window_sizes": [7, 14, 30, 90, 365]})()
    features = store._compute_features([], 1.0, time.time())
    assert features.shape == (34,), f"feature vector should be length 34, got {features.shape}"
    assert np.all(features == 0), "empty history should yield zero features"


def test_feature_dimensions():
    from src.streaming.feature_store import StreamingFeatureStore

    assert len(StreamingFeatureStore.FEATURE_NAMES) == 34, (
        f"expected 34 named features, got {len(StreamingFeatureStore.FEATURE_NAMES)}"
    )


def test_zscore_computation():
    values = np.random.randn(100)
    mean, std = values.mean(), values.std()
    zscore = (values[-1] - mean) / (std + 1e-8)
    assert abs(zscore) < 5, f"z-score should be in a reasonable range, got {zscore}"


def test_focal_loss():
    from src.models.anomaly_ensemble import FocalLoss

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.randn(10)
    targets = torch.zeros(10)
    targets[:1] = 1.0
    loss = criterion(logits, targets)
    assert loss.item() >= 0, f"focal loss should be non-negative, got {loss.item()}"


def test_neural_detector_shape():
    from src.models.anomaly_ensemble import NeuralAnomalyDetector

    model = NeuralAnomalyDetector(n_features=34, hidden_dim=64, n_layers=3)
    x = torch.randn(4, 34)
    out = model(x)
    assert out.shape == (4,), f"detector should output one score per row, got {tuple(out.shape)}"


@pytest.mark.parametrize(
    "score,expected_name",
    [
        (0.96, "CRITICAL"),
        (0.86, "HIGH"),
        (0.71, "WARNING"),
    ],
)
def test_alert_severity(score, expected_name):
    from src.streaming.alert_engine import AlertEngine, AlertSeverity

    engine = AlertEngine()
    severity = engine._classify_severity(score)
    expected = getattr(AlertSeverity, expected_name)
    assert severity == expected, (
        f"score {score} should map to {expected_name}, got {severity}"
    )
