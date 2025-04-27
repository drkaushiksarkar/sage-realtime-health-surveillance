"""Tests for streaming feature store."""
import numpy as np
import pytest
import time


class TestFeatureComputation:
    def test_empty_history(self):
        from src.streaming.feature_store import StreamingFeatureStore
        store = StreamingFeatureStore.__new__(StreamingFeatureStore)
        store.config = type("C", (), {"window_sizes": [7,14,30,90,365]})()
        features = store._compute_features([], 1.0, time.time())
        assert features.shape == (34,)
        assert np.all(features == 0)

    def test_feature_dimensions(self):
        from src.streaming.feature_store import StreamingFeatureStore
        assert len(StreamingFeatureStore.FEATURE_NAMES) == 34

    def test_zscore_computation(self):
        values = np.random.randn(100)
        mean, std = values.mean(), values.std()
        zscore = (values[-1] - mean) / (std + 1e-8)
        assert abs(zscore) < 5  # reasonable range


class TestAnomalyEnsemble:
    def test_focal_loss(self):
        import torch
        from src.models.anomaly_ensemble import FocalLoss
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(10)
        targets = torch.zeros(10)
        targets[:1] = 1.0  # 10% positive
        loss = criterion(logits, targets)
        assert loss.item() >= 0

    def test_neural_detector_shape(self):
        import torch
        from src.models.anomaly_ensemble import NeuralAnomalyDetector
        model = NeuralAnomalyDetector(n_features=34, hidden_dim=64, n_layers=3)
        x = torch.randn(4, 34)
        out = model(x)
        assert out.shape == (4,)

    def test_alert_severity(self):
        from src.streaming.alert_engine import AlertEngine, AlertSeverity
        engine = AlertEngine()
        assert engine._classify_severity(0.96) == AlertSeverity.CRITICAL
        assert engine._classify_severity(0.86) == AlertSeverity.HIGH
        assert engine._classify_severity(0.71) == AlertSeverity.WARNING
