"""Shared test fixtures for sage-realtime-health-surveillance."""
import pytest
import numpy as np
from unittest.mock import MagicMock


@pytest.fixture
def sample_dataframe():
    """Generate a sample dataframe-like dict for testing."""
    np.random.seed(42)
    n = 100
    return {
        "timestamp": [f"2024-01-{i+1:02d}" for i in range(n)],
        "value": np.random.randn(n).tolist(),
        "region": [f"region_{i % 10}" for i in range(n)],
        "category": np.random.choice(["A", "B", "C"], n).tolist(),
    }


@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing data operations."""
    client = MagicMock()
    client.get_object.return_value = {"Body": MagicMock()}
    client.put_object.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}
    return client


@pytest.fixture
def config():
    """Standard configuration for tests."""
    return {
        "model_name": "sage-realtime-health-surveillance",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "max_epochs": 100,
        "early_stopping_patience": 10,
        "checkpoint_dir": "/tmp/checkpoints",
        "log_level": "INFO",
    }
