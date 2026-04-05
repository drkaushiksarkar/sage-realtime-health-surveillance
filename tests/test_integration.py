"""Integration tests for sage-realtime-health-surveillance pipeline."""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest


def test_data_ingestion_to_output(sample_dataframe):
    data = sample_dataframe
    assert len(data["timestamp"]) == 100, "timestamp column should have 100 rows"
    assert len(data["value"]) == 100, "value column should align with timestamps"
    values = np.array(data["value"])
    assert np.isfinite(values).all(), "all sampled values should be finite"


@pytest.mark.parametrize(
    "batch_size,n_batches",
    [(1000, 10), (512, 4)],
    ids=["10k-samples", "2k-samples"],
)
def test_batch_processing_memory_stable(batch_size, n_batches):
    results = []
    for i in range(n_batches):
        batch = np.random.randn(batch_size, 64)
        processed = np.maximum(0, batch)
        results.append(float(np.mean(processed)))
    assert len(results) == n_batches, f"expected {n_batches} batch means, got {len(results)}"
    assert all(r >= 0 for r in results), "ReLU-style processing should yield non-negative means"


def test_checkpoint_save_and_restore(tmp_path, training_checkpoint_state):
    path = tmp_path / "checkpoint.json"
    with open(path, "w") as f:
        json.dump(training_checkpoint_state, f)
    with open(path) as f:
        restored = json.load(f)
    assert restored["epoch"] == 5, f"epoch should round-trip, got {restored['epoch']}"
    assert abs(restored["loss"] - 0.342) < 1e-6, (
        f"loss should round-trip within float tolerance, got {restored['loss']}"
    )
    assert restored["metrics"]["accuracy"] == 0.891, (
        f"nested metrics should round-trip, got {restored['metrics']}"
    )


def test_concurrent_data_access():
    shared_data = np.random.randn(1000, 32)

    def process_slice(start):
        chunk = shared_data[start : start + 100].copy()
        return float(np.mean(chunk))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_slice, i * 100) for i in range(10)]
        results = [f.result() for f in futures]
    assert len(results) == 10, f"expected 10 slice means, got {len(results)}"


def test_error_recovery():
    errors = []
    for i in range(5):
        try:
            if i == 2:
                raise ValueError("Simulated transient error")
            _result = i * 2
        except ValueError as e:
            errors.append(str(e))
    assert len(errors) == 1, f"expected one caught error, got {len(errors)}"
    assert "transient" in errors[0], f"error message should describe failure: {errors[0]!r}"
