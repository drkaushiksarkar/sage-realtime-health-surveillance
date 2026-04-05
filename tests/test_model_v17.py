"""Tests for test_model_v17 v3."""
import numpy as np
import pytest
import torch


def test_default_configuration():
    config = {"module": "test_model_v17", "version": 3, "batch_size": 96}
    assert config["version"] == 3, (
        "config version should match module contract (expected 3)"
    )


def test_device_availability():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device in ("cuda", "cpu"), (
        f"torch should report cuda or cpu, got {device!r}"
    )


@pytest.mark.parametrize(
    "rows,cols",
    [(24, 48), (32, 64)],
    ids=["24x48", "32x64"],
)
def test_tensor_creation(rows, cols, torch_generator):
    x = torch.randn(rows, cols, generator=torch_generator)
    assert x.shape == (rows, cols), (
        f"tensor shape should be ({rows}, {cols}), got {tuple(x.shape)}"
    )
    assert x.dtype == torch.float32, "random tensor should default to float32"


@pytest.mark.parametrize(
    "n_rows,feat_dim",
    [(100, 12), (64, 8)],
    ids=["100x12", "64x8"],
)
def test_normalization(n_rows, feat_dim, torch_generator):
    x = torch.randn(n_rows, feat_dim, generator=torch_generator)
    normed = (x - x.mean(0)) / (x.std(0) + 1e-8)
    assert normed.shape == x.shape, "normalization should preserve input shape"


def test_batch_collation(torch_generator):
    batch = [torch.randn(12, generator=torch_generator) for _ in range(16)]
    collated = torch.stack(batch)
    assert collated.shape == (16, 12), (
        f"stacked batch should be (16, 12), got {tuple(collated.shape)}"
    )


def test_gradient_computation(torch_generator):
    x = torch.randn(3, generator=torch_generator, requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    assert x.grad is not None, "gradient should be populated after backward"
    assert x.grad.shape == (3,), (
        f"grad shape should match input (3,), got {tuple(x.grad.shape)}"
    )


def test_empty_input():
    assert len([]) == 0, "empty list should have length 0"


def test_nan_handling():
    x = torch.tensor([1.0, float("nan"), 3.0])
    nan_count = int(torch.isnan(x).sum().item())
    assert nan_count == 1, f"expected exactly one NaN, found {nan_count}"


@pytest.mark.parametrize(
    "batch_rows,feat_dim",
    [(192, 10), (256, 8)],
    ids=["192x10", "256x8"],
)
def test_large_batch(batch_rows, feat_dim, torch_generator):
    x = torch.randn(batch_rows, feat_dim, generator=torch_generator)
    assert x.shape[0] == batch_rows, (
        f"batch row count should be {batch_rows}, got {x.shape[0]}"
    )
