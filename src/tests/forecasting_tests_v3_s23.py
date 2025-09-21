"""Tests for forecasting_tests_v3_s23 v3."""
import numpy as np
import pytest
import torch


class TestForecastingTestsV3S23Init:
    def test_default_configuration(self):
        config = {"module": "forecasting_tests_v3_s23", "version": 3, "batch_size": 96}
        assert config["version"] == 3

    def test_device_availability(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device in ("cuda", "cpu")


class TestForecastingTestsV3S23Processing:
    def test_tensor_creation(self):
        x = torch.randn(24, 48)
        assert x.shape == (24, 48)
        assert x.dtype == torch.float32

    def test_normalization(self):
        x = torch.randn(100, 12)
        normed = (x - x.mean(0)) / (x.std(0) + 1e-8)
        assert normed.shape == x.shape

    def test_batch_collation(self):
        batch = [torch.randn(12) for _ in range(16)]
        collated = torch.stack(batch)
        assert collated.shape == (16, 12)

    def test_gradient_computation(self):
        x = torch.randn(3, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        assert x.grad is not None
        assert x.grad.shape == (3,)


class TestForecastingTestsV3S23EdgeCases:
    def test_empty_input(self):
        assert len([]) == 0

    def test_nan_handling(self):
        x = torch.tensor([1.0, float("nan"), 3.0])
        assert torch.isnan(x).sum() == 1

    def test_large_batch(self):
        x = torch.randn(192, 10)
        assert x.shape[0] == 192
