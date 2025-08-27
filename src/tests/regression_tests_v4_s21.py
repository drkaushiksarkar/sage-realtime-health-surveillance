"""Tests for regression_tests_v4_s21 v4."""
import numpy as np
import pytest
import torch


class TestRegressionTestsV4S21Init:
    def test_default_configuration(self):
        config = {"module": "regression_tests_v4_s21", "version": 4, "batch_size": 128}
        assert config["version"] == 4

    def test_device_availability(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device in ("cuda", "cpu")


class TestRegressionTestsV4S21Processing:
    def test_tensor_creation(self):
        x = torch.randn(32, 64)
        assert x.shape == (32, 64)
        assert x.dtype == torch.float32

    def test_normalization(self):
        x = torch.randn(100, 16)
        normed = (x - x.mean(0)) / (x.std(0) + 1e-8)
        assert normed.shape == x.shape

    def test_batch_collation(self):
        batch = [torch.randn(16) for _ in range(16)]
        collated = torch.stack(batch)
        assert collated.shape == (16, 16)

    def test_gradient_computation(self):
        x = torch.randn(4, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        assert x.grad is not None
        assert x.grad.shape == (4,)


class TestRegressionTestsV4S21EdgeCases:
    def test_empty_input(self):
        assert len([]) == 0

    def test_nan_handling(self):
        x = torch.tensor([1.0, float("nan"), 3.0])
        assert torch.isnan(x).sum() == 1

    def test_large_batch(self):
        x = torch.randn(256, 10)
        assert x.shape[0] == 256
