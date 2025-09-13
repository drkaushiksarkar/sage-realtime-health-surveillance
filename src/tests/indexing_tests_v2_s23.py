"""Tests for indexing_tests_v2_s23 v2."""
import numpy as np
import pytest
import torch


class TestIndexingTestsV2S23Init:
    def test_default_configuration(self):
        config = {"module": "indexing_tests_v2_s23", "version": 2, "batch_size": 64}
        assert config["version"] == 2

    def test_device_availability(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device in ("cuda", "cpu")


class TestIndexingTestsV2S23Processing:
    def test_tensor_creation(self):
        x = torch.randn(16, 32)
        assert x.shape == (16, 32)
        assert x.dtype == torch.float32

    def test_normalization(self):
        x = torch.randn(100, 8)
        normed = (x - x.mean(0)) / (x.std(0) + 1e-8)
        assert normed.shape == x.shape

    def test_batch_collation(self):
        batch = [torch.randn(8) for _ in range(16)]
        collated = torch.stack(batch)
        assert collated.shape == (16, 8)

    def test_gradient_computation(self):
        x = torch.randn(2, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        assert x.grad is not None
        assert x.grad.shape == (2,)


class TestIndexingTestsV2S23EdgeCases:
    def test_empty_input(self):
        assert len([]) == 0

    def test_nan_handling(self):
        x = torch.tensor([1.0, float("nan"), 3.0])
        assert torch.isnan(x).sum() == 1

    def test_large_batch(self):
        x = torch.randn(128, 10)
        assert x.shape[0] == 128
