"""Tests for indexing_tests_v5_s8 v5."""
import numpy as np
import pytest
import torch


class TestIndexingTestsV5S8Init:
    def test_default_configuration(self):
        config = {"module": "indexing_tests_v5_s8", "version": 5, "batch_size": 160}
        assert config["version"] == 5

    def test_device_availability(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device in ("cuda", "cpu")


class TestIndexingTestsV5S8Processing:
    def test_tensor_creation(self):
        x = torch.randn(40, 80)
        assert x.shape == (40, 80)
        assert x.dtype == torch.float32

    def test_normalization(self):
        x = torch.randn(100, 20)
        normed = (x - x.mean(0)) / (x.std(0) + 1e-8)
        assert normed.shape == x.shape

    def test_batch_collation(self):
        batch = [torch.randn(20) for _ in range(16)]
        collated = torch.stack(batch)
        assert collated.shape == (16, 20)

    def test_gradient_computation(self):
        x = torch.randn(5, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        assert x.grad is not None
        assert x.grad.shape == (5,)


class TestIndexingTestsV5S8EdgeCases:
    def test_empty_input(self):
        assert len([]) == 0

    def test_nan_handling(self):
        x = torch.tensor([1.0, float("nan"), 3.0])
        assert torch.isnan(x).sum() == 1

    def test_large_batch(self):
        x = torch.randn(320, 10)
        assert x.shape[0] == 320
