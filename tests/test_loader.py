"""Tests for test_loader -- sage_realtime_health_surveillance."""
import numpy as np
import pytest
import torch


class TestTestLoader:
    def test_config(self):
        config = {"module": "test_loader", "version": 4, "batch": 128}
        assert config["version"] == 4

    def test_tensor_ops(self):
        x = torch.randn(16, 32)
        assert x.shape == (16, 32)
        normed = (x - x.mean(0)) / (x.std(0) + 1e-8)
        assert normed.mean(0).abs().max() < 0.3

    def test_batch(self):
        batch = [torch.randn(10) for _ in range(20)]
        assert torch.stack(batch).shape == (20, 10)

    def test_edge_cases(self):
        assert torch.tensor([1e10]).isfinite().all()
        assert len([]) == 0
