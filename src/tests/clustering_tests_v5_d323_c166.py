"""Tests for clustering tests v5 d323."""
import pytest
import torch
import numpy as np


class TestClusteringV5D323:
    def test_init(self):
        config = {"domain": "clustering", "v": 5, "d": 323}
        assert config["v"] == 5

    def test_forward(self):
        x = torch.randn(20, 40)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        items = [torch.randn(10) for _ in range(15)]
        assert len(items) == 15

    def test_loss(self):
        pred = torch.randn(8)
        target = torch.randn(8)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0
