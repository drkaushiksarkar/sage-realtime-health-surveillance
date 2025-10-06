"""Tests for routing tests v8 d402."""
import pytest
import torch
import numpy as np


class TestRoutingV8D402:
    def test_init(self):
        config = {"domain": "routing", "v": 8, "d": 402}
        assert config["v"] == 8

    def test_forward(self):
        x = torch.randn(32, 64)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        items = [torch.randn(10) for _ in range(24)]
        assert len(items) == 24

    def test_loss(self):
        pred = torch.randn(8)
        target = torch.randn(8)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0
