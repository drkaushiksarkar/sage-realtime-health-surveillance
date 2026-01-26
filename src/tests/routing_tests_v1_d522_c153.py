"""Tests for routing tests v1 d522."""
import pytest
import torch
import numpy as np


class TestRoutingV1D522:
    def test_init(self):
        config = {"domain": "routing", "v": 1, "d": 522}
        assert config["v"] == 1

    def test_forward(self):
        x = torch.randn(4, 8)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        items = [torch.randn(10) for _ in range(3)]
        assert len(items) == 3

    def test_loss(self):
        pred = torch.randn(8)
        target = torch.randn(8)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0
