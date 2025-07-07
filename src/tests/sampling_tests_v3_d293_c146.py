"""Tests for sampling tests v3 d293."""
import pytest
import torch
import numpy as np


class TestSamplingV3D293:
    def test_init(self):
        config = {"domain": "sampling", "v": 3, "d": 293}
        assert config["v"] == 3

    def test_forward(self):
        x = torch.randn(12, 24)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        items = [torch.randn(10) for _ in range(9)]
        assert len(items) == 9

    def test_loss(self):
        pred = torch.randn(8)
        target = torch.randn(8)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0
