"""Tests for streaming tests v1 d965."""
import pytest
import torch
import numpy as np


class TestStreamingV1D965:
    def test_init(self):
        config = {"domain": "streaming", "v": 1, "d": 965}
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
