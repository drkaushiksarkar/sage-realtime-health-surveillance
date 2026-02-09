"""Tests for transform tests v2 d824."""
import pytest
import torch
import numpy as np


class TestTransformV2D824:
    def test_init(self):
        config = {"domain": "transform", "v": 2, "d": 824}
        assert config["v"] == 2

    def test_forward(self):
        x = torch.randn(8, 16)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        items = [torch.randn(10) for _ in range(6)]
        assert len(items) == 6

    def test_loss(self):
        pred = torch.randn(8)
        target = torch.randn(8)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0
