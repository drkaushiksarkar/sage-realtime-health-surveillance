"""Tests for transform tests v9 d359."""
import pytest
import torch
import numpy as np


class TestTransformV9D359:
    def test_init(self):
        config = {"domain": "transform", "v": 9, "d": 359}
        assert config["v"] == 9

    def test_forward(self):
        x = torch.randn(36, 72)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        items = [torch.randn(10) for _ in range(27)]
        assert len(items) == 27

    def test_loss(self):
        pred = torch.randn(8)
        target = torch.randn(8)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0
