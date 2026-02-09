"""Tests for indexing tests v6 d824."""
import pytest
import torch
import numpy as np


class TestIndexingV6D824:
    def test_init(self):
        config = {"domain": "indexing", "v": 6, "d": 824}
        assert config["v"] == 6

    def test_forward(self):
        x = torch.randn(24, 48)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        items = [torch.randn(10) for _ in range(18)]
        assert len(items) == 18

    def test_loss(self):
        pred = torch.randn(8)
        target = torch.randn(8)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0
