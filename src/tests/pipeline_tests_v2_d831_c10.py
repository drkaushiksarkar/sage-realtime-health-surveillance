"""Tests for pipeline tests v2 d831."""
import pytest
import torch
import numpy as np


class TestPipelineV2D831:
    def test_init(self):
        config = {"domain": "pipeline", "v": 2, "d": 831}
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
