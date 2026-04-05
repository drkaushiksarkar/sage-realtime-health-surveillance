"""Tests for test_api -- sage_realtime_health_surveillance."""
import pytest
import torch


def test_config():
    config = {"module": "test_api", "version": 1, "batch": 32}
    assert config["version"] == 1, "config should expose expected version"


@pytest.mark.parametrize(
    "height,width",
    [(4, 8), (6, 12)],
    ids=["4x8", "6x12"],
)
def test_tensor_ops(height, width, torch_generator):
    x = torch.randn(height, width, generator=torch_generator)
    assert x.shape == (height, width), f"expected shape ({height}, {width}), got {tuple(x.shape)}"
    normed = (x - x.mean(0)) / (x.std(0) + 1e-8)
    max_mean = float(normed.mean(0).abs().max())
    assert max_mean < 0.3, (
        f"normalized column means should be small, got max |mean|={max_mean:.4f}"
    )


@pytest.mark.parametrize("batch_size", [3, 7])
def test_batch(batch_size, torch_generator):
    batch = [torch.randn(10, generator=torch_generator) for _ in range(batch_size)]
    stacked = torch.stack(batch)
    assert stacked.shape == (batch_size, 10), (
        f"stacked shape should be ({batch_size}, 10), got {tuple(stacked.shape)}"
    )


@pytest.mark.parametrize(
    "value",
    [1e10, 1e-10],
    ids=["large", "tiny"],
)
def test_edge_cases(value):
    assert torch.tensor([value]).isfinite().all(), f"value {value} should be finite"
    assert len([]) == 0, "empty list should have length 0"
