import pytest
import torch

from experiments.structure_net.ncm_isolated_quality_train import Config, learning_rate


def test_fixed_warmup_cosine_schedule_boundaries():
    c = Config()
    assert learning_rate(1) == c.max_lr / 32
    assert learning_rate(32) == c.max_lr
    assert 0 < learning_rate(33) < c.max_lr
    assert learning_rate(512) == 0
    assert all(learning_rate(i + 1) <= learning_rate(i) for i in range(32, 512))
    with pytest.raises(ValueError):
        learning_rate(513)


def test_private_mapped_torch_cache_has_unchanged_consumed_values(tmp_path):
    path = tmp_path / "example.pt"
    expected = torch.arange(40).reshape(1, 2, 5, 4).transpose(1, 2)
    torch.save({"prefix": expected, "targets": torch.tensor([1, 2, -100])}, path)
    actual = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert torch.equal(actual["prefix"], expected)
    _ = torch.cat([actual["prefix"]], 0).clone()
    assert torch.equal(actual["prefix"], expected)
