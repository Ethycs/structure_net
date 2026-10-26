import pytest
import torch

from experiments.structure_net import ncm_public_batch4_deterministic as recovery


def test_determinism_is_required_without_changing_batch_or_gradient_gates(monkeypatch):
    original = recovery.original.CONFIG
    for key in ('microbatch', 'effective_examples', 'updates', 'gradient_relative_tolerance',
                'gradient_absolute_tolerance', 'peak_limit_bytes', 'sampler_seed', 'noise_seed', 'dropout_seed'):
        assert getattr(recovery.CONFIG, key) == getattr(original, key)
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
              torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    with pytest.raises(ValueError): recovery.configure_runtime()
    try:
        monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        recovery.configure_runtime()
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = before[2:]
