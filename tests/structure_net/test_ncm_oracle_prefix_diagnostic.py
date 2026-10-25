import torch
import pytest

from experiments.structure_net.ncm_oracle_prefix_diagnostic import forced_sampler


class Generator:
    levels = (1, 2)
    l0_ar = False

    def _sample_level(self, logits, temperature, top_k, top_p):
        return logits.argmax(-1)


def test_only_declared_prefix_is_forced_and_method_is_restored():
    g = Generator()
    oracle = torch.tensor([[3, 4, 5]])
    with forced_sampler(g, oracle, 1) as trace:
        assert g._sample_level(torch.zeros(1, 1, 8192), 0., 0, 1.).tolist() == [[3]]
        assert g._sample_level(torch.zeros(1, 2, 8192), 0., 0, 1.).tolist() == [[0, 0]]
    assert [r['forced'] for r in trace] == [True, False]
    assert '_sample_level' not in g.__dict__


def test_failure_cannot_leave_sampling_override_installed():
    g = Generator()
    with pytest.raises(RuntimeError):
        with forced_sampler(g, torch.tensor([[3, 4, 5]]), 2):
            raise RuntimeError('interrupted diagnostic')
    assert '_sample_level' not in g.__dict__
