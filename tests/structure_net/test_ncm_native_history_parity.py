"""Oracle sampler must preserve order and restore inference even on an error."""
from contextlib import nullcontext

import pytest
import torch

from experiments.structure_net.ncm_native_history_parity import forced_logits


class Generator:
    levels = (1, 2)

    def _sample_level(self, *args):
        raise AssertionError('original sampler should not run')

    def generate(self, *args, **kwargs):
        return torch.cat([self._sample_level(torch.zeros(1, width, 5), 0., 0, 1.) for width in self.levels], dim=1)


def test_sampler_preserves_oracle_and_restores_class_method_on_success_and_error():
    generator = Generator()
    head = type('Head', (), {'generator': generator, 'source_context': lambda *args: nullcontext()})()
    source, mask, codes = torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.bool), torch.tensor([[1, 2, 3]])
    original = generator._sample_level.__func__
    assert forced_logits(head, source, mask, codes).shape == (1, 3, 5)
    assert '_sample_level' not in generator.__dict__ and generator._sample_level.__func__ is original
    generator.generate = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('interrupted'))
    with pytest.raises(ValueError, match='interrupted'):
        forced_logits(head, source, mask, codes)
    assert '_sample_level' not in generator.__dict__ and generator._sample_level.__func__ is original
