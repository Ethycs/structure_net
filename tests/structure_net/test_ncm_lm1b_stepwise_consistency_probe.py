from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.structure_net import ncm_lm1b_native_finetuning_pilot as run


class Generator(nn.Module):
    levels = (1, 2)

    def _sample_level(self, logits, temperature, top_k, top_p):
        return logits.argmax(-1)


class Concepts:
    def __init__(self, failure=False):
        self.generator = Generator()
        self.failure = failure

    def generate_codes(self, features, mask):
        result = []
        for i, width in enumerate(self.generator.levels):
            logits = torch.full((len(features), width, 8), float(i))
            result.append(self.generator._sample_level(logits, 0., 0, 1.))
            if self.failure:
                raise RuntimeError("controlled generation failure")
        return torch.cat(result, 1)


def test_forced_train_history_keeps_all_logits_and_restores_original_sampler():
    model = SimpleNamespace(concepts=Concepts())
    codes = torch.tensor([[3, 4, 5]], dtype=torch.long)
    logits, forced = run.stepwise_logits(model, torch.ones(1, 2, 2), None, codes)
    assert logits.shape == (1, 3, 8) and torch.equal(forced, codes)
    assert "_sample_level" not in model.concepts.generator.__dict__
    assert torch.equal(model.concepts.generator._sample_level(logits, 0., 0, 1.), torch.zeros(1, 3, dtype=torch.long))


def test_sampler_override_is_restored_even_if_generation_raises():
    model = SimpleNamespace(concepts=Concepts(failure=True))
    with pytest.raises(RuntimeError, match="controlled generation"):
        run.stepwise_logits(model, torch.ones(1, 2, 2), None, torch.tensor([[3, 4, 5]]))
    assert "_sample_level" not in model.concepts.generator.__dict__
