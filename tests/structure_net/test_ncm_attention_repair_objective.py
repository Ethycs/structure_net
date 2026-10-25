from contextlib import contextmanager
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from experiments.structure_net import ncm_attention_repair_objective as objective


class VQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.levels = (1, 2)
        self.embeddings = nn.ModuleList([nn.Embedding(3, 4), nn.Embedding(3, 4)])

    def _up_to_full(self, x, level, size):
        return x.expand(-1, size, -1)

    def _apply_quant_resi(self, x, level):
        return x

    def idx_to_z_q(self, codes):
        return self.embeddings[0](codes[:, :1]).expand(-1, 2, -1) + self.embeddings[1](codes[:, 1:])


class Renderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.codec = nn.Module()
        self.codec.vq = VQ()
        self.codec.decoder = nn.Linear(4, 6)
        self.eval().requires_grad_(False)

    @contextmanager
    def word_context(self, features, mask):
        assert not features.requires_grad
        yield


def test_word_forward_is_hard_and_gradients_only_reach_corrector_scores():
    torch.manual_seed(7)
    renderer = Renderer()
    scores = [torch.randn(1, n, 3, requires_grad=True) for n in (1, 2)]
    targets = torch.tensor([[0, 1, 2]])
    words = torch.tensor([[2, 3]])
    features = torch.randn(1, 5, 8, requires_grad=True)
    mask = torch.ones(1, 5, dtype=torch.bool)
    observed = []
    hook = renderer.codec.decoder.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach().clone()))
    value, metrics, codes = objective.loss(renderer, scores, targets, words, features, mask)
    hook.remove()
    hard = renderer.codec.vq.idx_to_z_q(codes)
    assert torch.equal(observed[0], hard)
    expected = F.cross_entropy(renderer.codec.decoder(hard).flatten(0, 1), words.flatten())
    assert metrics['hard_forward_word_CE'] == float(expected)
    value.backward()
    assert all(s.grad is not None and s.grad.abs().sum() > 0 for s in scores)
    assert features.grad is None and all(p.grad is None for p in renderer.parameters())
