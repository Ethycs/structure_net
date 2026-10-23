"""VQ cutoff objectives train the encoder/projections while books use EMA."""
from types import SimpleNamespace

import torch
from torch import nn

from experiments.structure_net.ncm_codec_training_v2 import ema_update, normalize_ema, quantize_cutoff


class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.levels = [1, 2]
        self.embeddings = nn.ModuleList([nn.Embedding(3, 2), nn.Embedding(3, 2)])
        self.projection = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(2))
            for e in self.embeddings:
                e.weight.copy_(torch.tensor([[0., 0.], [1., 0.], [0., 1.]]))
                e.requires_grad_(False)
        self.epsilon, self.commitment_cost, self.learned_l1_sampling = 1e-5, .25, True
        for i in range(2):
            self.register_buffer(f"ema_cluster_size_{i}", torch.tensor([1., 2., 3.]))
            self.register_buffer(f"ema_w_{i}", self.embeddings[i].weight.clone())

    def _distances(self, x, e):
        return (x[:, None]-e[None]).square().sum(-1)

    def _down_for_quant(self, x, i, width):
        return x.mean(1, keepdim=True)

    def _up_to_full(self, x, i, length):
        return x.expand(-1, length, -1)

    def _apply_quant_resi(self, x, i):
        return self.projection(x)


def test_partial_reconstruction_is_real_and_gradient_routes_match_paper():
    q = Quantizer()
    vae = SimpleNamespace(vq=q)
    z = torch.tensor([[[1.1, .1], [.1, 1.1]]], requires_grad=True)
    partial, partial_loss, details = quantize_cutoff(vae, z, 1)
    full, full_loss, _ = quantize_cutoff(vae, z, 2)
    assert not torch.equal(partial, full)
    assert details["cutoff"] == 1
    assert not torch.equal(partial_loss, full_loss)
    (partial.square().mean()+partial_loss).backward()
    assert z.grad is not None and z.grad.norm() > 0
    assert q.projection.weight.grad is not None and q.projection.weight.grad.norm() > 0
    assert all(e.weight.grad is None for e in q.embeddings)


def test_ema_reference_normalization_preserves_vectors_then_updates_without_grad():
    q = Quantizer()
    before = [e.weight.clone() for e in q.embeddings]
    normalize_ema(q, 32)
    assert all(torch.equal(e.weight, old) for e, old in zip(q.embeddings, before))
    assert torch.isclose(q.ema_cluster_size_1.sum(), torch.tensor(64.))
    ema_update(q, 0, torch.tensor([[[2., 1.]]]), torch.tensor([[1]]), .9)
    assert not torch.equal(q.embeddings[0].weight, before[0])
    assert torch.isfinite(q.embeddings[0].weight).all()
    assert q.embeddings[0].weight.grad is None
    assert torch.equal(q.embeddings[1].weight, before[1])


def test_straight_through_keeps_exact_lookup_values_despite_scale_difference():
    q = Quantizer()
    with torch.no_grad():
        q.embeddings[0].weight.fill_(.1234567)
    z = torch.full((1, 2, 2), 512., requires_grad=True)
    actual, _, _ = quantize_cutoff(SimpleNamespace(vq=q), z, 1)
    assert torch.equal(actual, torch.full_like(actual, q.embeddings[0].weight[0, 0]))
    actual.sum().backward()
    assert torch.equal(z.grad, torch.ones_like(z))
