from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.structure_net.ncm_lm1b_source_attention import FrozenMemoryConceptHead


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_level0 = self.use_post_upsample_conv = self.shared_output_head = True
        self.use_ctx_encoder = self.use_flex_attention = False
        self.prefix_len = 1
        self.config = SimpleNamespace(n_embd=4, n_head=2)
        self.transformer = nn.ModuleDict({"blocks": nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])})

    def forward(self, hidden):
        for block in self.transformer.blocks:
            hidden = block(hidden)
        return hidden


def setup():
    torch.manual_seed(42)
    g, codec = Generator(), nn.Linear(4, 4)
    x = torch.randn(2, 5, 4)
    expected = g(x).detach()
    head = FrozenMemoryConceptHead(g, codec, source_width=6, layers=(1, 2), seed=7, dropout=0.)
    source = torch.randn(2, 7, 6, requires_grad=True)
    mask = torch.tensor([[True] * 4 + [False] * 3, [True] * 7])
    return head, x, expected, source, mask


def test_zero_output_initialization_preserves_native_backbone_and_clears_context():
    head, x, expected, source, mask = setup()
    with head.source_context(source, mask):
        assert torch.equal(head.generator(x), expected)
        with pytest.raises(ValueError):
            with head.source_context(source, mask):
                pass
    assert head._bound is None
    with pytest.raises(RuntimeError):
        with head.source_context(source, mask):
            raise RuntimeError("injected")
    assert head._bound is None


def test_only_new_attention_trains_and_padded_memory_cannot_affect_output():
    head, x, expected, source, mask = setup()
    frozen = {k: v.detach().clone() for k, v in head.generator.state_dict().items()}
    head.train()
    assert not head.generator.training and not head.codec.training
    optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with head.source_context(source, mask):
            loss = (head.generator(x) - torch.randn_like(expected)).square().mean()
        loss.backward()
        optimizer.step()
    assert head.source_memory.kv.weight.grad.abs().sum() > 0
    assert all(a.q.weight.grad.abs().sum() > 0 for a in head.adapters.values())
    assert source.grad is None
    assert all(p.grad is None for p in head.generator.parameters())
    assert all(torch.equal(v, head.generator.state_dict()[k]) for k, v in frozen.items())
    head.eval()
    with head.source_context(source, mask):
        normal = head.generator(x)
    changed = source.detach().masked_fill(~mask[..., None], float('nan'))
    with head.source_context(changed, mask):
        assert torch.equal(head.generator(x), normal)
    assert torch.equal(normal[:, :1], expected[:, :1])


def test_mask_integrity_and_adapter_only_reload():
    head, x, _, source, mask = setup()
    bad = mask.clone(); bad[0, 0] = False
    with pytest.raises(ValueError):
        with head.source_context(source, bad):
            pass
    head2, _, _, _, _ = setup()
    saved = head.adapter_state()
    assert set(saved) == {"source_memory", "adapters"}
    head2.load_adapter_state(saved)
    with head.source_context(source, mask):
        a = head.generator(x)
    with head2.source_context(source, mask):
        b = head2.generator(x)
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        head.load_adapter_state({"generator": {}})
