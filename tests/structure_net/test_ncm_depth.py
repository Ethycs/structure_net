import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import ncm_depth as depth


def fixture():
    vendor = Path('data/experiments/ncm_llama/vendor/ncm-20261021')
    sys.path.insert(0, str(vendor.resolve()))
    from models.common import Block
    cfg = SimpleNamespace(n_layer=12, n_embd=8, n_head=2, bias=True, dropout=0.)
    blocks = torch.nn.ModuleList([Block(cfg) for _ in range(12)])
    calls = []
    for i in (8, 10, 11):
        blocks[i].register_forward_hook(lambda m, a, out, i=i: calls.append(i))
    generator = SimpleNamespace(transformer=SimpleNamespace(blocks=blocks), config=cfg)
    return SimpleNamespace(concepts=SimpleNamespace(generator=generator, _bound=None,
                                                    layers=(8, 10, 11))), calls


def forward(model, x):
    for block in model.concepts.generator.transformer.blocks:
        x = block(x)
    return x


def test_real_blocks_preserve_outputs_hooks_and_receive_gradients():
    torch.manual_seed(42)
    model, calls = fixture()
    x = torch.randn(2, 3, 8)
    expected = forward(model, x).detach()
    assert calls == [8, 10, 11]
    calls.clear()
    info = depth.expand(model, 24)
    assert info['source_block_positions'] == [16, 20, 22]
    actual = forward(model, x)
    assert torch.equal(actual, expected)
    assert calls == [8, 10, 11]
    actual.square().sum().backward()
    for i in info['inserted']:
        block = model.concepts.generator.transformer.blocks[i]
        assert block.attn.c_proj.weight.grad.abs().sum() > 0
        assert block.mlp.c_proj.weight.grad.abs().sum() > 0
        assert block.attn.c_attn.weight.data_ptr() != model.concepts.generator.transformer.blocks[i-1].attn.c_attn.weight.data_ptr()
    reloaded, _ = fixture()
    depth.expand(reloaded, 24)
    reloaded.concepts.generator.transformer.blocks.load_state_dict(
        model.concepts.generator.transformer.blocks.state_dict(), strict=True)
    assert torch.equal(forward(reloaded, x), actual)


def test_rejects_reexpansion_or_active_source_context():
    model, _ = fixture()
    depth.expand(model, 24)
    with pytest.raises(ValueError):
        depth.expand(model, 24)
    model, _ = fixture()
    model.concepts._bound = object()
    with pytest.raises(ValueError):
        depth.expand(model, 24)
