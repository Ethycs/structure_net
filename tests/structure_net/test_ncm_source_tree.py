import copy

import pytest
import torch

from experiments.structure_net import ncm_source_tree as tree
from tests.structure_net.test_ncm_source_coverage import toy, forward


def test_weighted_root_and_padding_preserve_original_leaves():
    k = torch.arange(18, dtype=torch.float32).reshape(2, 1, 9, 1)
    valid = torch.tensor([[True] * 9, [True] * 3 + [False] * 6])[:, None, None, :]
    k[1, :, 3:] = float('nan')
    original = (k, k.clone(), valid)
    memory = tree.build_tree(original, tree.Config(2, 4))
    assert memory[0] is k and memory[2] is valid
    assert [m[0].shape[2] for m in memory.ancestors] == [5, 3, 2, 1]
    assert torch.equal(memory.ancestors[-1][0].flatten(), torch.tensor([4., 10.]))
    assert all(torch.isfinite(m[0]).all() and torch.isfinite(m[1]).all() for m in memory.ancestors)
    assert torch.equal(memory.ancestors[0][2][1].flatten(), torch.tensor([True, True, False, False, False]))


def test_install_preserves_eval_outputs_parameters_rng_and_reload():
    model = toy()
    x, source = torch.randn(2, 4, 8), torch.randn(2, 17, 8)
    baseline = forward(model, x, source)
    old = [(p, p.detach().clone()) for p in model.parameters()]
    rng = torch.get_rng_state()
    meta = tree.install(model)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(baseline, forward(model, x, source))
    assert meta['source_blocks'] == list(range(12)) and len(model.concepts._handles) == 13
    assert all(torch.equal(p, saved) and any(p is q for q in model.parameters()) for p, saved in old)
    restored = toy()
    tree.install(restored)
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    assert torch.equal(forward(restored, x, source), baseline)


def test_all_tree_branches_learn_then_level_routing_receives_gradients():
    model = toy()
    tree.install(model)
    x = torch.randn(2, 4, 8)
    source = torch.randn(2, 17, 8, requires_grad=True)
    forward(model, x, source).square().sum().backward()
    for adapter in model.concepts.adapters.values():
        assert torch.isfinite(adapter.tree_out.grad).all() and adapter.tree_out.grad.abs().sum() > 0
        assert adapter.level_weights.grad.count_nonzero() == 0
        with torch.no_grad():
            adapter.tree_out.copy_(torch.eye(8) * .01)
    assert source.grad is None
    model.zero_grad(set_to_none=True)
    forward(model, x, source).square().sum().backward()
    for adapter in model.concepts.adapters.values():
        assert torch.isfinite(adapter.level_weights.grad).all() and adapter.level_weights.grad.abs().sum() > 0
    assert source.grad is None


def test_invalid_geometry_and_repeated_install_fail():
    with pytest.raises(ValueError):
        tree.Config(1, 5)
    model = toy()
    with pytest.raises(ValueError):
        tree.install(model, tree.Config(2, 2))
    assert model.concepts.layers == (8, 10, 11)
    tree.install(model)
    with pytest.raises(ValueError):
        tree.install(model)
    for handle in model.concepts._handles:
        handle.remove()
    assert not model.concepts.source_memory._forward_hooks
    assert not any(b._forward_hooks for b in model.concepts.generator.transformer.blocks)
