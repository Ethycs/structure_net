"""CPU contracts for isolated prefix replacement and the fixed bridge gate."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import ncm_xsum_prefix_bridge as bridge
from experiments.neural_architecture_lab import ncm_xsum_prefix_bridge_results as results


def entries():
    return [{'id': str(i), 'source_sha256': str(i), 'source_component': str(i),
             'source_positions': 10+i} for i in range(10)]


def test_fingerprint_binds_configuration_and_input_identity():
    proof = {'config': {'seed': 5827}, 'source_hashes': {}, 'input_hashes': {}}
    proof['fingerprint'] = bridge.fingerprint(proof)
    bridge.verify_proof(proof)
    proof['config']['seed'] += 1
    with pytest.raises(ValueError): bridge.verify_proof(proof)


def test_selection_is_order_independent_source_disjoint_and_unclipped():
    rows = entries(); rows[-1]['source_positions'] = 513
    selected = bridge.select(rows, 3, 2)
    assert selected == bridge.select(list(reversed(rows)), 3, 2)
    assert len({r['id'] for group in selected.values() for r in group}) == 5
    assert all(r['source_positions'] <= 512 for group in selected.values() for r in group)
    with pytest.raises(FrozenInstanceError): bridge.CONFIG.fit_cases = 2


@pytest.mark.parametrize('fault', ['id', 'source_sha256', 'source_component', 'insufficient'])
def test_invalid_population_rejected(fault):
    rows = entries()
    if fault == 'insufficient': rows = rows[:2]
    else: rows[-1][fault] = rows[0][fault]
    with pytest.raises(ValueError): bridge.select(rows, 8, 2)


def test_bridge_initial_mean_and_frozen_projection_gradient_route():
    stats = {'input_mean': torch.ones(3), 'input_std': torch.ones(3)*2,
             'embedding_mean': torch.tensor([.2, .3])}
    model = bridge.Bridge(stats, hidden=4)
    projection = torch.nn.Linear(2, 5).requires_grad_(False)
    x = torch.randn(6, 3); target = torch.randn(6, 5)
    assert torch.equal(model(x), stats['embedding_mean'].expand(6, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    for _ in range(2):
        opt.zero_grad(); (projection(model(x))-target).square().mean().backward(); opt.step()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters())
    assert all(p.grad is None for p in projection.parameters())
    assert all(not v.requires_grad for v in model.buffers())


def test_hook_replaces_only_dummy_prefix_and_is_removed_on_failure():
    embedding = torch.nn.Embedding(8, 3)
    dummy = torch.zeros(1, 4, dtype=torch.long); replacement = torch.randn(1, 4, 3)
    with bridge.replace_prefix(embedding, replacement, dummy):
        assert torch.equal(embedding(dummy), replacement)
    with pytest.raises(ValueError):
        with bridge.replace_prefix(embedding, replacement, dummy): embedding(dummy+1)
    assert not embedding._forward_hooks
    with pytest.raises(RuntimeError):
        with bridge.replace_prefix(embedding, replacement, dummy): raise RuntimeError('interrupted')
    assert not embedding._forward_hooks


def test_hook_rejects_no_call_and_multiple_calls():
    embedding = torch.nn.Embedding(2, 3); dummy = torch.zeros(1, 4, dtype=torch.long)
    for count in (0, 2):
        with pytest.raises(ValueError):
            with bridge.replace_prefix(embedding, torch.zeros(1, 4, 3), dummy):
                for _ in range(count): embedding(dummy)
        assert not embedding._forward_hooks


def test_bridged_generator_receives_only_dummy_content_ids_and_exact_padding():
    embedding = torch.nn.Embedding(50304, 3)
    vae = SimpleNamespace(encoder=SimpleNamespace(wte=embedding))
    seen = []
    class Generator:
        def generate(self, idx, ids, **kwargs):
            seen.append(ids.clone())
            return embedding(ids)
    learned = torch.randn(4, 3)
    result = bridge.codes_bridged(vae, Generator(), learned)
    assert torch.equal(seen[0][0, :4], torch.zeros(4, dtype=torch.long))
    assert torch.all(seen[0][0, 4:] == 50257)
    assert torch.equal(result[0, :4], learned)
    assert torch.equal(result[0, 4:], embedding.weight[50257].expand(508, 3))
    assert not embedding._forward_hooks


def record():
    prediction = {'codes': [0]*200, 'words': [5, 6, 50256]+[50257]*45}
    return {'id': 'a', 'source_positions': 2, 'projected_elements': 1536,
        'bridge_squared_error': 1., 'mean_squared_error': 4.,
        **{arm: deepcopy(prediction) for arm in ('native', 'bridge', 'mean')}}


def test_raw_replay_counts_and_content_gate_prevent_padding_success():
    row = record(); good = results.replay_counts([row])
    assert good['preservation_gate_passed']
    assert good['bridge_MSE_relative_to_mean'] == .25
    row['bridge']['words'][0] = 7
    failed = results.replay_counts([row])
    assert failed['scores']['bridge']['word_agreement'] == pytest.approx(47/48)
    assert failed['scores']['bridge']['content_agreement'] == .5
    assert not failed['preservation_gate_passed']


@pytest.mark.parametrize('fault', ['boolean', 'nan', 'no_mean_error', 'short_codes', 'empty_native'])
def test_bad_or_empty_outputs_cannot_pass(fault):
    row = record()
    if fault == 'boolean': row['bridge']['codes'][0] = True
    if fault == 'nan': row['bridge_squared_error'] = float('nan')
    if fault == 'no_mean_error': row['mean_squared_error'] = 0.
    if fault == 'short_codes': row['bridge']['codes'].pop()
    if fault == 'empty_native':
        row['native']['words'][0] = 50256
        assert not results.replay_counts([row])['preservation_gate_passed']
    else:
        with pytest.raises(ValueError): results.replay_counts([row])


def test_exact_nested_optimizer_state_comparison_and_prefix_limit():
    a = {'state': {0: {'step': torch.tensor(2.), 'exp_avg': torch.tensor([.1])}}, 'groups': [1, 2]}
    b = deepcopy(a); assert bridge.equal_state(a, b)
    b['state'][0]['exp_avg'][0] += .01; assert not bridge.equal_state(a, b)
    assert bridge.prefix_ids([3, 4], 'cpu').shape == (1, 512)
    assert bridge.prefix_ids([3, 4], 'cpu')[0, 2] == 50257
    with pytest.raises(ValueError): bridge.prefix_ids([1]*513, 'cpu')
