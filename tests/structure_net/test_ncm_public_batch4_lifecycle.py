import pytest
import torch

from experiments.structure_net import ncm_public_batch4_lifecycle as lifecycle


def test_four_cases_include_both_independent_length_extremes():
    rows = [dict(id=str(i), source_positions=a, native_prompt_positions=b) for i, (a, b) in enumerate(
        [(192, 300), (796, 900), (4124, 2210), (2095, 2275), (514, 700), (1585, 1700)])]
    assert lifecycle.select(rows) == ['0', '1', '3', '2']
    with pytest.raises(ValueError): lifecycle.select(rows[:3])


def test_checkpoint_preserves_updated_weights_moments_and_separate_rngs(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    model(torch.ones(1, 2)).sum().backward(); optimizer.step()
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    sampler = torch.Generator().manual_seed(1)
    noise = torch.Generator().manual_seed(2)
    torch.rand(5, generator=sampler); torch.rand(7, generator=noise)
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    path = tmp_path/'checkpoint.pt'
    lifecycle.save(path, model, optimizer, scaler, sampler, noise, 1, {'fingerprint': 'proof'})
    with torch.no_grad():
        model.weight.zero_()
    saved = torch.load(path, weights_only=True)
    assert all(torch.equal(saved['model'][k], v) for k, v in expected.items())
    assert saved['optimizer']['state'] and saved['step'] == 1
    assert torch.equal(saved['sampler_rng'], sampler.get_state())
    assert torch.equal(saved['noise_rng'], noise.get_state())
    with pytest.raises(ValueError):
        lifecycle.save(path, model, optimizer, scaler, sampler, noise, 2, {'fingerprint': 'proof'})
