from dataclasses import asdict

import pytest
import torch

from experiments.structure_net import ncm_public_training_engine as engine
from experiments.structure_net import ncm_cpu_adamw as offload


def test_noise_matches_serial_examples_and_preserves_global_rng():
    epoch = engine.curriculum.Epoch(1)
    refs = epoch.update(0)[:4]
    gold = torch.arange(20).reshape(4, 5)
    state = torch.get_rng_state().clone()
    actual, changes = engine.histories(gold, refs, [2, 3])
    rows = [engine.histories(gold[i:i+1], refs[i:i+1], [2, 3]) for i in range(4)]
    assert torch.equal(actual, torch.cat([r[0] for r in rows]))
    assert changes == [r[1][0] for r in rows]
    assert torch.equal(state, torch.get_rng_state())


def test_checkpoint_restores_real_optimizer_and_rng_before_next_step(tmp_path):
    torch.manual_seed(19)
    model = torch.nn.Linear(3, 2)
    opt = offload.CPUAdamW(model.parameters(), lr=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    def step():
        opt.zero_grad(set_to_none=True)
        model(torch.randn(4, 3)).square().mean().backward()
        opt.step()
    step()
    path = tmp_path/'step1.pt'
    engine.save(path, model, opt, scaler, step=1, fingerprint='fixed')
    step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    expected_rng = torch.get_rng_state().clone()
    saved = torch.load(path, weights_only=True)
    assert engine.restore(saved, model, opt, scaler, fingerprint='fixed') == 1
    step()
    assert all(torch.equal(expected[k], v) for k, v in model.state_dict().items())
    assert torch.equal(expected_rng, torch.get_rng_state())
    with pytest.raises(ValueError): engine.restore(saved, model, opt, scaler, fingerprint='different')
    with pytest.raises(ValueError): engine.save(path, model, opt, scaler, step=2, fingerprint='fixed')


def test_mixed_reads_preserve_exact_presentation_order():
    class Public:
        pairs = type('Pairs', (), {'entries': [{'id': 'p0'}, {'id': 'p1'}]})()
        def __getitems__(self, indices):
            return [{'id': 'p'+str(i), 'features': torch.ones(i+1, 3), 'oracle_codes': [i]*641,
                     'native_word_ids': [101]*128, 'source_sha256': 's'+str(i)} for i in indices]
    class Cache:
        def get(self, row):
            return torch.zeros(2, 3)
    anchors = [{'id': 'a0', 'split': 'TRAIN', 'source_sha256': 'a',
                'oracle_codes': [7]*641, 'native_word_ids': [101]*128}]
    source = engine.MixedSources(Public(), anchors, Cache())
    refs = [engine.curriculum.Example(pop, i, 1, pos, False, 1)
            for pos, (pop, i) in enumerate([('public', 1), ('anchor', 0), ('public', 0), ('public', 1)])]
    rows = source.fetch(refs)
    assert [r['id'] for r in rows] == ['p1', 'a0', 'p0', 'p1']
    assert [r['presentation'] for r in rows] == list(map(asdict, refs))
    digest = engine.batch_digest(rows)
    rows[0]['features'][0, 0] += .001
    assert engine.batch_digest(rows) != digest
