from collections import Counter
from dataclasses import FrozenInstanceError

import pytest
import torch

from experiments.structure_net import ncm_onpolicy_residual_training as run


def test_fixed_matched_budget_schedule_and_separate_roots():
    c = run.CONFIG
    assert c.updates == 1024 and c.accumulation == 4 and c.warmup == 64
    assert run.rate(1) == c.max_lr/64 and run.rate(64) == c.max_lr
    assert run.rate(1024) == c.min_lr
    assert len(set(run.ROOTS.values())) == 4
    for step in (0,1025):
        with pytest.raises(ValueError): run.rate(step)
    with pytest.raises(FrozenInstanceError): c.updates = 8


def test_matched_without_replacement_sampling_covers_every_train_case():
    rows = [{'id':str(i)} for i in range(1376)]
    def draw():
        permutation, generator = [],torch.Generator().manual_seed(run.CONFIG.sampler_seed)
        chosen = [run.select_row(rows,None,None,permutation,generator,j%4,'train')['id'] for j in range(4096)]
        return chosen,permutation,generator.get_state()
    a,b = draw(),draw()
    assert a[:2] == b[:2] and torch.equal(a[2],b[2])
    counts = Counter(a[0])
    assert len(counts) == 1376 and set(counts.values()) == {2,3}
    assert len(set(a[0][:1376])) == 1376


def test_pilot_uses_fixed_first_and_longest_without_consuming_sampler():
    lookup = {'first':{'id':'first'},'long':{'id':'long'}}
    selection = {'first_id':'first','longest_source_id':'long'}
    generator = torch.Generator().manual_seed(7)
    initial = generator.get_state().clone()
    permutation = []
    actual = [run.select_row(None,lookup,selection,permutation,generator,j,'pilot')['id'] for j in range(4)]
    assert actual == ['first','long','first','long']
    assert not permutation and torch.equal(initial,generator.get_state())


def test_atomic_checkpoint_contains_optimizer_rng_and_pending_sampler(tmp_path):
    model = torch.nn.Linear(2,2)
    opt = run.broad.offload.CPUAdamW(model.parameters(),lr=1e-4)
    model(torch.ones(1,2)).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler('cuda',enabled=False)
    sampler = torch.Generator().manual_seed(4)
    path = tmp_path/'model.pt'
    proof = {'arm':'fixed_targets','fingerprint':'test'}
    run.save(path,model,opt,scaler,sampler,[4,1,7],2,proof)
    value = torch.load(path,map_location='cpu',weights_only=True)
    assert value['permutation'] == [4,1,7] and value['step'] == 2
    assert value['arm'] == 'fixed_targets' and value['producer_fingerprint'] == 'test'
    assert torch.equal(value['sampler_rng'],sampler.get_state())
    assert all(torch.equal(v,value['model'][k]) for k,v in model.state_dict().items())
    assert len(value['optimizer']['state']) == 2 and not path.with_suffix('.partial').exists()
    with pytest.raises(ValueError,match='overwrite'):
        run.save(path,model,opt,scaler,sampler,[],2,proof)
