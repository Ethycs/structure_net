from dataclasses import replace

import pytest
import torch

from experiments.structure_net import ncm_task_supervision_schedule as study


def config():return replace(study.CONFIG,updates=16,effective_examples=8,microbatch=4,warmup=2)


def test_complete_permutations_visit_every_case_before_reuse():
    for cases in (7,11):
        stream=study.Schedule(cases,config())
        for offset in range(0,len(stream.indices)-cases+1,cases):
            assert sorted(stream.indices[offset:offset+cases])==list(range(cases))
        assert max(stream.indices)<cases and min(stream.indices)>=0


def test_both_arms_share_update_budget_and_exact_noise_positions():
    a,b=study.Schedule(7,config()),study.Schedule(11,config())
    assert len(a.indices)==len(b.indices)==128
    assert torch.equal(a.noisy,b.noisy) and int(a.noisy.sum())==64
    for i in range(128):
        assert a.example(i).corruption_seed==b.example(i).corruption_seed
    assert len({a.example(i).corruption_seed for i in range(128)})==128


def test_resuming_at_completed_update_reproduces_exact_microbatch_suffix():
    first=study.Schedule(17,config());again=study.Schedule(17,config())
    original=list(first.batches())
    assert list(again.batches(start_update=5))==original[5*2:]
    for index,denominator,examples in original:
        assert denominator==8 and len(examples)==4
        assert all(e.epoch==1 and e.population=='anchor' for e in examples)
        assert [e.position for e in examples] in (list(range(index*8,index*8+4)),list(range(index*8+4,index*8+8)))


def test_lr_schedule_uses_the_full_fixed_budget():
    c=config()
    assert study.shared.rate(c.warmup,c)==pytest.approx(c.max_lr)
    assert study.shared.rate(c.updates,c)==pytest.approx(c.min_lr)
    assert study.CONFIG.presentations_per_epoch==65536
    assert study.CONFIG.updates_per_epoch==2048


@pytest.mark.parametrize('change',[{'epochs':2},{'microbatch':3},{'warmup':16},{'min_lr':0},{'updates':0}])
def test_invalid_geometry_cannot_change_exposure_or_denominators(change):
    with pytest.raises(ValueError):study.Schedule(17,replace(config(),**change))


def test_out_of_range_resume_and_empty_population_rejected():
    with pytest.raises(ValueError):study.Schedule(0,config())
    s=study.Schedule(17,config())
    with pytest.raises(IndexError):s.update(16)
    with pytest.raises(ValueError):list(s.batches(5,4))
