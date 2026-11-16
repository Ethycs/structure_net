from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.structure_net import jspace_3b_ablation32_data as data


def records():
    rows=[]
    for family,size in [('a',12),('b',5),('c',9)]:
        for i in range(size):
            rows.append(SimpleNamespace(split='train',example_id=f'{family}{i}',group_id=f'{family}{i//2}',
                provenance={'source_family':family},response_ids=[1,2]))
    rows.append(SimpleNamespace(split='development',example_id='dev',group_id='dev',provenance={'source_family':'a'},response_ids=[1]))
    return rows


def test_group_disjoint_selection_and_distillation_no_dev():
    rows=records();p=data.choose(rows,fit_count=6,selection_count=4)
    held={rows[i].group_id for i in p['selection_indices']}
    assert len(held)==4
    for name in ('training_indices','distillation_indices'):
        assert not held.intersection(rows[i].group_id for i in p[name])
        assert all(rows[i].split=='train' for i in p[name])
    assert len({rows[i].group_id for i in p['training_indices']})==6
    assert p==data.choose(rows,fit_count=6,selection_count=4)
    assert len(p['distillation_indices'])<len(rows)-4


def test_scarce_families_exhaust_without_duplicates_and_insufficient_data_fails():
    rows=records();p=data.choose(rows,fit_count=10,selection_count=4)
    assert len(p['training_indices'])==10
    with pytest.raises(ValueError,match='Insufficient'):
        data.choose(rows,fit_count=100,selection_count=4)


def test_anchors_use_only_selected_fit_rows_and_global_fallback():
    values=torch.tensor([[1.,2.],[3.,4.],[100.,200.],[5.,6.]],dtype=torch.bfloat16)
    bank=values.view(torch.int16).numpy()
    rows=[dict(offset=0,stop=2),dict(offset=2,stop=3),dict(offset=3,stop=4)]
    means,counts=data.positional_means(bank,rows,[0,2],capacity=3,width=2)
    torch.testing.assert_close(means,torch.tensor([[3.,4.],[3.,4.],[3.,4.]]))
    assert counts.tolist()==[2,1,0]
    assert means.dtype==torch.float32


def test_epoch_order_covers_fitting_only_and_varies_deterministically():
    plan=dict(training_indices=list(range(50)),seed=220301)
    a=data.epoch_order(plan,0);b=data.epoch_order(plan,1)
    assert sorted(a)==sorted(b)==plan['training_indices'] and a!=b
    assert data.epoch_order(plan,0)==a
    with pytest.raises(ValueError):data.epoch_order(plan,8)
