from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from experiments.structure_net import ncm_lm1b_grounded_training as run


def test_fixed_budget_schedule_and_immutable_config():
    c=run.Config()
    assert c.updates==4096 and run.rate(128)==c.max_lr and run.rate(4096)==c.min_lr
    with pytest.raises(ValueError):run.rate(0)
    with pytest.raises(ValueError):run.rate(4097)
    with pytest.raises(FrozenInstanceError):c.updates=2


def test_actual690_and_both_permanent_DEV_cohorts_are_kept_disjoint():
    train=run.q.read(run.native.CASES)['train'];dev=run.q.read(run.DEV)
    proof=run.disjoint(train,dev)
    assert proof['source_id']=={'train_unique':690,'short_unique':64,'long_unique':64,'overlap':0}
    assert all('target_text' not in r and 'word_ids' not in r and 'concept_ids' not in r for part in dev.values() for r in part)
    bad=deepcopy(train);bad[0]['group_id']=dev['short_development'][0]['group_id']
    with pytest.raises(ValueError,match='overlap'):run.disjoint(bad,dev)


def test_complete_scope_cannot_be_reduced_for_a_passing_gate():
    train=run.q.read(run.native.CASES)['train'];dev=run.q.read(run.DEV)
    with pytest.raises(ValueError,match='full690/64/64'):run.disjoint(train[:-1],dev)
