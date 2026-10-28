from dataclasses import replace

import pytest

from experiments.structure_net import ncm_recovered_training_schedule as schedule


def test_full_contract_and_exposure_match_qualified_sampler():
    c = schedule.CONFIG; c.validate()
    assert (c.updates, c.presentations_per_epoch, c.updates_per_epoch) == (6416, 51312, 1604)
    assert c.epochs*c.presentations_per_epoch == 205248
    for name in schedule.balancing.Config.__dataclass_fields__:
        assert getattr(c, name) == getattr(schedule.balancing.CONFIG, name)
    assert len({c.sampler_seed, c.variant_seed, c.corruption_seed, c.dropout_seed}) == 4
    with pytest.raises(ValueError): replace(c, warmup=c.updates).validate()
    with pytest.raises(ValueError): replace(c, physical_GPU=0).validate()


def test_epoch_boundaries_partial_denominators_and_resume():
    c = replace(schedule.CONFIG, recovered_sources=7, anchor_cases=3, epochs=4,
                effective_examples=4, microbatch=2, warmup=1)
    rows = [dict(id=f'p{i}', source_sha256=f's{i//2}', source_component=f'c{i//2}') for i in range(14)]
    groups = schedule.balancing.groups_for(rows, c)
    all_batches = list(schedule.batches(groups, config=c))
    boundaries = [(refs[0].epoch, index, size, refs[0].position) for index, size, refs in all_batches]
    assert (1, 2, 2, 8) in boundaries and (2, 0, 4, 0) in boundaries
    start = c.updates_per_epoch-1
    resumed = list(schedule.batches(groups, start_update=start, config=c))
    expected = [b for b in all_batches if (b[2][0].epoch-1)*c.updates_per_epoch+b[0] >= start]
    assert resumed == expected
    assert len([e for _, _, refs in all_batches for e in refs]) == c.epochs*c.presentations_per_epoch
    assert list(schedule.batches(groups, start_update=c.updates, config=c)) == []
    with pytest.raises(ValueError): list(schedule.batches(groups, start_update=-1, config=c))
