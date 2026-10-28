from collections import Counter
from dataclasses import replace

import pytest

from experiments.structure_net import ncm_recovered_source_balancing as study


def fixture():
    rows=[{'id':f'{s}-{v}','source_sha256':str(s),'source_component':str(s)} for s,n in enumerate((1,2,3,4,1,2,1)) for v in range(n)]
    config=study.Config(recovered_sources=7,anchor_cases=3,effective_examples=4,microbatch=1)
    return rows,config,study.groups_for(rows,config)


def test_every_source_has_equal_weight_and_every_target_is_seen():
    rows,config,groups=fixture();sources=Counter();targets=Counter()
    for number in range(1,5):
        epoch=study.Epoch(number,groups,config)
        examples=[epoch.example(i) for i in range(config.presentations_per_epoch)]
        assert sum(e.noisy for e in examples)==config.presentations_per_epoch//2
        for e in examples:
            if e.population=='recovered':sources[rows[e.index]['source_sha256']]+=1;targets[rows[e.index]['id']]+=1
    assert set(sources.values())=={4} and set(targets)=={r['id'] for r in rows}
    for group in groups:
        values=[targets[k] for k in group['pair_ids']]
        assert max(values)-min(values)<=1


def test_partial_update_keeps_actual_denominator_and_resume_is_exact():
    _,config,groups=fixture();epoch=study.Epoch(3,groups,config)
    batches=list(epoch.batches())
    assert batches[-1][1]==config.presentations_per_epoch%config.effective_examples==2
    assert [r for r in batches if r[0]>=1]==list(study.Epoch(3,groups,config).batches(start_update=1))


def test_variant_overflow_duplicate_pair_and_cross_component_source_are_rejected():
    rows,config,_=fixture()
    with pytest.raises(ValueError):study.groups_for(rows+[rows[0]],config)
    with pytest.raises(ValueError):study.groups_for(rows,replace(config,epochs=3))
    changed=[dict(r) for r in rows];changed[2]['source_component']='other'
    with pytest.raises(ValueError):study.groups_for(changed,config)
