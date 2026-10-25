import pytest

from experiments.structure_net import ncm_engineering_attention_repair_v2 as run


def test_fixed_learning_rate_endpoints():
    c=run.Config()
    assert run.rate(1)==pytest.approx(c.max_lr/c.warmup)
    assert run.rate(c.warmup)==c.max_lr
    assert run.rate(c.updates)==pytest.approx(c.min_lr)
    with pytest.raises(ValueError):run.rate(c.updates+1)


def test_train_probes_cover_components_without_target_selection():
    rows=[{'id':f'{group}{i}','source_component':group,'target_text':'irrelevant'} for group in ('a','b','c','d') for i in range(8)]
    index=[{'id':r['id'],'source_positions':i+1} for i,r in enumerate(rows)]
    chosen=run.diagnostic_ids(rows,index)
    assert len(chosen)==len(set(chosen))==16
    assert {f'{g}{i}' for g in ('a','b','c','d') for i in (0,7)}<=set(chosen)
    assert chosen==run.diagnostic_ids([{**r,'target_text':'changed'} for r in rows],index)
