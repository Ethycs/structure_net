import pytest

from experiments.structure_net import ncm_engineering_attention_repair_v3 as run


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


@pytest.mark.parametrize('change',[
    {'status':'capacity_gate_failed'},
    {'last64_mean_word_CE':6.},
    {'last64_mean_word_CE':float('nan')},
    {'empty_outputs':1},
    {'decoded_TRAIN_text_adequate_to_scale':False},
    {'capacity_result_sha256':'different-run'},
])
def test_completed_capacity_cannot_bypass_quality_or_provenance_gate(change):
    gate={'status':'capacity_gate_passed','full_v3_training_admitted':True,
        'first64_mean_word_CE':8.,'last64_mean_word_CE':3.,'updates':512,'TRAIN_cases':16,
        'changed_outputs':16,'empty_outputs':0,'strict_reload_passed':True,
        'decoded_TRAIN_text_reviewed':True,'decoded_TRAIN_text_adequate_to_scale':True,
        'capacity_result_sha256':'this-run',
        'frozen_renderer_sha256':'2effdc560c0a46bc4e5b1ce419a2f47896230eadff5cbccae70d5e7983561376'}
    run.validate_capacity_gate(gate,'this-run')
    with pytest.raises(ValueError):run.validate_capacity_gate({**gate,**change},'this-run')
