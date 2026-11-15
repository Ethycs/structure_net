"""CPU lifecycle tests: untouched-data admission and mocked reserved phases."""
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from experiments.structure_net import jspace_3b_final_gate as study


def cases():
    return [dict(identity=f'{family}-{i}',family=family,answer=0,choices=['a','b'],input_ids=[1,2,3,4])
            for family in study.confirmation.DATASETS for i in range(128)]


def rows(items):
    return [dict(identity=c['identity'],family=c['family'],answer=c['answer'],prediction=0,logits=[2.,1.]) for c in items]


def tasks():
    development=[dict(c,split='development',loss_positions=[0,1,2,3],targets=[2,3,4,0],offset=4*i,stop=4*i+4)
                 for i,c in enumerate([c for c in cases() if int(c['identity'].rsplit('-',1)[1])<16])]
    return dict(confirmation=cases(),development=development,all_original_items=development,
        benchmark=study.benchmark_cases(development))


def runtime(benchmark,rate):
    samples=[dict(repeat=repeat,identity=c['identity'],tokens=32,seconds=32/rate) for repeat in range(5) for c in benchmark]
    return dict(graph_static_exact=True,timings={'cuda_graph':dict(samples=samples,total_tokens=1280,
        total_seconds=sum(r['seconds'] for r in samples),tokens_per_second=rate)})


def phase_result(phase,plan,data):
    result=dict(status='completed',phase=phase,fingerprint=plan['fingerprint'])
    if phase=='compact_export':return {**result,'embedding_control':{'exact':True},'selected_development':{'passed':True},
        'native_cache_controls':{'eager':{'passed':True},'sdpa':{'passed':True}},'tensor_bytes':600}
    if phase=='compact_validate':return {**result,'fresh_export_reload_exact':True,'selected_development':{'passed':True},'cache':{'passed':True}}
    rate={'native_eager_before':80,'native_sdpa_before':100,'compact':190,'native_sdpa_after':200}[phase]
    return {**result,'runtime':runtime(data['benchmark'],rate),'tensor_bytes':600 if phase=='compact' else 1000,
        'fresh_export_reload_exact':True,'confirmation_reload_exact':True,'baseline_quality_exact':True}


def install(root,phase,plan,data,*,change=None):
    output=root/phase;output.mkdir();result=phase_result(phase,plan,data)
    if change:change(result)
    if phase in ('compact_export','compact_validate'):
        study.base.write(output/'development.json',{'records':rows(data['development'])})
    if phase in ('native_eager_before','native_sdpa_before','compact','native_sdpa_after'):
        study.base.write(output/'quality.json',{'records':rows(data['confirmation'])})
    if phase=='compact':study.base.write(output/'reload_quality.json',{'records':rows(data['confirmation'])})
    study.base.write(output/'result.json',result)
    study.base.write(output/'manifest.json',dict(status='completed',fingerprint=plan['fingerprint'],
        artifacts={p.name:study.base.sha(p) for p in output.iterdir() if p.is_file()}))


def fixture_root(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    data=tasks();study.base.write(tmp_path/'tasks.json',data)
    reference=tmp_path/'selected.json';study.base.write(reference,{'records':rows(data['development'])})
    plan=dict(fingerprint='final-fingerprint',model={'path':'native'},selected_reference=str(reference),
        label_ids=[0,1,2,3],fits={str(i):'unused' for i in range(36)},selected_checkpoint='selected.pt',native_boundary_zero='boundary0')
    monkeypatch.setattr(study,'verify',lambda root:plan)
    return plan,data


def test_failed_full_model_admission_precedes_any_confirmation_read(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(study.campaign,'verify_root',lambda root:{})
    campaign=tmp_path/'campaign';campaign.mkdir()
    study.base.write(campaign/'plan.json',dict(schema=study.campaign.SCHEMA))
    study.base.write(campaign/'outcome.json',dict(status='quality_stop',last_accepted_cut=28,compressed_blocks=8))
    monkeypatch.setattr(study.confirmation,'verify',lambda root:pytest.fail('confirmation opened before admission'))
    with pytest.raises(ValueError,match='full36'):study.prepare(tmp_path/'final',campaign,tmp_path/'closed_confirmation')
    assert not (tmp_path/'final').exists()


@pytest.mark.parametrize('schema,module',[('jspace.qwen3b-native-budget.v1','jspace_qwen3b_native_budget'),
    ('jspace.3b-staged-campaign.v1','jspace_3b_staged_campaign')])
def test_native_budget_admission_keeps_seed_policy_and_checks_every_new_transition(tmp_path,monkeypatch,schema,module):
    """Producer validation owns the seed audit; final admission retains that receipt."""
    receipt=tmp_path/'seed_gate_receipt.json'
    study.base.write(receipt,dict(old_quality_pass=False,new_quality=dict(passed=True,
        policy='original_native_only',parent_gate_required=False,parent=dict(passed=False))))
    plan=dict(schema=schema,fingerprint='campaign',seed_cut=2,
        seed_correct=47,seed_checkpoint_sha256='seed',seed_gate_receipt=str(receipt))
    study.base.write(tmp_path/'plan.json',plan)
    study.base.write(tmp_path/'outcome.json',dict(status='all_blocks_recovered',last_accepted_cut=0,compressed_blocks=36))
    checked=[]
    def stage_dir(root,cut):return Path(root)/'stages'/f'block_{cut:02d}'
    def check(directory,phase,stage):
        checked.append((stage['cut'],phase))
        return study.read(directory/phase/'result.json')
    producer=SimpleNamespace(verify_root=lambda root:plan,stage_dir=stage_dir,
        check_plan=lambda stage:None,checked=check)
    monkeypatch.setitem(sys.modules,'experiments.structure_net.'+module,producer)
    previous='seed'
    for cut in (1,0):
        directory=stage_dir(tmp_path,cut);(directory/'train').mkdir(parents=True);(directory/'verify').mkdir()
        stage=dict(root_fingerprint='campaign',fingerprint=f'stage{cut}',cut=cut,
            parent_checkpoint_sha256=previous,parent_correct=47)
        study.base.write(directory/'plan.json',stage)
        trained=dict(quality_pass=True,selected_checkpoint_sha256=f'checkpoint{cut}',selected={'correct':47},
            selected_epoch=0,quality=dict(passed=True,policy='original_native_only',parent_gate_required=False))
        for phase,value in (('train',trained),('verify',dict(passed=True))):
            study.base.write(directory/phase/'manifest.json',dict(status='completed'))
            study.base.write(directory/phase/'result.json',value)
        study.base.write(directory/'train/epoch_00_development.json',dict(records=[]))
        previous=f'checkpoint{cut}'
    value=study.admit_campaign(tmp_path)
    assert checked==[(1,'train'),(1,'verify'),(0,'train'),(0,'verify')]
    assert [s['cut'] for s in value['chain']]==[1,0]
    assert value['seed_gate_receipt']['old_quality_pass'] is False
    assert str(receipt) in value['input_sha256']
    broken=study.read(stage_dir(tmp_path,0)/'plan.json');broken['parent_checkpoint_sha256']='wrong'
    (stage_dir(tmp_path,0)/'plan.json').write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='parent chain'):study.admit_campaign(tmp_path)


def test_unknown_campaign_schema_cannot_open_confirmation(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    campaign=tmp_path/'campaign';campaign.mkdir()
    study.base.write(campaign/'plan.json',dict(schema='unknown'))
    monkeypatch.setattr(study.confirmation,'verify',lambda root:pytest.fail('confirmation opened'))
    with pytest.raises(ValueError,match='Unsupported'):study.prepare(tmp_path/'final',campaign)


def test_confirmation_population_and_prior_prediction_leakage():
    population=cases()
    assert study.validate_confirmation(population,set())['count']==512
    with pytest.raises(ValueError,match='separation'):study.validate_confirmation(population,{population[0]['identity']})
    changed=deepcopy(population);changed[-1]=changed[0]
    with pytest.raises(ValueError):study.validate_confirmation(changed,set())
    assert study._recorded_predictions({'excluded_identities':['heldout'],'records':[{'identity':'used','prediction':0}]})=={'used'}


def test_historical_task_exclusions_are_not_misclassified_as_scoring(tmp_path):
    root=tmp_path/'jspace_previous/run';root.mkdir(parents=True)
    study.base.write(root/'tasks.json',{'splits':{'train':[{'identity':'train'}],'confirmation':[{'identity':'unscored'}]}})
    study.base.write(root/'quality.json',{'records':[{'identity':'scored','prediction':0}],'excluded_identities':['unscored']})
    used,paths=study.historical_usage(tmp_path,tmp_path/'different_confirmation')
    assert used=={'train','scored'} and len(paths)==2


def test_checked_requires_fresh_reload_and_authenticated_confirmation(tmp_path,monkeypatch):
    plan,data=fixture_root(tmp_path,monkeypatch)
    install(tmp_path,'compact_validate',plan,data,change=lambda result:result.update(fresh_export_reload_exact=False))
    with pytest.raises(ValueError,match='before confirmation'):study.checked(tmp_path,'compact_validate',plan)
    install(tmp_path,'native_eager_before',plan,data)
    assert study.checked(tmp_path,'native_eager_before',plan)['tensor_bytes']==1000
    (tmp_path/'native_eager_before/quality.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact'):study.checked(tmp_path,'native_eager_before',plan)


def test_finalize_uses_pooled_sdpa_rate_and_has_no_family_gate(tmp_path,monkeypatch):
    plan,data=fixture_root(tmp_path,monkeypatch)
    for phase in study.PHASES:install(tmp_path,phase,plan,data)
    result=study.finalize(tmp_path)
    assert result['pooled_sdpa_tokens_per_second']==pytest.approx(2560/(12.8+6.4))
    assert result['decision']['speedup_vs_sdpa']==pytest.approx(190/(2560/19.2))
    assert result['decision']['passed'] and not result['decision']['family_accuracy_is_gate']
    assert result['confirmed'] is False


def mock_gpu(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_full_lease
    plan,data=fixture_root(tmp_path,monkeypatch)
    lease=dict(id='test-lease',allocation={'GPU-0370fdb0-test':11264})
    monkeypatch.setattr(jspace_full_lease,'wait_ready',lambda slots:lease)
    monkeypatch.setattr(study.campaign,'cap_allocator',lambda lease,config:{'allocator_limit_bytes':1024})
    for name in ('init','reset_peak_memory_stats','empty_cache'):monkeypatch.setattr(study.torch.cuda,name,lambda:None)
    monkeypatch.setattr(study.torch.cuda,'max_memory_reserved',lambda:0)
    monkeypatch.setattr(study.torch,'set_num_threads',lambda n:None)
    monkeypatch.setattr(study.torch,'manual_seed',lambda seed:None)
    monkeypatch.setattr(study.evaluator,'arithmetic_policy',lambda:nullcontext())
    monkeypatch.setattr(study.AutoTokenizer,'from_pretrained',lambda *a,**k:object())
    monkeypatch.setattr(study.evaluator,'score_development',lambda model,items,labels,**kw:rows(items))
    monkeypatch.setattr(study.evaluator,'score_model',lambda model,items,labels,**kw:rows(items))
    monkeypatch.setattr(study,'_model_bytes',lambda model:600)
    monkeypatch.setattr(study,'_cache_controls',lambda *a,**k:{'passed':True,'records':[]})
    monkeypatch.setattr(study.evaluator,'runtime_adapter',lambda model,tokenizer,items,**kw:runtime(items,190))
    return plan,data


def test_compact_phase_loads_only_standalone_bundle_and_recomputes_confirmation(tmp_path,monkeypatch):
    plan,data=mock_gpu(tmp_path,monkeypatch)
    for phase in study.PHASES[:study.PHASES.index('compact')]:install(tmp_path,phase,plan,data)
    loads=[]
    def load(path,**kwargs):
        assert path==tmp_path/'compact_export/model'
        assert kwargs['expected_fingerprint']==plan['fingerprint'];loads.append(path)
        return object(),object()
    monkeypatch.setattr(study.evaluator,'load_model_bundle',load)
    monkeypatch.setattr(study.evaluator,'build_compact_model',lambda *a,**k:pytest.fail('recompiled original checkpoint'))
    result=study.run(tmp_path,'compact')
    assert result==dict(status='completed',phase='compact') and len(loads)==2
    assert study.checked(tmp_path,'compact',plan)['confirmation_reload_exact']


def test_denied_lease_precedes_model_or_confirmation_scoring(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_full_lease
    fixture_root(tmp_path,monkeypatch)
    def denied(slots):raise RuntimeError('lease denied')
    monkeypatch.setattr(jspace_full_lease,'wait_ready',denied)
    monkeypatch.setattr(study.evaluator,'load_native_model',lambda *a,**k:pytest.fail('model before lease'))
    with pytest.raises(RuntimeError,match='lease denied'):study.run(tmp_path,'compact_export')
    failure=study.read(tmp_path/'compact_export/failure.json')
    assert failure['lease_id'] is None and failure['peak_reserved_bytes'] is None
    with pytest.raises(FileExistsError):study.run(tmp_path,'compact_export')


def test_missing_compact_validation_blocks_confirmation_phase_without_lease(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_full_lease
    plan,data=fixture_root(tmp_path,monkeypatch);install(tmp_path,'compact_export',plan,data)
    monkeypatch.setattr(jspace_full_lease,'wait_ready',lambda slots:pytest.fail('lease before previous validity'))
    with pytest.raises(FileNotFoundError):study.run(tmp_path,'native_eager_before')
