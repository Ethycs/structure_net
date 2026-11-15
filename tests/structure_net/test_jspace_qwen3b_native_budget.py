"""CPU-only orchestration, resume and authenticated stage-gate contracts."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.structure_net import jspace_qwen3b_native_budget as study


def stage(cut=25):
    value=dict(cut=cut,root_fingerprint='campaign',parent_cut=cut+1,parent_correct=51,
        teacher_fingerprint='teacher',layers=list(range(cut,36)))
    value['fingerprint']=study.base.digest(value)
    return value


def write_phase(directory,phase,result,current,extra=None):
    path=directory/phase;path.mkdir()
    for name,value in (extra or {}).items():
        if isinstance(value,bytes):(path/name).write_bytes(value)
        else:study.base.write(path/name,value)
    study.base.write(path/'result.json',result)
    study.base.write(path/'manifest.json',dict(status='completed',fingerprint=current['fingerprint'],
        artifacts={p.name:study.base.sha(p) for p in path.iterdir() if p.is_file()}))


def result_for(phase,current,correct=51):
    cut=current['cut'];layers=current['layers']
    if phase=='capture':return dict(status='completed',controls={'passed':True},
        teacher_cache_diagnostic={'predictions_exact':True},teacher_fingerprint='teacher'),{}
    if phase=='fit':return dict(status='completed',cut=cut,teacher_fingerprint='teacher'),{}
    if phase=='pilot':return dict(status='completed',passed=True,systems_only=True,discarded=True,
        layer_indices=layers,update={'update':1},fp32_master_changed_by_block=[True]*len(layers),
        norm_fp32_master_changed=True,fixed_state_unchanged=True,checkpoint_strict_load=True,inheritance={'passed':True}),{}
    metrics=dict(correct=correct,count=64,ce=1.)
    if phase=='train':
        extra={'selected.pt':b'checkpoint'};history=[]
        checkpoint_sha=__import__('hashlib').sha256(extra['selected.pt']).hexdigest()
        for epoch in range(13):
            filename=f'epoch_{epoch:02d}_development.json'
            payload=dict(metrics=metrics,records=[]);extra[filename]=payload
            # base.write uses indent2/sorted keys; calculate from the actual file
            # later in install() so this fixture cannot silently assume encoding.
            training=[] if epoch==0 else [dict(update=(epoch-1)*32+j+1,
                gradients_present_by_block=[True]*len(layers)) for j in range(32)]
            row=dict(epoch=epoch,metrics=metrics,training=training,development_sha256='pending')
            if epoch==0:row['checkpoint_sha256']=checkpoint_sha
            history.append(row)
        quality=study.training.gates(correct,native_correct=48,parent_correct=51,count=64)
        return dict(status='completed',config=asdict(study.CONFIG),completed_epochs=12,updates=384,
            layer_indices=layers,strict_reload_exact=True,fixed_state_unchanged=True,history=history,
            selected_epoch=0,selected=metrics,initial=metrics,selected_checkpoint_sha256=checkpoint_sha,
            quality=quality,quality_pass=quality['passed']),extra
    if phase=='verify':return dict(status='completed',passed=True,comparison={'passed':True},fixed_state_unchanged=True,
        metrics=metrics,selected_checkpoint_sha256=__import__('hashlib').sha256(b'checkpoint').hexdigest(),
        quality=study.training.gates(correct,native_correct=48,parent_correct=51,count=64),compressed_blocks=36-cut,storage={'full_model_tensor_bytes':123}),{}
    raise AssertionError(phase)


def install(directory,phase,current,correct=51,mutate=None):
    result,extra=result_for(phase,current,correct)
    if phase=='train':
        path=directory/phase;path.mkdir()
        for name,value in extra.items():
            if isinstance(value,bytes):(path/name).write_bytes(value)
            else:study.base.write(path/name,value)
        for row in result['history']:row['development_sha256']=study.base.sha(path/f"epoch_{row['epoch']:02d}_development.json")
        if mutate:mutate(result)
        study.base.write(path/'result.json',result)
        study.base.write(path/'manifest.json',dict(status='completed',fingerprint=current['fingerprint'],
            artifacts={p.name:study.base.sha(p) for p in path.iterdir() if p.is_file()}))
    else:
        if mutate:mutate(result)
        write_phase(directory,phase,result,current,extra)
    return result


@pytest.mark.parametrize('phase,mutation',[
    ('pilot',lambda r:r.update(passed=False)),
    ('pilot',lambda r:r.update(norm_fp32_master_changed=False)),
    ('train',lambda r:r.update(updates=383)),
    ('train',lambda r:r.update(selected_epoch=1)),
    ('capture',lambda r:r.update(teacher_cache_diagnostic={'predictions_exact':False})),
])
def test_completed_manifest_does_not_override_failed_validity(tmp_path,phase,mutation):
    current=stage();install(tmp_path,phase,current,mutate=mutation)
    with pytest.raises(ValueError):study.checked(tmp_path,phase,current)


def test_scientific_quality_miss_remains_completed_and_fresh_verifiable(tmp_path):
    current=stage();install(tmp_path,'train',current,correct=44);install(tmp_path,'verify',current,correct=44)
    assert not study.checked(tmp_path,'train',current)['quality_pass']
    assert study.checked(tmp_path,'verify',current)['passed']
    (tmp_path/'train/selected.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):study.checked(tmp_path,'verify',current)


def test_unauthenticated_result_and_wrong_fingerprint_are_rejected(tmp_path):
    current=stage();install(tmp_path,'fit',current)
    path=tmp_path/'fit/manifest.json';manifest=study.read(path);del manifest['artifacts']['result.json']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='authenticated'):study.checked(tmp_path,'fit',current)
    manifest['fingerprint']='other';path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='matching'):study.checked(tmp_path,'fit',current)


def campaign_fixture(tmp_path,monkeypatch,*,miss=False,fail_phase=None):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(study,'verify_root',lambda root:{'fingerprint':'campaign'})
    def prepare(root,cut):
        directory=study.stage_dir(root,cut);directory.mkdir(parents=True)
        value=stage(cut);study.base.write(directory/'plan.json',value);return value
    monkeypatch.setattr(study,'prepare_stage',prepare)
    monkeypatch.setattr(study,'verify_stage',lambda root,cut:({'fingerprint':'campaign'},study.read(study.stage_dir(root,cut)/'plan.json')))
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv);phase=argv[argv.index('-m')+2];cut=int(argv[argv.index('--cut')+1])
        if phase==fail_phase:return SimpleNamespace(returncode=17)
        directory=study.stage_dir(tmp_path,cut);current=study.read(directory/'plan.json')
        install(directory,phase,current,correct=44 if miss else 51)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(study.subprocess,'run',run)
    return calls,prepare


def test_campaign_resumes_only_completed_phases_and_each_job_is_reserved(tmp_path,monkeypatch):
    calls,prepare=campaign_fixture(tmp_path,monkeypatch)
    current=prepare(tmp_path,25);install(study.stage_dir(tmp_path,25),'capture',current)
    result=study.campaign(tmp_path,stop_cut=25)
    assert result['compressed_blocks']==11
    assert [c[c.index('-m')+2] for c in calls]==['fit','pilot','train','verify']
    for command in calls:
        assert command[:5]==['/usr/bin/python3','/usr/local/bin/inference-ctl','lease','run','--allocation']
        assert command[5]=='GPU-0370fdb0=11264'
        assert command[command.index('--')+1:command.index('-m')]==['pixi','run','--locked','--offline','-e','ncm','python']
        assert command[command.index('-m')+1]=='experiments.structure_net.jspace_qwen3b_native_budget'


def test_quality_miss_verifies_then_stops_without_preparing_next_layer(tmp_path,monkeypatch):
    calls,_=campaign_fixture(tmp_path,monkeypatch,miss=True)
    result=study.campaign(tmp_path,stop_cut=24)
    assert result['status']=='quality_stop' and result['last_accepted_cut']==26
    assert [c[c.index('-m')+2] for c in calls]==list(study.PHASES)
    assert not study.stage_dir(tmp_path,24).exists()


def test_infrastructure_failure_stops_without_retry_or_later_phase(tmp_path,monkeypatch):
    calls,_=campaign_fixture(tmp_path,monkeypatch,fail_phase='pilot')
    with pytest.raises(RuntimeError,match='pilot failed'):study.campaign(tmp_path,stop_cut=24)
    assert [c[c.index('-m')+2] for c in calls]==['capture','fit','pilot']
    assert not study.stage_dir(tmp_path,24).exists()
    assert not (tmp_path/'outcome.json').exists()


def test_incomplete_phase_is_not_resumed_or_overwritten(tmp_path,monkeypatch):
    calls,prepare=campaign_fixture(tmp_path,monkeypatch);prepare(tmp_path,25)
    (study.stage_dir(tmp_path,25)/'capture').mkdir()
    with pytest.raises(RuntimeError,match='Incomplete phase'):study.campaign(tmp_path,stop_cut=25)
    assert calls==[]


@pytest.mark.parametrize('quality,verified',[(False,True),(True,False)])
def test_stage_preparation_rejects_unaccepted_parent(tmp_path,monkeypatch,quality,verified):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(study,'verify_root',lambda root:{'fits':{},'fingerprint':'campaign'})
    parent=study.stage_dir(tmp_path,25);parent.mkdir(parents=True);study.base.write(parent/'plan.json',stage())
    monkeypatch.setattr(study,'checked',lambda directory,phase,current:
        {'quality_pass':quality} if phase=='train' else {'passed':verified})
    with pytest.raises(ValueError,match='unaccepted'):study.prepare_stage(tmp_path,24)
    assert not study.stage_dir(tmp_path,24).exists()


def mocked_run_phase(tmp_path,monkeypatch):
    """Reach real branch serialization without CUDA/model execution."""
    from experiments.structure_net import jspace_full_lease
    current=stage();directory=study.stage_dir(tmp_path,25);directory.mkdir(parents=True)
    items=tmp_path/'items.json';study.base.write(items,{'items':[]})
    reference=tmp_path/'reference.json';study.base.write(reference,[])
    current['parent_records']=str(reference)
    plan=dict(label_ids=[32,33,34,35],model={'path':'unused'},items_path=str(items),
        native_cache=str(tmp_path),calibration_indices=[0],allocation_mib=11264,allocator_mib=11008)
    monkeypatch.setattr(study,'verify_stage',lambda root,cut:(plan,current))
    lease=dict(id='mock-lease',allocation={'GPU-0370fdb0-test':11264})
    monkeypatch.setattr(jspace_full_lease,'wait_ready',lambda slots:lease)
    monkeypatch.setattr(study,'cap_allocator',lambda supplied,configuration:{'allocator_limit_bytes':1024})
    monkeypatch.setattr(study.torch.cuda,'init',lambda:None)
    monkeypatch.setattr(study.torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(study.torch.cuda,'max_memory_reserved',lambda:0)
    monkeypatch.setattr(study.torch,'set_num_threads',lambda count:None)
    monkeypatch.setattr(study.torch,'manual_seed',lambda seed:None)
    class Module:
        def __init__(self):self.loads=[]
        def to(self,*args,**kwargs):return self
        def pack(self,**kwargs):return self
        def load_state_dict(self,value,strict):
            assert strict;self.loads.append(value)
    stack,norm,readout,rotary=[Module() for _ in range(4)]
    monkeypatch.setattr(study,'make_stack',lambda *a,**k:(stack,norm,readout,rotary,{'passed':True}))
    data=SimpleNamespace(training=list(range(32)),lengths=list(range(4,36)),development=[],items=[],closed=False)
    def close():data.closed=True
    data.close=close
    monkeypatch.setattr(study,'data_for',lambda *a,**k:data)
    return current,directory,stack,norm,data


def test_run_phase_pilot_serializes_real_probe_keys_and_strictly_reloads(tmp_path,monkeypatch):
    current,directory,stack,norm,data=mocked_run_phase(tmp_path,monkeypatch)
    for phase in ('capture','fit'):install(directory,phase,current)
    class Trainer:
        def __init__(self,*args,**kwargs):pass
        def packed_development(self):return {'correct':51,'count':64},[]
        def master_update_probe(self,indices):
            assert indices==list(range(31,15,-1))
            result,_=result_for('pilot',current)
            # Actual trainer probe has systems_only/discarded already, but the
            # lifecycle wrapper supplies status and initial/reload fields.
            for key in ('status','inheritance','checkpoint_strict_load'):result.pop(key)
            return result
        def snapshot(self):return {'stack':{},'norm':{}}
    monkeypatch.setattr(study.training,'NativeBudgetTrainer',Trainer)
    result=study.run_phase(tmp_path,25,'pilot')
    assert result==dict(status='completed',cut=25,phase='pilot')
    actual=study.checked(directory,'pilot',current)
    assert actual['systems_only'] and actual['discarded'] and actual['initial']['correct']==51
    assert stack.loads==[{}] and norm.loads==[{}] and data.closed
    assert not (directory/'pilot/failure.json').exists()


def test_run_phase_capture_progress_accepts_existing_cut_and_phase_fields(tmp_path,monkeypatch):
    reused=False
    current,directory,_,_,data=mocked_run_phase(tmp_path,monkeypatch)
    if reused:
        plan,_=study.verify_stage(tmp_path,25)
        original=tmp_path/'old_capture';original.mkdir()
        artifact=original/'accepted_teacher_logits.npy';artifact.write_bytes(b'differentiable original')
        plan['calibration_reuse']=dict(root=str(original),source_fingerprint='original',
            files={artifact.name:study.base.sha(artifact)})
    monkeypatch.setattr(study,'controls',lambda *a,**k:{'passed':True})
    monkeypatch.setattr(study.native,'load_native_block',lambda *a,**k:object())
    teacher=SimpleNamespace(closed=False)
    def close():teacher.closed=True
    teacher.close=close
    monkeypatch.setattr(study.calibration,'CachedPrefixTeacher',lambda *a,**k:teacher)
    def capture(output,model,indices,*,teacher_fingerprint,progress):
        assert not reused,'Reused VJPs must not be recaptured'
        output.mkdir()
        progress(dict(kind='case_completed',cut=999,phase='nested',case=7))
        return {'completed':True}
    monkeypatch.setattr(study.calibration,'capture',capture)
    calls=[]
    def data_for(plan,cut,path=None):
        calls.append((cut,None if path is None else path.name));return data
    monkeypatch.setattr(study,'data_for',data_for)
    def packed(path,*args,**kwargs):
        path.write_bytes(b'packed targets');return {'status':'completed','calibration_jacobians_changed':False}
    monkeypatch.setattr(study.targets,'capture_packed_teacher',packed)
    monkeypatch.setattr(study.old,'rows_from_cache',lambda *a,**k:[])
    comparisons=iter([{'predictions_exact':False,'passed':False},{'predictions_exact':True,'passed':True}])
    monkeypatch.setattr(study.old,'compare_rows',lambda *a,**k:next(comparisons))
    study.run_phase(tmp_path,25,'capture')
    result=study.checked(directory,'capture',current)
    assert result['controls']['passed'] and result['teacher_cache_logit_gate_applied']
    assert not result['differentiable_teacher_cache_diagnostic']['predictions_exact']
    assert calls==[(25,'accepted_teacher_logits.npy'),(26,None),(25,'packed_teacher_logits.npy')]
    assert 'packed_teacher_logits.npy' in study.read(directory/'capture/manifest.json')['artifacts']
    events=[json.loads(line) for line in (tmp_path/'progress.jsonl').read_text().splitlines()]
    matched=[event for event in events if event.get('kind',event.get('event'))=='case_completed']
    if reused:
        assert matched==[] and result['calibration_reused']
        assert (directory/'capture/accepted_teacher_logits.npy').read_bytes()==b'differentiable original'
    else:
        assert len(matched)==1 and matched[0]['cut']==25 and matched[0]['phase']=='capture'
    assert data.closed and teacher.closed is (not reused)



def seed_evidence():
    result=dict(selected_checkpoint_sha256=study.SEED_SHA,selected={'correct':47,'count':64},quality_pass=False,
        quality=study.prior.training.gates(47,native_correct=48,parent_correct=52,count=64))
    verified=dict(passed=True,selected_checkpoint_sha256=study.SEED_SHA)
    outcome=dict(status='quality_stop',failed_cut=26,last_accepted_cut=27,selected=result['selected'])
    return result,verified,outcome


def test_native_budget_admits_named_old_miss_without_rewriting_old_result():
    args=seed_evidence();before=deepcopy(args)
    receipt=study.seed_gate(*args)
    assert args==before
    assert receipt['historical_quality_pass'] is False
    assert receipt['prospective_quality']['passed']
    assert receipt['prospective_quality']['minimum_correct']==45
    assert receipt['prospective_quality']['parent']['passed'] is False
    assert receipt['old_result_reclassified'] is False


@pytest.mark.parametrize('mutation',[
    lambda r,v,o:r.update(selected_checkpoint_sha256='other'),
    lambda r,v,o:v.update(passed=False),
    lambda r,v,o:r.update(quality_pass=True),
    lambda r,v,o:o.update(failed_cut=25),
    lambda r,v,o:r['quality'].update(minimum_correct=45),
])
def test_seed_admission_rejects_unverified_or_rewritten_legacy_evidence(mutation):
    args=seed_evidence();mutation(*args)
    with pytest.raises(ValueError):study.seed_gate(*args)


def test_parent_diagnostic_miss_does_not_stop_native_budget_campaign(tmp_path,monkeypatch):
    calls,_=campaign_fixture(tmp_path,monkeypatch)
    # The fixture's parent is51:47 misses its48 floor but passes fixed native45.
    real_install=install
    def run(argv,**kwargs):
        calls.append(argv);phase=argv[argv.index('-m')+2];cut=int(argv[argv.index('--cut')+1])
        directory=study.stage_dir(tmp_path,cut);current=study.read(directory/'plan.json')
        real_install(directory,phase,current,correct=47)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(study.subprocess,'run',run)
    result=study.campaign(tmp_path,stop_cut=24)
    assert result['status']=='requested_prefix_completed' and result['compressed_blocks']==12
    assert len(calls)==10
    trained=study.read(study.stage_dir(tmp_path,25)/'train/result.json')
    assert trained['quality_pass'] and not trained['quality']['parent']['passed']


def test_prepare_authenticates_all_old_phases_and_starts_at25(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    seed=tmp_path/'legacy';seed.mkdir();fits={}
    for layer in range(26,36):
        directory=seed/f'fit_{layer}';directory.mkdir();fits[str(layer)]=str(directory)
        for name in ('fit.pt','fit.json','relation.pt','relations.json','interface.json'):
            (directory/name).write_bytes(b'authenticated fixture')
    plan=dict(seed_cut=29,native_correct=48,input_sha256={},source_sha256={},fingerprint='old-campaign',
        model={'path':'unused'},native_cache=str(seed),items_path=str(seed/'items.json'),
        original_indices=list(range(576)),label_ids=[32,33,34,35],calibration_indices=list(range(128)))
    study.base.write(seed/'plan.json',plan)
    stages={};evidence={};calls=[]
    for cut in (28,27,26):
        directory=study.prior.stage_dir(seed,cut);directory.mkdir(parents=True)
        current=dict(cut=cut,fingerprint=f'stage{cut}',fits=fits,input_sha256={});stages[cut]=current
        study.base.write(directory/'plan.json',current)
        for phase in study.PHASES:
            path=directory/phase;path.mkdir();study.base.write(path/'manifest.json',{'artifacts':{}})
        (directory/'train/selected.pt').write_bytes(b'fixture checkpoint')
        study.base.write(directory/'train/epoch_00_development.json',{'records':[]})
        correct=47 if cut==26 else 52
        evidence[cut]=dict(selected_checkpoint_sha256=study.base.sha(directory/'train/selected.pt'),
            selected={'correct':correct,'count':64},selected_epoch=0,quality_pass=cut!=26,
            quality=study.prior.training.gates(correct,native_correct=48,parent_correct=52,count=64))
    monkeypatch.setattr(study,'SEED_SHA',evidence[26]['selected_checkpoint_sha256'])
    study.base.write(seed/'outcome.json',dict(status='quality_stop',failed_cut=26,last_accepted_cut=27,selected=evidence[26]['selected']))
    monkeypatch.setattr(study.prior,'verify_root',lambda root:plan)
    monkeypatch.setattr(study.prior,'verify_stage',lambda root,cut:(plan,stages[cut]))
    def checked(directory,phase,current):
        calls.append((current['cut'],phase))
        return evidence[current['cut']] if phase=='train' else dict(passed=True,
            selected_checkpoint_sha256=evidence[current['cut']]['selected_checkpoint_sha256'])
    monkeypatch.setattr(study.prior,'checked',checked)
    monkeypatch.setattr(study.shutil,'disk_usage',lambda _:SimpleNamespace(free=2**50))
    root=tmp_path/'new_parent'/'new_run';study.prepare(root,seed)
    prepared=study.verify_root(root)
    assert calls==[(cut,phase) for cut in (28,27,26) for phase in study.PHASES]
    assert prepared['seed_cut']==26 and prepared['seed_correct']==47
    receipt=study.read(prepared['seed_gate_receipt'])
    assert receipt['historical_quality_pass'] is False and receipt['prospective_quality']['passed']
    current=study.prepare_stage(root,25)
    assert current['parent_cut']==26 and current['parent_correct']==47
    assert current['parent_checkpoint_sha256']==study.SEED_SHA
    assert sorted(map(int,current['fits']))==list(range(25,36))
    Path(prepared['seed_gate_receipt']).write_text('{}')
    with pytest.raises(ValueError):study.verify_root(root)
