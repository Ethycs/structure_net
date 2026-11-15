"""CPU-only orchestration, resume and authenticated stage-gate contracts."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.structure_net import jspace_qwen3b_progressive as study


def stage(cut=28):
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
        quality=study.old.gates(correct,parent=51),compressed_blocks=36-cut,storage={'full_model_tensor_bytes':123}),{}
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
    current=stage();install(tmp_path,'train',current,correct=47);install(tmp_path,'verify',current,correct=47)
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
        install(directory,phase,current,correct=47 if miss else 51)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(study.subprocess,'run',run)
    return calls,prepare


def test_campaign_resumes_only_completed_phases_and_each_job_is_reserved(tmp_path,monkeypatch):
    calls,prepare=campaign_fixture(tmp_path,monkeypatch)
    current=prepare(tmp_path,28);install(study.stage_dir(tmp_path,28),'capture',current)
    result=study.campaign(tmp_path,stop_cut=28)
    assert result['compressed_blocks']==8
    assert [c[c.index('-m')+2] for c in calls]==['fit','pilot','train','verify']
    for command in calls:
        assert command[:5]==['/usr/bin/python3','/usr/local/bin/inference-ctl','lease','run','--allocation']
        assert command[5]=='GPU-0370fdb0=11264'
        assert command[command.index('--')+1:command.index('-m')]==['pixi','run','--locked','--offline','-e','ncm','python']
        assert command[command.index('-m')+1]=='experiments.structure_net.jspace_qwen3b_progressive'


def test_quality_miss_verifies_then_stops_without_preparing_next_layer(tmp_path,monkeypatch):
    calls,_=campaign_fixture(tmp_path,monkeypatch,miss=True)
    result=study.campaign(tmp_path,stop_cut=27)
    assert result['status']=='quality_stop' and result['last_accepted_cut']==29
    assert [c[c.index('-m')+2] for c in calls]==list(study.PHASES)
    assert not study.stage_dir(tmp_path,27).exists()


def test_infrastructure_failure_stops_without_retry_or_later_phase(tmp_path,monkeypatch):
    calls,_=campaign_fixture(tmp_path,monkeypatch,fail_phase='pilot')
    with pytest.raises(RuntimeError,match='pilot failed'):study.campaign(tmp_path,stop_cut=27)
    assert [c[c.index('-m')+2] for c in calls]==['capture','fit','pilot']
    assert not study.stage_dir(tmp_path,27).exists()
    assert not (tmp_path/'outcome.json').exists()


def test_incomplete_phase_is_not_resumed_or_overwritten(tmp_path,monkeypatch):
    calls,prepare=campaign_fixture(tmp_path,monkeypatch);prepare(tmp_path,28)
    (study.stage_dir(tmp_path,28)/'capture').mkdir()
    with pytest.raises(RuntimeError,match='Incomplete phase'):study.campaign(tmp_path,stop_cut=28)
    assert calls==[]


@pytest.mark.parametrize('quality,verified',[(False,True),(True,False)])
def test_stage_preparation_rejects_unaccepted_parent(tmp_path,monkeypatch,quality,verified):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(study,'verify_root',lambda root:{'fits':{},'fingerprint':'campaign'})
    parent=study.stage_dir(tmp_path,28);parent.mkdir(parents=True);study.base.write(parent/'plan.json',stage())
    monkeypatch.setattr(study,'checked',lambda directory,phase,current:
        {'quality_pass':quality} if phase=='train' else {'passed':verified})
    with pytest.raises(ValueError,match='unaccepted'):study.prepare_stage(tmp_path,27)
    assert not study.stage_dir(tmp_path,27).exists()
