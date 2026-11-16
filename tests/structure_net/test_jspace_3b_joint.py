from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net import jspace_3b_joint as joint


def candidate(step,correct,eligible):
    return dict(updates=step,metrics={'correct':correct,'count':256},eligibility={'passed':eligible})


def test_selection_preserves_early_eligible_and_separate_diagnostic_best():
    raw=candidate(0,173,False);early=candidate(72,189,True)
    late=candidate(288,189,True);bad=candidate(144,193,False)
    result=joint.choose([raw,early,bad,late])
    assert result['candidate']==early and result['diagnostic_best']==bad
    assert result['development_passed'] and not result['confirmed']
    result=joint.choose([raw,bad])
    assert result['candidate']==bad and not result['development_passed']


def test_wrong_gpu_is_rejected_before_cuda_is_touched(tmp_path,monkeypatch):
    monkeypatch.setattr(joint,'verify',lambda root:{'phases':['pilot']})
    monkeypatch.setattr(joint,'wait_ready',lambda ids:{'id':'wrong','allocation':{'GPU-0370fdb0-test':5632}})
    def forbidden(*a,**k):raise AssertionError('CUDA must not be touched before admission')
    monkeypatch.setattr(joint.torch.cuda,'set_device',forbidden)
    with pytest.raises(ValueError,match='6GB2060'):joint.run(tmp_path,'pilot')
    failure=json.loads((tmp_path/'pilot/failure.json').read_text())
    assert failure['status']=='failed' and failure['peak_reserved_bytes']is None


def test_2060_receipt_requires_all36_blocks_and_memory_cap(tmp_path):
    p,_,_,_=evidence(tmp_path);out=tmp_path/'pilot'
    manifest=joint.read(out/'manifest.json')
    assert joint.checked(tmp_path,'pilot',p)['passed']
    manifest['peak_reserved_bytes']=5376*1024**2+1;(out/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='bounded RTX2060'):joint.checked(tmp_path,'pilot',p)


def test_incomplete_pilot_prevents_training_before_lease(tmp_path,monkeypatch):
    monkeypatch.setattr(joint,'verify',lambda root:{'phases':list(joint.PHASES)})
    def incomplete(*_):raise ValueError('pilot incomplete')
    monkeypatch.setattr(joint,'checked',incomplete)
    monkeypatch.setattr(joint,'wait_ready',lambda *_:pytest.fail('reserved before pilot admission'))
    with pytest.raises(ValueError,match='incomplete'):joint.run(tmp_path,'train')
    assert not (tmp_path/'train').exists()


def cases():
    families=('arc','boolq','hellaswag','winogrande')
    return [dict(identity=f'dev{i}',family=families[i%4],split='development',choices=['a','b'],answer=0)
            for i in range(256)]


def development(items,correct):
    correct=set(correct)
    records=[dict(identity=c['identity'],family=c['family'],answer=0,
                  prediction=0 if i in correct else 1,logits=[2.,0.] if i in correct else [0.,2.])
             for i,c in enumerate(items)]
    metrics=dict(count=256,correct=len(correct),accuracy=len(correct)/256,loss=.37,kl=.1,ce=.3,
                 by_family={f:dict(count=64,correct=sum(i in correct and c['family']==f for i,c in enumerate(items)))
                            for f in sorted({c['family'] for c in items})})
    return dict(metrics=metrics,records=records)


def emit(root,phase,p,result):
    out=root/phase;out.mkdir(exist_ok=True)
    (out/'result.json').write_text(json.dumps(result))
    artifacts={x.name:joint.base.sha(x) for x in out.iterdir() if x.is_file() and x.name!='manifest.json'}
    uuid='GPU-8cf440df-fixture'
    manifest=dict(status='completed',phase=phase,fingerprint=p['fingerprint'],artifacts=artifacts,
        artifact_identity={n:joint.old.file_identity(out/n) for n in artifacts},
        dependencies={d:joint.base.sha(root/d/'manifest.json') for d in joint.DEPENDENCIES[phase]},
        allocation={uuid:5632},allocator={'uuid':uuid,'allocator_limit_bytes':5376*1024**2},peak_reserved_bytes=1000)
    (out/'manifest.json').write_text(json.dumps(manifest))
    return manifest


def evidence(root):
    rows=cases();native=development(rows,range(195))
    items=[dict(identity=f'train{i}',split='train') for i in range(4608)]+rows
    (root/'items.json').write_text(json.dumps({'items':items}))
    (root/'initializer.pt').write_bytes(b'untouched complete native-projected initializer')
    p=dict(phases=list(joint.PHASES),fingerprint='fixture',items_path=str(root/'items.json'),
           initializer=str(root/'initializer.pt'),initializer_sha256=joint.base.sha(root/'initializer.pt'),
           order=list(range(4608)),native_records=native['records'])
    initial=development(rows,range(160));fixed={'sha256':'fixed','tensor_bytes':12}
    masters=dict(blocks=[f'master{i}' for i in range(36)],norm='norm-master')
    def update(i):
        return dict(update=i,cases=16,indices=list(range((i-1)*16,i*16)),gradients_present_by_block=[True]*36,
                    learning_rate=5e-6,alpha=0.,feature_loss=0.,normalization_energy=None,
                    loss=.37,endpoint_loss=.37,kl=.1,ce=.3)
    pilot=dict(status='completed',passed=True,discarded=True,gradients_present_by_block=[True]*36,
               masters_changed_by_block=[True]*36,norm_master_changed=True,fixed_state_unchanged=True,
               initial=initial['metrics'],initial_records=initial['records'],initial_masters=masters,
               fixed_state=fixed,update=update(1))
    emit(root,'pilot',p,pilot)
    train=root/'train';train.mkdir();history=[]
    bad=set(range(195))-set(range(0,28,4))
    bad.update([i for i in range(195,256) if i%4!=0][:8])
    populations={0:set(range(160)),72:set(range(185)),144:bad,216:set(range(184)),288:set(range(185))}
    for step in joint.CHECKPOINTS:
        report=train/f'update_{step:03d}_development.json';value=development(rows,populations[step])
        report.write_text(json.dumps(value))
        checkpoint=Path(p['initializer']) if step==0 else train/f'update_{step:03d}.pt'
        if step:checkpoint.write_bytes(f'actual trained state at {step}'.encode())
        history.append(dict(updates=step,checkpoint=str(checkpoint),checkpoint_sha256=joint.base.sha(checkpoint),
            report=str(report),report_sha256=joint.base.sha(report),metrics=value['metrics'],
            eligibility=joint.broad.eligibility(value['records'],native['records'],rows)))
    trained=dict(status='completed',history=history,training=[update(i) for i in range(1,289)],
                 **joint.choose(history),initial_masters=masters,fixed_state=fixed,fixed_state_unchanged=True,
                 config=asdict(joint.training.CONFIG))
    emit(root,'train',p,trained)
    checks={}
    for c in (trained['candidate'],trained['diagnostic_best']):
        reference=joint.read(c['report'])
        checks[c['checkpoint_sha256']]=dict(passed=True,checkpoint_sha256=c['checkpoint_sha256'],
            strict_state_load=True,fixed_state_unchanged=True,fixed_state=fixed,**reference)
    verified=dict(status='completed',passed=True,checks=checks,confirmed=False)
    emit(root,'verify',p,verified)
    return p,pilot,trained,verified


def test_complete_chain_keeps_eligible_and_diagnostic_states_distinct(tmp_path):
    p,_,trained,verified=evidence(tmp_path)
    assert trained['candidate']['updates']==72 and trained['diagnostic_best']['updates']==144
    assert joint.checked(tmp_path,'verify',p)==verified
    assert len(verified['checks'])==2 and verified['confirmed']is False


@pytest.mark.parametrize('defect',['missing_check','records','metrics','fixed','checkpoint','strict','confirmation'])
def test_fresh_replay_requires_every_exact_checkpoint_and_record(tmp_path,defect):
    p,_,_,r=evidence(tmp_path);r=deepcopy(r);proof=next(iter(r['checks'].values()))
    if defect=='missing_check':r['checks'].pop(next(iter(r['checks'])))
    elif defect=='records':proof['records'][0]['prediction']=1
    elif defect=='metrics':proof['metrics']['ce']+=.1
    elif defect=='fixed':proof['fixed_state']={'sha256':'different'}
    elif defect=='checkpoint':proof['checkpoint_sha256']='different'
    elif defect=='strict':proof['strict_state_load']=False
    else:r['confirmed']=True
    emit(tmp_path,'verify',p,r)
    with pytest.raises(ValueError,match='Fresh'):joint.checked(tmp_path,'verify',p)


@pytest.mark.parametrize('defect',['order','feature_auxiliary','selection','report','weights','initializer'])
def test_training_rejects_wrong_recipe_selection_and_checkpoint_links(tmp_path,defect):
    p,pilot,r,_=evidence(tmp_path);r=deepcopy(r)
    if defect=='order':r['training'][0]['indices'].reverse()
    elif defect=='feature_auxiliary':r['training'][0]['feature_loss']=.2
    elif defect=='selection':r['candidate']=r['history'][-1]
    elif defect=='report':r['history'][1]['report_sha256']='wrong'
    elif defect=='weights':r['history'][1]['checkpoint_sha256']='wrong'
    else:r['initial_masters']['norm']='changed'
    emit(tmp_path,'train',p,r)
    with pytest.raises(ValueError):joint.checked(tmp_path,'train',p)


def test_receipt_requires_result_hash_exact_lineage_and_memory_cap(tmp_path):
    p,pilot,_,_=evidence(tmp_path)
    path=tmp_path/'pilot/manifest.json';original=joint.read(path)
    for alter in ('missing_result','extra_identity'):
        m=deepcopy(original)
        if alter=='missing_result':m['artifacts'].pop('result.json');m['artifact_identity'].pop('result.json')
        else:m['artifact_identity']['unknown.json']={}
        path.write_text(json.dumps(m))
        with pytest.raises(ValueError):joint.checked(tmp_path,'pilot',p)
    path.write_text(json.dumps(original))
    pilot['systems_only']=True;emit(tmp_path,'pilot',p,pilot)
    with pytest.raises(ValueError,match='lineage'):joint.checked(tmp_path,'train',p)


def test_cpu_strict_builder_forwards_guard_to_runtime_rotary_and_preserves_bf16(monkeypatch):
    from test_jspace_3b_joint_train import components
    stack,norm,E,rotary,_,_,_=components()
    state=dict(stack=deepcopy(stack.state_dict()),norm=deepcopy(norm.state_dict()))
    p=dict(model={'path':'fixture'},fit_map={str(i):'fit' for i in range(36)},
           geometry_readout_fingerprint=joint.geometry(stack,E))
    monkeypatch.setattr(joint.cache.builder,'load_previous',lambda _:state)
    monkeypatch.setattr(joint.cache.builder,'build_stack',lambda *a,**k:(stack,norm,E,rotary,{'passed':True}))
    monkeypatch.setattr(joint.cache.builder.native,'_config',lambda *_:object())
    events=[]
    def guard():events.append('guard')
    def make_rotary(config,device,*,guard=None):
        assert guard is not None;guard();events.append('rotary');return rotary
    monkeypatch.setattr(joint.generation,'training_rotary',make_rotary)
    actual=joint.build(p,'checkpoint','cpu',guard)
    assert actual[0]is stack and events[-1]=='rotary'
    assert all(v.dtype==torch.bfloat16 for v in (*stack.parameters(),*norm.parameters()))
    assert joint.geometry(stack,E)==p['geometry_readout_fingerprint']
