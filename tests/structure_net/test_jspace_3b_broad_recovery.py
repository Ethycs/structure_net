"""Paired recovery eligibility, real feature gradients and retained quarter states."""
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.structure_net import jspace_3b_broad_recovery as study
from test_jspace_3b_sanity_evaluate import tiny


FAMILIES=('arc','boolq','hellaswag','winogrande')


def test_model_device_uses_explicit_cuda_target_and_restores_caller(monkeypatch):
    current=['cuda:1'];seen=[]
    @contextmanager
    def device_context(device):
        seen.append(str(device));previous=current[0];current[0]=str(device)
        try:yield
        finally:current[0]=previous
    monkeypatch.setattr(torch.cuda,'device',device_context)
    with study._model_device('cpu'):
        assert current[0]=='cuda:1' and not seen
    with pytest.raises(RuntimeError,match='kernel failure'):
        with study._model_device('cuda:0'):
            assert current[0]=='cuda:0'
            raise RuntimeError('kernel failure')
    assert seen==['cuda:0'] and current[0]=='cuda:1'


@pytest.mark.parametrize('kernel_error',[False,True])
@torch.inference_mode()
def test_packed_score_scopes_readout_device_after_guard_even_if_tail_was_current(tiny,monkeypatch,kernel_error):
    from test_jspace_3b_sanity_evaluate import arguments
    native,stack,norm,items,labels=tiny;case=items[-1]
    _,hidden=study.cache.sanity.native_case(native,case,labels,2)
    batch=dict(hidden=hidden,**arguments(hidden,native.model.rotary_emb),
        loss_positions=torch.tensor([[0,1,2,len(case['input_ids'])-1]]),
        targets=torch.tensor([[1,2,3,labels[case['answer']]]]),teacher=torch.zeros(1,4,len(native.lm_head.weight)))
    data=SimpleNamespace(development=[0],items=[case],batch=lambda *_a,**_k:batch)
    current=['tail'];events=[]
    def guard():current[0]='tail';events.append('guard')
    @contextmanager
    def model_device(device):
        assert events[-1]=='guard'
        previous=current[0];current[0]=str(device);events.append(('enter',str(device)))
        try:yield
        finally:current[0]=previous;events.append('exit')
    def entering_stack(_module,_args):
        assert current[0]==str(native.lm_head.weight.device)
        events.append('kernel')
        if kernel_error:raise RuntimeError('kernel failure')
    monkeypatch.setattr(study,'_model_device',model_device)
    handle=stack.register_forward_pre_hook(entering_stack)
    try:
        if kernel_error:
            with pytest.raises(RuntimeError,match='kernel failure'):
                study.score_packed(stack,norm,native.lm_head.weight,native.model.rotary_emb,data,labels,guard)
        else:
            _,records=study.score_packed(stack,norm,native.lm_head.weight,native.model.rotary_emb,data,labels,guard)
            assert len(records)==1
    finally:handle.remove()
    assert events==['guard',('enter','cpu'),'kernel','exit'] and current[0]=='tail'


def test_run_restores_front_after_both_allocator_resets_before_model_work(tmp_path,monkeypatch):
    events=[];plan=dict(fingerprint='plan')
    monkeypatch.setattr(study,'verify',lambda *_:plan)
    monkeypatch.setattr(study,'wait_ready',lambda *_:events.append('lease') or dict(id='lease',allocation={}))
    settings=dict(front=dict(device=1),tail=dict(device=0))  # Reversed visible order is intentional.
    monkeypatch.setattr(study.source.replay,'allocator_settings',lambda *_:events.append('allocator') or settings)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda d:events.append(('reset',d)))
    monkeypatch.setattr(torch.cuda,'set_device',lambda d:events.append(('current',d)))
    monkeypatch.setattr(torch,'set_num_threads',lambda *_:None)
    monkeypatch.setattr(torch,'manual_seed',lambda *_:None)
    def stop_at_dataset(*_):
        assert events==['lease','allocator',('reset',1),('reset',0),('current',1)]
        raise RuntimeError('stop before model allocation')
    monkeypatch.setattr(study,'dataset_for',stop_at_dataset)
    with pytest.raises(RuntimeError,match='stop before model allocation'):
        study.run(tmp_path,'parent_legacy','pilot')


def cases(count=256):
    return [dict(identity=f'dev{i}',family=FAMILIES[i%4],answer=0,choices=['a','b'],
        input_ids=[1,2,3],split='development') for i in range(count)]


def rows_for(items,correct):
    correct=set(correct)
    return [dict(identity=c['identity'],family=c['family'],answer=0,prediction=0 if i in correct else 1,
        logits=[2.,0.] if i in correct else [0.,2.]) for i,c in enumerate(items)]


def candidate(arm,updates,correct=200,passed=True,sha=None):
    return dict(arm=arm,updates=updates,metrics=dict(count=256,correct=correct,ce=100.),
        eligibility=dict(passed=passed),checkpoint_sha256=sha or f'{arm}-{updates}')


def test_eligibility_uses_paired_overall_and_family_accuracy_not_ce():
    items=cases();native=rows_for(items,range(256))
    # 12/256 and 6/64 are inside the declared 5pp / 10pp allowances.
    misses={0,4,8,12,16,20,1,5,9,13,17,21}
    good=study.eligibility(rows_for(items,set(range(256))-misses),native,items)
    assert good['passed'] and good['overall_passed'] and all(good['family_passed'].values())
    family_failure=study.eligibility(rows_for(items,set(range(256))-set(range(0,28,4))),native,items)
    assert family_failure['overall_passed'] and not family_failure['passed']
    assert family_failure['family_passed']['arc'] is False
    overall_failure=study.eligibility(rows_for(items,range(13,256)),native,items)
    assert not overall_failure['overall_passed'] and all(overall_failure['family_passed'].values())


def test_ineligible_seed_is_not_a_veto_on_later_recovery_and_diagnostic_stays_separate():
    seed=candidate('parent_legacy',0,201,False)
    recovered=candidate('parent_legacy',72,200,True)
    result=study.choose([seed,recovered])
    assert result['candidate']==recovered and result['development_passed']
    assert result['diagnostic_best']==seed
    failed=study.choose([seed,candidate('parent_legacy',72,199,False)])
    assert failed['candidate']==seed and not failed['development_passed']


def test_accuracy_then_earliest_then_control_arm_and_duplicate_seed_deduplication():
    seed=candidate('parent_legacy',0,200,True,'seed')
    duplicate=candidate('native_content',0,200,True,'seed')
    later=candidate('parent_legacy',72,200,True);later['metrics']['ce']=.001
    assert study.choose([later,duplicate,seed])['candidate']==seed
    best=candidate('native_content',144,201,True)
    assert study.choose([seed,best])['candidate']==best
    assert study.global_candidates([duplicate,later,seed])==[seed,later]


def marker_trainer():
    stack=nn.Module();stack.mark=nn.Parameter(torch.tensor(0.));stack.register_buffer('fixed',torch.tensor(7.))
    norm=nn.Linear(1,1,bias=False);dev=cases()
    data=SimpleNamespace(training=list(range(4608)),development=list(range(4608,4864)),
        items=[dict(identity=f'train{i}',split='train') for i in range(4608)]+dev)
    trainer=SimpleNamespace(stack=stack,norm=norm,readout=torch.tensor([2.]),rotary=nn.Identity(),
        dataset=data,config=study.CONFIG,updates=0,finished=False,alpha=study.cache.ALPHA,
        feature_targets=SimpleNamespace(energy=2.))
    # Early checkpoint 72 and later144 tie; 72 must win regardless better later CE.
    counts={0:236,72:248,144:248,216:246,288:245}
    def score():
        n=counts[trainer.updates];records=rows_for(dev,range(n))
        return dict(count=256,correct=n,accuracy=n/256,loss=3.,kl=.4,ce=10./(trainer.updates+1)),records
    def update(indices,learning_rate):
        trainer.updates+=1
        with torch.no_grad():stack.mark.fill_(trainer.updates)
        return dict(update=trainer.updates,cases=len(indices),learning_rate=learning_rate,
            loss=1.,endpoint_loss=.8,kl=.1,ce=.2,feature_loss=.3,alpha=trainer.alpha,
            normalization_energy=2.,gradients_present_by_block=[True]*14)
    trainer.packed_development=score;trainer.train_window=update
    trainer.snapshot=lambda:dict(stack=deepcopy(stack.state_dict()),norm=deepcopy(norm.state_dict()))
    return trainer,dev


def trial_fixture(tmp_path,monkeypatch):
    trainer,dev=marker_trainer();seed=tmp_path/'seed.pt';study.old.save(seed,trainer.snapshot())
    monkeypatch.setattr(study.cache,'SEED_SHA',study.base.sha(seed))
    metrics,records=trainer.packed_development();native=rows_for(dev,range(256))
    order=list(reversed(trainer.dataset.training))
    result=study.fit_trial(trainer,tmp_path/'train',arm='parent_legacy',order=order,native_records=native,
        seed_checkpoint=seed,pilot_reference=dict(initial=metrics,initial_records=records))
    cache=tmp_path/'cache';cache.mkdir();study.base.write(cache/'items.json',dict(items=trainer.dataset.items))
    plan=dict(cache_root=str(cache),config=asdict(study.CONFIG),checkpoint=str(seed),checkpoints=list(study.CHECKPOINT_UPDATES),order=order,
        feature_receipt=dict(normalization_energy=2.),native_records=native)
    return result,plan,trainer


def test_complete_288_updates_retains_actual_quarter_weights_and_early_task_winner(tmp_path,monkeypatch):
    result,plan,trainer=trial_fixture(tmp_path,monkeypatch)
    assert result['updates']==288 and trainer.finished
    assert [r['updates'] for r in result['history']]==[0,72,144,216,288]
    assert [i for update in result['training'] for i in update['indices']]==plan['order']
    assert {u['learning_rate'] for u in result['training']}=={5e-6}
    assert result['candidate']['updates']==72 and result['candidate']['metrics']['correct']==248
    assert result['history'][0]['eligibility']['passed'] is False
    for c in result['history']:
        saved=torch.load(c['checkpoint'],weights_only=True)
        assert saved['stack']['mark'].item()==c['updates'] and saved['stack']['fixed']==7
    assert len({r['checkpoint_sha256'] for r in result['history']})==5
    study.validate_history(result,plan)


@pytest.mark.parametrize('defect',['order','budget','lr','gradient','selection','metric','weights'])
def test_completed_history_rejects_partial_or_misrepresented_evidence(tmp_path,monkeypatch,defect):
    result,plan,_=trial_fixture(tmp_path,monkeypatch);bad=deepcopy(result)
    if defect=='order':bad['training'][0]['indices'][0]=bad['training'][0]['indices'][1]
    elif defect=='budget':bad['training'].pop()
    elif defect=='lr':bad['training'][50]['learning_rate']=1e-4
    elif defect=='gradient':bad['training'][50]['gradients_present_by_block'][0]=False
    elif defect=='selection':bad['candidate']=bad['history'][2]
    elif defect=='metric':bad['history'][1]['metrics']['correct']+=1
    else:Path(bad['history'][1]['checkpoint']).write_bytes(b'substituted checkpoint')
    with pytest.raises(ValueError):study.validate_history(bad,plan)


def test_fit_checks_initializer_before_any_scientific_update(tmp_path):
    trainer,dev=marker_trainer();seed=tmp_path/'seed.pt';study.old.save(seed,trainer.snapshot())
    metrics,records=trainer.packed_development();metrics['correct']-=1
    with pytest.raises(ValueError,match='initializer'):
        study.fit_trial(trainer,tmp_path/'train',arm='parent_legacy',order=trainer.dataset.training,
            native_records=rows_for(dev,range(256)),seed_checkpoint=seed,
            pilot_reference=dict(initial=metrics,initial_records=records))
    assert trainer.updates==0


def test_real_bf16_resident_feature_gradient_reaches_all_blocks_and_norm(tmp_path):
    from test_jspace_3b_feature_recovery import trainer as make_reference
    reference=make_reference()
    model=study.BroadTrainer(reference.stack,reference.norm,reference.readout,reference.rotary,reference.dataset,[0,1],
        targets=reference.feature_targets,alpha=study.cache.ALPHA,config=replace(reference.config,learning_rate=5e-6),
        native_correct=2,parent_correct=2,count=2,resident_device='cpu',resident_budget_bytes=1024)
    before=model.snapshot();fixed=study.source.old.fixed_state(model.stack,model.readout,model.rotary)
    batch=model.dataset.batch([0,1,2,3]);endpoint,kl,ce,feature=model.objectives(batch)
    feature.backward()
    assert all(any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in block.parameters()) for block in model.stack.blocks)
    assert model.readout.grad is None and all(b.grad is None for b in model.stack.buffers())
    model.optimizer.zero_grad(set_to_none=True)
    update=model.train_window([0,1,2,3],learning_rate=5e-6)
    assert update['gradients_present_by_block']==[True]*3 and update['update']==1
    assert update['loss']==pytest.approx(update['endpoint_loss']+study.cache.ALPHA*update['feature_loss'],rel=1e-6)
    assert study.source.old.fixed_state(model.stack,model.readout,model.rotary)==fixed
    assert any(not torch.equal(v,model.snapshot()['stack'][k]) for k,v in before['stack'].items() if torch.is_tensor(v))
    assert model.optimizer.state[next(model.norm.parameters())]['step']==1
    with pytest.raises(ValueError,match='training'):model.train_window([32])


@torch.inference_mode()
def test_packed_development_uses_singleton_choice_readout_and_recovered_norm(tiny,tmp_path):
    from test_jspace_3b_broad_recovery_cache import items
    import numpy as np
    native,stack,norm,_,labels=tiny;rows=items()
    for r in rows:r['split']='development'
    # Preserve one TRAIN row to exercise the real streaming dataset's contract.
    rows[0]['split']='train';boundaries=[];teachers=[]
    for c in rows:
        h,t,_=study.cache.native_outputs(native,c,labels,cuts=(2,))
        boundaries.append(h[2][0]);teachers.append(t)
    for name,value in [('h',torch.cat(boundaries)),('t',torch.stack(teachers))]:
        np.save(tmp_path/f'{name}.npy',value.contiguous().view(torch.int16).numpy())
    data=study.cache.RecoveryDataset(rows,tmp_path/'h.npy',tmp_path/'t.npy',arm='native_content',label_ids=labels)
    try:
        metrics,records=study.score_packed(stack,norm,native.lm_head.weight,native.model.rotary_emb,data,labels)
        expected=study.cache.sanity.compact_case(stack,norm,native.lm_head.weight,native.model.rotary_emb,boundaries[1][None],rows[1],labels)
        assert records==[expected] and metrics['count']==1
        assert metrics['correct']==int(expected['prediction']==expected['answer'])
        assert all(torch.isfinite(torch.tensor(metrics[k])) for k in ('loss','kl','ce'))
    finally:data.close()


def test_matched_arms_require_identical_initial_records_masters_and_fixed_target():
    value=dict(initial_records=[{'identity':'same'}],checkpoint_sha256=study.cache.SEED_SHA,
        alpha=study.cache.ALPHA,normalization_energy=2.,
        probe=dict(fp32_master_before=['master'],norm_fp32_master_before='norm',fixed_state='fixed'))
    pilots={arm:deepcopy(value) for arm in study.cache.ARMS}
    assert study.matched_initializers(pilots)
    pilots['native_content']['probe']['fp32_master_before']=['different']
    with pytest.raises(ValueError,match='identical'):study.matched_initializers(pilots)


def test_incomplete_pilot_blocks_training_before_reservation_or_dataset(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'verify',lambda *_:{'fingerprint':'p'})
    def denied(*_):raise ValueError('pilot incomplete')
    monkeypatch.setattr(study,'checked',denied)
    monkeypatch.setattr(study,'wait_ready',lambda *_:pytest.fail('lease before completed pilot'))
    monkeypatch.setattr(study,'dataset_for',lambda *_:pytest.fail('dataset before completed pilot'))
    with pytest.raises(ValueError,match='pilot incomplete'):study.run(tmp_path,'parent_legacy','train')


@pytest.mark.parametrize('defect',['none','fresh_rows','partial','cap'])
def test_fresh_receipt_authenticates_actual_retained_selection_and_reserved_caps(tmp_path,monkeypatch,defect):
    arm='parent_legacy';root=tmp_path/arm;root.mkdir()
    trained,plan,_=trial_fixture(root,monkeypatch);plan['fingerprint']='plan'
    initial=trained['history'][0];initial_report=study.read(initial['report'])
    pilot=dict(status='completed',arm=arm,passed=True,discarded=True,
        probe=dict(passed=True,layer_indices=list(range(22,36)),fixed_state_unchanged=True,fixed_state=trained['fixed_state']),
        geometry_readout_unchanged=True,alpha=study.cache.ALPHA,normalization_energy=2.,
        checkpoint_sha256=study.cache.SEED_SHA,initial=initial_report['metrics'],initial_records=initial_report['records'])
    checks={}
    for c in (trained['candidate'],trained['diagnostic_best']):
        report=study.read(c['report']);digest=c['checkpoint_sha256']
        checks[digest]=dict(checkpoint_sha256=digest,strict_state_load=True,fixed_state_unchanged=True,
            fixed_state=trained['fixed_state'],records=report['records'],metrics=report['metrics'])
    fresh=dict(status='completed',arm=arm,passed=True,checks=checks)
    if defect=='fresh_rows':next(iter(checks.values()))['records'][0]['logits'][0]+=1
    if defect=='partial':checks.clear()
    for phase,result in [('pilot',pilot),('train',trained),('verify',fresh)]:
        out=root/phase;out.mkdir(exist_ok=True)
        study.base.write(out/'result.json',result)
        if phase!='verify':
            allocation={'GPU-0370fdb0-fixture':11264,'GPU-7b1db3cb-fixture':7168}
            allocator={role:dict(uuid=uuid,allocator_limit_bytes=(mib-256)*1024**2)
                for role,(uuid,mib) in zip(('front','tail'),allocation.items())}
            peaks=dict(front=1024,tail=1024)
        else:
            allocation={'GPU-0370fdb0-fixture':11264}
            allocator=dict(physical_uuid='GPU-0370fdb0-fixture',allocator_limit_bytes=11008*1024**2)
            peaks=1024
            if defect=='cap':allocator['allocator_limit_bytes']+=1
        manifest=dict(status='completed',arm=arm,phase=phase,fingerprint='plan',allocation=allocation,
            allocator=allocator,peak_reserved_bytes=peaks,
            artifacts={str(p.relative_to(out)):study.base.sha(p) for p in out.rglob('*') if p.is_file()})
        study.base.write(out/'manifest.json',manifest)
    if defect=='none':assert study.checked(tmp_path,arm,'verify',plan)==fresh
    else:
        with pytest.raises(ValueError):study.checked(tmp_path,arm,'verify',plan)
