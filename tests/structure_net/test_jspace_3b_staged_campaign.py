"""New staged lifecycle, immutable reuse, and real singleton-native seam controls."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM

from experiments.structure_net import jspace_3b_staged_campaign as study
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def stage(cut=26):
    value=dict(schema=study.SCHEMA,root_fingerprint='campaign',cut=cut,parent_cut=cut+1,
        parent_correct=52,teacher_fingerprint='teacher',layers=list(range(cut,36)),calibration_reuse=None)
    value['fingerprint']=study.base.digest(value)
    return value


def quality(correct=47,ce=1.,kl=.1):
    rows=[dict(identity=f'case{i}',family='f',answer=0,prediction=int(i>=correct),
        logits=[1.,0.] if i<correct else [0.,1.]) for i in range(64)]
    return dict(metrics=dict(correct=correct,count=64,accuracy=correct/64,ce=ce,original_kl=kl),records=rows)


def write_phase(directory,phase,result,current,files=None):
    path=directory/phase;path.mkdir(exist_ok=True)
    for name,value in (files or {}).items():
        target=path/name;target.parent.mkdir(parents=True,exist_ok=True)
        if name.endswith('.pt'):torch.save(value,target)
        else:write(target,value)
    write(path/'result.json',result)
    manifest=dict(status='completed',fingerprint=current['fingerprint'],
        artifacts={str(p.relative_to(path)):study.base.sha(p) for p in path.rglob('*') if p.is_file() and p.name!='manifest.json'})
    write(path/'manifest.json',manifest)
    return result


def reseal(path):
    manifest=study.read(path/'manifest.json')
    manifest['artifacts']={str(p.relative_to(path)):study.base.sha(p) for p in path.rglob('*') if p.is_file() and p.name!='manifest.json'}
    write(path/'manifest.json',manifest)


def pilot(directory,current):
    raw={'stack':{'blocks.0.weight':torch.ones(2,dtype=torch.bfloat16),
                  'blocks.0.basis':torch.eye(2,dtype=torch.bfloat16),
                  'blocks.0._extra_state':{'packed':False}},'norm':{'weight':torch.ones(2,dtype=torch.bfloat16)}}
    initial=quality();path=directory/'pilot';path.mkdir()
    torch.save(raw,path/'initializer.pt');write(path/'initial_development.json',initial)
    result=dict(status='completed',passed=True,systems_only=True,discarded=True,layer_indices=current['layers'],
        update={'update':1},fp32_master_changed_by_block=[True]*len(current['layers']),norm_fp32_master_changed=True,
        fixed_state_unchanged=True,checkpoint_strict_load=True,inheritance={'passed':True},
        entrance_probe=dict(passed=True,discarded=True,systems_only=True,update={'update':1},fixed_state_unchanged=True,
            fp32_master_changed_by_parameter={'weight':True}),entrance_restore_exact=True,
        initializer_sha256=study.base.sha(path/'initializer.pt'),initial_development_sha256=study.base.sha(path/'initial_development.json'),
        entrance_parameter_names=['weight'],entrance_buffer_names=['basis'],initial=initial['metrics'],initial_records=initial['records'])
    write_phase(directory,'pilot',result,current)
    return raw,result


def training(directory,current,*,best=2,best_correct=48):
    raw,p=pilot(directory,current);path=directory/'train';path.mkdir()
    history=[]
    for epoch in range(-1,13):
        report=quality(best_correct,.9,.09) if epoch==best and epoch>=0 else quality()
        filename=f'epoch_{epoch:02d}_development.json';write(path/filename,report)
        updates=[] if epoch<=0 else [dict(update=(epoch-1)*32+j+1,gradients_present_by_block=[True]*len(current['layers'])) for j in range(32)]
        history.append(dict(epoch=epoch,metrics=report['metrics'],training=updates,
            eligibility=study.training.candidate_eligibility(report['metrics'],p['initial']),
            development_sha256=study.base.sha(path/filename),checkpoint_updated=epoch in [-1,best]))
    torch.save(raw,path/'selected.pt');selected_sha=study.base.sha(path/'selected.pt')
    winner=study.training.best_eligible(history,p['initial']);winner['checkpoint_sha256']=selected_sha
    # The history records the protected snapshot even when another epoch wins.
    history[0].setdefault('checkpoint_sha256',study.base.sha(directory/'pilot/initializer.pt'))
    replay=dict(protected_replay_exact=True,warm_restore_exact=True,optimizer_untouched=True,
        warm_state_fingerprint_before={'sha':'same'},warm_state_fingerprint_after={'sha':'same'})
    write(path/'protected_replay.json',replay)
    gate=study.training.inherited.gates(winner['metrics']['correct'],native_correct=48,parent_correct=52,count=64)
    result=dict(status='completed',config=asdict(study.CONFIG),completed_epochs=12,updates=384,layer_indices=current['layers'],
        strict_reload_exact=True,fixed_state_unchanged=True,protected_epoch=-1,initial=p['initial'],warm_initial=history[1]['metrics'],
        selection_policy=study.training.SELECTION,eligibility_tolerance=study.training.TOLERANCE,history=history,
        protected_replay=replay,protected_reference=dict(metrics=p['initial'],development_sha256=history[0]['development_sha256'],checkpoint_sha256=history[0]['checkpoint_sha256']),selected_epoch=winner['epoch'],selected=winner['metrics'],selected_checkpoint_sha256=selected_sha,
        quality=gate,quality_pass=gate['passed'])
    write_phase(directory,'train',result,current)
    return result


def test_protected_raw_history_and_cautious_winner_are_authenticated(tmp_path):
    current=stage();result=training(tmp_path,current)
    assert study.checked(tmp_path,'train',current)['selected_epoch']==2
    changed=deepcopy(result);changed['history'][3]['metrics']['ce']=1.2
    changed['history'][3]['eligibility']=study.training.candidate_eligibility(changed['history'][3]['metrics'],changed['initial'])
    changed['selected']=changed['history'][3]['metrics']
    write(tmp_path/'train/epoch_02_development.json',dict(metrics=changed['selected'],records=quality(48)['records']))
    changed['history'][3]['development_sha256']=study.base.sha(tmp_path/'train/epoch_02_development.json')
    write(tmp_path/'train/result.json',changed);reseal(tmp_path/'train')
    with pytest.raises(ValueError,match='Eligible accuracy-first'):study.checked(tmp_path,'train',current)


def test_raw_minus_one_is_a_real_selected_candidate_and_fresh_verified(tmp_path):
    current=stage();result=training(tmp_path,current,best=-1,best_correct=47)
    # Make protected report match authenticated pilot exactly; it stays earliest.
    assert study.checked(tmp_path,'train',current)['selected_epoch']==-1
    verify=dict(status='completed',passed=True,comparison={'passed':True},fixed_state_unchanged=True,
        metrics=result['selected'],selected_checkpoint_sha256=result['selected_checkpoint_sha256'],quality=result['quality'],compressed_blocks=10)
    write_phase(tmp_path,'verify',verify,current)
    assert study.checked(tmp_path,'verify',current)['passed']
    (tmp_path/'pilot/initializer.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):study.checked(tmp_path,'verify',current)


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(entrance_restore_exact=False),
    lambda r:r['entrance_probe']['fp32_master_changed_by_parameter'].update(weight=False),
    lambda r:r['fp32_master_changed_by_block'].__setitem__(1,False),
])
def test_both_pilots_must_pass_before_protected_state_is_usable(tmp_path,mutation):
    current=stage();_,value=pilot(tmp_path,current);mutation(value)
    write(tmp_path/'pilot/result.json',value);reseal(tmp_path/'pilot')
    with pytest.raises(ValueError,match='pilot validity'):study.checked(tmp_path,'pilot',current)


def test_delta_changes_only_existing_entrance_parameters(tmp_path):
    current=stage();raw,p=pilot(tmp_path,current)
    delta=dict(block={k.removeprefix('blocks.0.'):deepcopy(v) for k,v in raw['stack'].items()},
        layer_idx=26,normalization={'floor':.01},config=asdict(study.ENTRANCE_CONFIG))
    delta['block']['weight'].add_(.125)
    assert study.validate_delta(raw,delta,p,26)['fixed_geometry_exact']
    delta['block']['basis'][0,0]=0
    with pytest.raises(ValueError,match='frozen entrance geometry'):study.validate_delta(raw,delta,p,26)
    delta['block']['basis']=raw['stack']['blocks.0.basis'].clone();delta['norm']={}
    with pytest.raises(ValueError,match='only the selected entrance'):study.validate_delta(raw,delta,p,26)


def test_prealign_min_feature_selection_and_frozen_export_are_checked(tmp_path):
    current=stage();raw,p=pilot(tmp_path,current);path=tmp_path/'prealign';path.mkdir();history=[]
    for epoch in range(13):
        metrics={'loss':1/(epoch+1)};file=path/f'epoch_{epoch:02d}_development.json'
        write(file,dict(metrics=metrics,quality_diagnostic=None))
        updates=[] if not epoch else [dict(update=(epoch-1)*32+j+1,only_entrance_gradients=True) for j in range(32)]
        history.append(dict(epoch=epoch,metrics=metrics,training=updates,development_sha256=study.base.sha(file)))
    normalization=dict(floor=.01,train_case_count=512)
    delta=dict(block={k.removeprefix('blocks.0.'):deepcopy(v) for k,v in raw['stack'].items()},
        layer_idx=26,normalization=normalization,config=asdict(study.ENTRANCE_CONFIG))
    torch.save(delta,path/'selected_entrance.pt');digest=study.base.sha(path/'selected_entrance.pt');history[-1]['checkpoint_sha256']=digest
    proof=study.validate_delta(raw,delta,p,26);write(path/'delta_validation.json',proof)
    result=dict(status='completed',config=asdict(study.ENTRANCE_CONFIG),completed_epochs=12,updates=384,
        layer_idx=26,inherited_layers=current['layers'][1:],history=history,strict_reload_exact=True,
        fixed_state_unchanged=True,task_quality_guaranteed=False,selected_epoch=12,selected=history[-1]['metrics'],
        initial=history[0]['metrics'],selected_checkpoint_sha256=digest,normalization=normalization)
    write_phase(tmp_path,'prealign',result,current)
    assert study.checked(tmp_path,'prealign',current)['selected_epoch']==12
    result['selected_epoch']=0;write(path/'result.json',result);reseal(path)
    with pytest.raises(ValueError,match='Feature-selected'):study.checked(tmp_path,'prealign',current)


def test_reuse_is_independent_bytes_with_original_provenance_and_no_mutation(tmp_path):
    old=tmp_path/'old';old.mkdir();(old/'bank.npy').write_bytes(b'original-bank')
    write(old/'result.json',dict(status='completed',teacher_fingerprint='teacher'))
    artifacts={p.name:study.base.sha(p) for p in old.iterdir()}
    write(old/'manifest.json',dict(status='completed',fingerprint='old-fingerprint',artifacts=artifacts))
    reuse=dict(directory=str(old),source_fingerprint='old-fingerprint',manifest_sha256=study.base.sha(old/'manifest.json'),artifacts=artifacts)
    before={p.name:(study.old.file_identity(p),study.base.sha(p)) for p in old.iterdir()}
    result=study.copy_reused_phase(reuse,tmp_path/'new')
    assert result['teacher_fingerprint']=='teacher'
    assert study.read(tmp_path/'new/reuse.json')==reuse
    assert study.base.sha(tmp_path/'new/original_manifest.json')==reuse['manifest_sha256']
    for name,(identity,digest) in before.items():
        assert study.old.file_identity(old/name)==identity and study.base.sha(old/name)==digest
    assert (old/'bank.npy').stat().st_ino!=(tmp_path/'new/bank.npy').stat().st_ino
    with pytest.raises(FileExistsError):study.copy_reused_phase(reuse,tmp_path/'new')
    (old/'bank.npy').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Reused artifact changed'):study.copy_reused_phase(reuse,tmp_path/'other')


def test_import_closure_follows_dependencies_without_a_source_glob(tmp_path,monkeypatch):
    monkeypatch.setattr(study.base,'ROOT',tmp_path)
    (tmp_path/'a.py').write_text('from . import b\n')
    (tmp_path/'b.py').write_text('from .c import value\n')
    (tmp_path/'c.py').write_text('value=3\n')
    (tmp_path/'unrelated.py').write_text('raise RuntimeError()\n')
    paths=study.local_import_closure([tmp_path/'a.py'])
    assert {p.name for p in paths}=={'a.py','b.py','c.py'}


def test_campaign_reserved_six_phases_and_incomplete_phase_cannot_retry(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');calls=[]
    monkeypatch.setattr(study,'verify_root',lambda r:{'fingerprint':'campaign'})
    def prepare(root,cut):
        path=study.stage_dir(root,cut);path.mkdir(parents=True);write(path/'plan.json',stage(cut))
    monkeypatch.setattr(study,'prepare_stage',prepare)
    monkeypatch.setattr(study,'verify_stage',lambda root,cut:({},study.read(study.stage_dir(root,cut)/'plan.json')))
    def checked(directory,phase,current):
        assert (directory/phase/'manifest.json').exists()
        return dict(quality_pass=True,selected={'correct':47},quality={},storage={})
    monkeypatch.setattr(study,'checked',checked)
    def run(argv,**kwargs):
        calls.append(argv);phase=argv[argv.index('-m')+2];cut=int(argv[argv.index('--cut')+1])
        output=study.stage_dir(tmp_path,cut)/phase;output.mkdir();(output/'manifest.json').write_text('{}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(study.subprocess,'run',run)
    outcome=study.campaign(tmp_path,stop_cut=26)
    assert outcome['compressed_blocks']==10 and outcome['status']=='requested_prefix_completed'
    assert [c[c.index('-m')+2] for c in calls]==list(study.PHASES)
    for command in calls:
        assert command[:6]==['/usr/bin/python3','/usr/local/bin/inference-ctl','lease','run','--allocation','GPU-0370fdb0=11264']
        assert command[command.index('--minutes')+1]=='120' and command[command.index('-m')+1]==study.MODULE
    prepare(tmp_path,25);(study.stage_dir(tmp_path,25)/'capture').mkdir();calls.clear()
    with pytest.raises(RuntimeError,match='Incomplete phase'):study.campaign(tmp_path,stop_cut=25)
    assert calls==[]


def test_run_denied_lease_before_cuda_or_initializer(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_full_lease
    current=stage();directory=study.stage_dir(tmp_path,26);directory.mkdir(parents=True)
    monkeypatch.setattr(study,'verify_stage',lambda *a:({},current))
    monkeypatch.setattr(study,'checked',lambda *a:{})
    calls=[]
    def denied(*a):raise RuntimeError('lease denied')
    monkeypatch.setattr(jspace_full_lease,'wait_ready',denied)
    monkeypatch.setattr(study.torch.cuda,'init',lambda:calls.append('cuda'))
    monkeypatch.setattr(study,'make_stack',lambda *a,**k:calls.append('model'))
    with pytest.raises(RuntimeError,match='lease denied'):study.run_phase(tmp_path,26,'pilot')
    assert not calls and not (directory/'pilot/initializer.pt').exists()
    assert study.read(directory/'pilot/failure.json')['failure_kind']=='execution_failure'


def test_actual_bf16_singleton_boundary_then_parent_batch4_replays_cache(tmp_path):
    torch.manual_seed(78)
    cfg=Qwen2Config(hidden_size=8,intermediate_size=12,num_attention_heads=2,num_key_value_heads=1,
        num_hidden_layers=3,vocab_size=16,max_position_embeddings=32,tie_word_embeddings=True,attention_dropout=0.)
    cfg._attn_implementation='eager';native=Qwen2ForCausalLM(cfg).bfloat16().eval().requires_grad_(False)
    blocks=[AffineQwenBlock(native.model.layers[i],torch.linalg.qr(torch.randn(8,8)).Q[:,:4],torch.zeros(1,8),trainable=False) for i in (1,2)]
    parent=CompactQwenStack(blocks,geometry_mode='bf16',pad_to_multiple=128).eval()
    parent.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16).requires_grad_(False)
    items=[];before=[];after=[];offset=0;rotary=native.model.rotary_emb
    for i,length in enumerate((3,6,4,5,3)):
        hidden=torch.randn(1,length,8).bfloat16();positions=torch.arange(length)[None]
        mask=torch.zeros(1,1,length,length,dtype=torch.bfloat16).masked_fill(
            ~(torch.arange(length)[:,None]>=torch.arange(length)[None,:]),torch.finfo(torch.bfloat16).min)
        with torch.inference_mode():target=native.model.layers[0](hidden,attention_mask=mask,position_ids=positions,
            cache_position=positions[0],position_embeddings=rotary(hidden,positions),use_cache=False)[0]
        before.append(hidden[0]);after.append(target[0]);items.append(dict(identity=f'case{i}',family='f',answer=0,choices=['a','b'],
            split='train' if i==0 else 'development',input_ids=[1]*length,offset=offset,stop=offset+length,
            loss_positions=[0,(length-2)//2,length-2,length-1],targets=[0]*4));offset+=length
    np.save(tmp_path/'before.npy',torch.cat(before).view(torch.int16).numpy());np.save(tmp_path/'after.npy',torch.cat(after).view(torch.int16).numpy())
    np.save(tmp_path/'teacher.npy',np.zeros((5,4,16),dtype=np.int16))
    data=study.core.BoundaryDataset(items,tmp_path/'before.npy',tmp_path/'teacher.npy')
    next_data=study.core.BoundaryDataset(items,tmp_path/'after.npy',tmp_path/'teacher.npy')
    try:
        actual,boundary=study.seam_rows(native.model.layers[0],parent,native.model.norm,native.lm_head.weight,rotary,data,next_data,[0,1,2,3],lambda:None)
        expected=study.old.score(parent,native.model.norm,native.lm_head.weight,rotary,next_data,[0,1,2,3],lambda:None)
        assert actual==expected and boundary['exact'] and boundary['max_abs']==0
    finally:data.close();next_data.close()


def test_pilot_saves_canonical_raw_before_local_and_joint_discarded_updates(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_full_lease
    current=stage();directory=study.stage_dir(tmp_path,26);directory.mkdir(parents=True)
    plan={'label_ids':[0,1,2,3]};events=[]
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(2,dtype=torch.float32));self.register_buffer('basis',torch.eye(2))
    class Stack(torch.nn.Module):
        def __init__(self):
            super().__init__();self.blocks=torch.nn.ModuleList([Block() for _ in range(10)])
    stack=Stack();norm=Block();E=torch.ones(4,2);rotary=Block()
    data=SimpleNamespace(training=list(range(32)),lengths=list(range(32)),close=lambda:events.append('closed'))
    monkeypatch.setattr(study,'verify_stage',lambda *a:(plan,current))
    monkeypatch.setattr(study,'checked',lambda *a:{})
    lease={'id':'unit','allocation':{'GPU-0370fdb0-test':11264}}
    monkeypatch.setattr(jspace_full_lease,'wait_ready',lambda *a:lease)
    monkeypatch.setattr(study,'cap_allocator',lambda *a:{'allocator_limit_bytes':1000})
    for name in ['init','reset_peak_memory_stats']:monkeypatch.setattr(study.torch.cuda,name,lambda:None)
    monkeypatch.setattr(study.torch.cuda,'max_memory_reserved',lambda:0)
    monkeypatch.setattr(study.torch,'set_num_threads',lambda _:None)
    monkeypatch.setattr(study,'make_stack',lambda *a,**k:(stack,norm,E,rotary,{'passed':True}))
    monkeypatch.setattr(study,'data_for',lambda *a,**k:data)
    monkeypatch.setattr(study,'entrance_pairs',lambda *a:data)
    class Trainer:
        def __init__(self,*args):
            events.append('joint_construct')
            for module in (stack,norm):
                for parameter in module.parameters():parameter.data=parameter.data.bfloat16()
        def packed_development(self):
            events.append('raw_score');report=quality();return report['metrics'],report['records']
        def snapshot(self):return study.core.recursive_cpu({'stack':stack.state_dict(),'norm':norm.state_dict()})
        def master_update_probe(self,indices):
            assert indices==list(range(31,15,-1))
            assert all(torch.equal(p,torch.ones_like(p)) for p in stack.parameters())
            assert torch.equal(norm.weight,torch.ones_like(norm.weight))
            events.append('joint_probe')
            with torch.no_grad():
                for module in (stack,norm):
                    for p in module.parameters():p.add_(.5)
            return dict(passed=True,systems_only=True,discarded=True,update={'update':1},layer_indices=current['layers'],
                fp32_master_changed_by_block=[True]*10,norm_fp32_master_changed=True,fixed_state_unchanged=True)
    class Local:
        def __init__(self,*a,**k):events.append('entrance_construct')
        def pilot(self,indices):
            raw=study.native._load(directory/'pilot/initializer.pt')
            assert all(value.dtype==torch.bfloat16 for name,value in raw['stack'].items() if name.endswith('weight'))
            assert raw['stack']['blocks.0.basis'].dtype==torch.float32
            assert (directory/'pilot/initial_development.json').exists()
            events.append('entrance_probe')
            with torch.no_grad():stack.blocks[0].weight.add_(2)
            return dict(passed=True,discarded=True,systems_only=True,update={'update':1},fixed_state_unchanged=True,
                fp32_master_changed_by_parameter={'weight':True})
    monkeypatch.setattr(study,'_trainer',lambda *a,**k:Trainer())
    monkeypatch.setattr(study.entrance,'EntranceTrainer',Local)
    study.run_phase(tmp_path,26,'pilot')
    assert events[:6]==['joint_construct','raw_score','entrance_construct','entrance_probe','joint_construct','joint_probe']
    raw=study.native._load(directory/'pilot/initializer.pt')
    assert torch.equal(raw['stack']['blocks.0.weight'],torch.ones(2,dtype=torch.bfloat16))
    assert study.read(directory/'pilot/result.json')['entrance_restore_exact']
    assert study.read(directory/'pilot/manifest.json')['artifacts']['initializer.pt']==study.base.sha(directory/'pilot/initializer.pt')
