"""CPU parity for joint packed-logit/feature capture and resident feature KD."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm,Qwen2RotaryEmbedding

from experiments.structure_net import jspace_3b_parent_feature as study
from test_jspace_3b_hidden_feature_run import tiny,inputs
from test_jspace_3b_feature_recovery import TinyData

PARENT='a'*64


def source(interleaved=False):
    stack=tiny()
    for p in stack.parameters():p.data=p.data.bfloat16()
    stack.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16,strip_unused=True).eval().requires_grad_(False)
    norm=Qwen2RMSNorm(8).bfloat16().eval().requires_grad_(False)
    E=torch.randn(16,8).bfloat16().requires_grad_(False)
    rotary=Qwen2RotaryEmbedding(Qwen2Config(hidden_size=8,num_attention_heads=2,max_position_embeddings=16)).eval().requires_grad_(False)
    data=TinyData();offset=0
    for index,case in enumerate(data.items):
        length=data.lengths[index]
        case.update(input_ids=list(range(length)),offset=offset,stop=offset+length,loss_positions=[0,0,1,length-1]);offset+=length
        if interleaved:case['split']='development' if index in (1,5) else 'train'
    data.training=[i for i,c in enumerate(data.items) if c['split']=='train']
    data.development=[i for i,c in enumerate(data.items) if c['split']=='development']
    return stack,norm,E,rotary,data


def test_explicit_packed_execution_matches_original_and_returns_actual_final_z():
    stack,norm,E,rotary,data=source();x,batch,rope=inputs();before=deepcopy(stack.state_dict())
    with torch.inference_mode():
        expected=stack(x,attention_mask=batch['attention_mask'],position_ids=batch['position_ids'],
            cache_position=batch['position_ids'][0],position_embeddings=rope,use_cache=False)[0]
    output,z,execution=study.packed_coordinates(stack,x,attention_mask=batch['attention_mask'],position_ids=batch['position_ids'],position_embeddings=rope)
    assert torch.equal(output,expected) and z.shape==(2,5,4)
    actual=study.feature.capture.diagnostic.trace_coordinates(stack,x,attention_mask=batch['attention_mask'],
        position_ids=batch['position_ids'],position_embeddings=rope,layers=[35])
    assert torch.equal(z,actual['coordinates'][35])
    assert execution==dict(fuse_qkv=False,fuse_mlp=False)
    assert study.feature.old.state_equal(stack.state_dict(),before)


def test_combined_capture_matches_original_logits_with_interleaved_case_rows(tmp_path):
    stack,norm,E,rotary,data=source(interleaved=True)
    legacy=study.original.capture_packed_teacher(tmp_path/'legacy.npy',stack,norm,E,rotary,data,label_ids=[0,1,2,3])
    result=study.capture_packed_parent(tmp_path/'combined',stack,norm,E,rotary,data,parent_checkpoint_sha256=PARENT,label_ids=[0,1,2,3])
    assert np.array_equal(np.load(tmp_path/'legacy.npy'),np.load(tmp_path/'combined/packed_teacher_logits.npy'))
    assert result['logits']['records']==legacy['records'] and result['logits']['batches']==legacy['batches']
    targets,receipt=study.load_targets(tmp_path/'combined',data.items,PARENT)
    assert receipt==result['features'] and receipt['source_indices']==data.training
    assert len(targets.lookup)==32 and all(i not in targets.lookup for i in data.development)
    bank=np.load(tmp_path/'combined/parent_training_final_answers.npy')
    for start in range(0,len(data.training),4):
        chosen=data.training[start:start+4];batch=data.batch(chosen)
        with torch.inference_mode():rope=rotary(batch['hidden'],batch['position_ids'])
        _,z,_=study.packed_coordinates(stack,batch['hidden'],attention_mask=batch['attention_mask'],
            position_ids=batch['position_ids'],position_embeddings=rope)
        answer=z[torch.arange(len(chosen)),torch.tensor(batch['lengths'])-1]
        assert np.array_equal(bank[start:start+len(chosen)],answer.view(torch.int16).numpy())
    assert receipt['parent_cut']==33 and receipt['parent_layers']==[33,34,35] and receipt['normalization_energy']==targets.energy
    assert receipt['development_fitted'] is False
    with pytest.raises(FileExistsError):study.capture_packed_parent(tmp_path/'combined',stack,norm,E,rotary,data,parent_checkpoint_sha256=PARENT)


def test_development_changes_cannot_change_training_targets_or_energy(tmp_path):
    stack,norm,E,rotary,data=source(interleaved=True)
    a=study.capture_packed_parent(tmp_path/'a',stack,norm,E,rotary,data,parent_checkpoint_sha256=PARENT)
    data.x[data.development]*=12
    b=study.capture_packed_parent(tmp_path/'b',stack,norm,E,rotary,data,parent_checkpoint_sha256=PARENT)
    assert a['features']['sha256']==b['features']['sha256']
    assert a['features']['normalization_energy']==b['features']['normalization_energy']
    assert a['logits']['sha256']!=b['logits']['sha256']


@pytest.mark.parametrize('change',['parent','rows','bits'])
def test_loader_rejects_wrong_parent_row_identity_or_corrupt_bank(tmp_path,change):
    stack,norm,E,rotary,data=source();directory=tmp_path/'capture'
    study.capture_packed_parent(directory,stack,norm,E,rotary,data,parent_checkpoint_sha256=PARENT)
    items=deepcopy(data.items);parent=PARENT
    if change=='parent':parent='b'*64
    elif change=='rows':items[0]['identity']='different'
    else:(directory/'parent_training_final_answers.npy').write_bytes(b'wrong bits')
    with pytest.raises(ValueError):study.load_targets(directory,items,parent)


def test_guard_and_frozen_teacher_checks_precede_capture_or_forward(tmp_path,monkeypatch):
    stack,norm,E,rotary,data=source()
    def denied():raise PermissionError('lease missing')
    with pytest.raises(PermissionError):study.capture_packed_parent(tmp_path/'denied',stack,norm,E,rotary,data,parent_checkpoint_sha256=PARENT,guard=denied)
    assert not (tmp_path/'denied').exists()
    stack.train()
    with pytest.raises(ValueError,match='frozen'):study.capture_packed_parent(tmp_path/'train',stack,norm,E,rotary,data,parent_checkpoint_sha256=PARENT)
    assert not (tmp_path/'train').exists()


def test_resident_adoption_preserves_initial_state_and_cpu_feature_update_math():
    torch.manual_seed(813)
    stack=tiny();norm=Qwen2RMSNorm(8).bfloat16();E=torch.randn(16,8).bfloat16().requires_grad_(False)
    rotary=Qwen2RotaryEmbedding(Qwen2Config(hidden_size=8,num_attention_heads=2,max_position_embeddings=16))
    data=TinyData();targets=study.feature.AnswerTargets(torch.randn(32,4).bfloat16(),data.training)
    config=replace(study.feature.CONFIG,epochs=1,microbatch=4,accumulation=1,optimizer_chunk_bytes=64)
    kwargs=dict(targets=targets,alpha=.2,config=config,native_correct=2,parent_correct=2,count=2)
    cpu=study.feature.FeatureTrainer(deepcopy(stack),deepcopy(norm),E.clone(),deepcopy(rotary),data,[0,1],**kwargs)
    resident=study.ResidentFeatureTrainer(deepcopy(stack),deepcopy(norm),E.clone(),deepcopy(rotary),data,[0,1],
        resident_device='cpu',resident_budget_bytes=4000,**kwargs)
    assert isinstance(resident.optimizer.optimizer,study.ResidentSpillAdamW)
    assert set(resident.optimizer.state)==set(resident.parameters)
    for p,q in zip(cpu.parameters,resident.parameters):
        assert torch.equal(p,q)
        for key in ('master','exp_avg','exp_avg_sq'):assert torch.equal(cpu.optimizer.state[p][key],resident.optimizer.state[q][key])
    a=cpu.train_window([0,1,2,3]);b=resident.train_window([0,1,2,3])
    for key in ('loss','endpoint_loss','feature_loss','kl','ce','alpha','gradients_present_by_block'):assert a[key]==b[key]
    assert b['synchronized_train_window_seconds']>0
    for p,q in zip(cpu.parameters,resident.parameters):
        assert torch.equal(p,q)
        for key in ('master','exp_avg','exp_avg_sq'):assert torch.equal(cpu.optimizer.state[p][key],resident.optimizer.state[q][key])
    assert study.ResidentFeatureTrainer.fit is study.feature.FeatureTrainer.fit
    assert study.ResidentFeatureTrainer.calibrate is study.feature.FeatureTrainer.calibrate
    assert study.ResidentFeatureTrainer.packed_development is study.feature.FeatureTrainer.packed_development
