from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer,Qwen3RMSNorm,Qwen3RotaryEmbedding

from experiments.structure_net.jspace_qwen3_block import Qwen3AffineBlock,Qwen3CompactStack
from experiments.structure_net.jspace_qwen3_capture import FP32BoundaryDataset
from experiments.structure_net import jspace_qwen3_train as module
from experiments.structure_net.jspace_streaming_adamw import StreamingCPUAdamW


@pytest.mark.parametrize("dtype",[torch.float32,torch.bfloat16])
def test_offloaded_accumulation_and_master_step_equal_resident_cpu_reference(dtype):
    p=torch.nn.Parameter(torch.tensor([.2,-.7,1.1,3.],dtype=dtype)); q=torch.nn.Parameter(p.detach().clone())
    off=module.CPUGradientAdamW([p],lr=.002,chunk_bytes=8)
    resident=StreamingCPUAdamW([q],lr=.002,chunk_bytes=8)
    for update in range(3):
        off.zero_grad();resident.zero_grad()
        for micro in range(3):
            x=torch.tensor([.3+micro,1.3,.11,.7],dtype=dtype)
            (p*x).float().square().mean().div(3).backward()
            (q*x).float().square().mean().div(3).backward()
            assert p.grad is None
        assert torch.equal(off.cpu_gradients[p],q.grad)
        assert off.received[p]==3 and off.cpu_gradients[p].dtype==dtype
        a=off.step(max_grad_norm=1.); b=resident.step(max_grad_norm=1.)
        assert a['unclipped_gradient_norm']==b['unclipped_gradient_norm']
        assert a['master_changed_by_parameter']==[True]
        assert torch.equal(p,q)
        for key in ('master','exp_avg','exp_avg_sq'): assert torch.equal(off.state[p][key],resident.state[q][key])
    off.close()


def test_nonfinite_offloaded_gradient_rejects_before_any_update():
    p=torch.nn.Parameter(torch.ones(3));q=torch.nn.Parameter(torch.ones(3))
    opt=module.CPUGradientAdamW([p,q],chunk_bytes=8)
    (p.sum()+q.sum()*float('nan')).backward()
    before=[opt.state[x]['master'].clone() for x in (p,q)]
    with pytest.raises(FloatingPointError):opt.step(max_grad_norm=1.)
    assert all(torch.equal(a,opt.state[b]['master']) for a,b in zip(before,(p,q)))
    assert all(opt.state[x]['step']==0 for x in (p,q))
    opt.close()


def components(dtype):
    torch.manual_seed(219)
    config=Qwen3Config(hidden_size=8,intermediate_size=12,num_hidden_layers=2,num_attention_heads=2,
        num_key_value_heads=1,head_dim=4,vocab_size=16,attention_dropout=0.,max_position_embeddings=16,tie_word_embeddings=False)
    config._attn_implementation='eager';blocks=[]
    for layer in range(2):
        native=Qwen3DecoderLayer(config,layer).to(dtype).eval().requires_grad_(False)
        basis=torch.linalg.qr(torch.randn(8,8)).Q[:,:6]
        block=Qwen3AffineBlock(native,basis,torch.randn(8,8)/10,
            head_projection={'rope_pairs':torch.tensor([[0]]),'value_basis':torch.eye(4)[None,:,:2]})
        fit={'anchors':[0,1],'mean':torch.zeros(12),'scale':torch.ones(12),'coefficients':{'2':torch.randn(12,6)/100}}
        blocks.append(block.compile_polynomial(fit,[4,5]))
    return Qwen3CompactStack(blocks,pad_to_multiple=16),Qwen3RMSNorm(8).to(dtype),torch.randn(16,8).to(dtype)/4,Qwen3RotaryEmbedding(config)


def dataset(path):
    path.mkdir(exist_ok=True);torch.manual_seed(2191);items=[];offset=0
    for i,length in enumerate([5,7,4,6,5,7,4,5,7]):
        ids=list(range(length));pos=[0,(length-2)//2,length-2,length-1];answer=i%2
        items.append(dict(identity=f'item{i}',family='arc',prompt=f'question{i}',split='train' if i<7 else 'development',
            input_ids=ids,offset=offset,stop=offset+length,loss_positions=pos,
            targets=[ids[p+1] for p in pos[:-1]]+[answer],answer=answer,choices=['a','b']))
        offset+=length
    np.save(path/'boundary.npy',torch.randn(offset,8).numpy())
    logits=torch.randn(len(items),4,16).numpy();np.save(path/'accepted.npy',logits)
    return FP32BoundaryDataset(items,path/'boundary.npy',path/'accepted.npy',label_ids=[0,1,2,3],
        original_logits_path=path/'accepted.npy',original_indices=list(range(len(items))))


def make_trainer(data,dtype=torch.float32,**kwargs):
    config=module.RecoveryConfig(dtype=str(dtype).split('.')[-1],microbatch=2,accumulation=2,epochs=2,
        learning_rate=.001,optimizer_chunk_bytes=128,seed=20861)
    return module.Qwen3SuffixTrainer(*components(dtype),data,[0,1,2,3],config=config,host_reserve_bytes=0,**kwargs)


def same_state(a,b):
    return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) if isinstance(a[k],torch.Tensor) else a[k]==b[k] for k in a)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_actual_training_all_blocks_norm_and_hooks_survive_packed_development(tmp_path,dtype):
    data=dataset(tmp_path);trainer=make_trainer(data,dtype)
    fixed=module.fixed_state(trainer.stack,trainer.readout,trainer.rotary)
    update=trainer.train_window([0,1,2])
    assert update['gradients_present_by_block']==[True,True]
    assert update['fp32_master_changed_by_block']==[True,True]
    assert update['norm_gradients_present'] and update['norm_fp32_master_changed']
    assert all(p.grad is None for p in trainer.parameters)
    before=trainer.snapshot();masters=[s['master'].clone() for s in trainer.optimizer.state.values()]
    metrics,rows=trainer.packed_development()
    assert metrics['count']==2 and len(rows)==2 and metrics['selection_metric']==module.SELECTION
    assert metrics['accepted_original_kl']==pytest.approx(0.,abs=1e-7)
    assert same_state(before['stack'],trainer.stack.state_dict()) and same_state(before['norm'],trainer.norm.state_dict())
    assert all(torch.equal(a,b['master']) for a,b in zip(masters,trainer.optimizer.state.values()))
    assert tuple(map(id,trainer.parameters))==trainer.parameter_ids
    assert all(not p.is_inference() for p in trainer.parameters)
    second=trainer.train_window([3,4]);assert all(second['gradients_present_by_block'])
    assert module.fixed_state(trainer.stack,trainer.readout,trainer.rotary)==fixed
    assert trainer.readout.grad is None
    assert any(p.dtype==torch.float32 for name,p in trainer.stack.named_parameters() if name.endswith('polynomial_bias'))
    trainer.optimizer.close();data.close()


def test_fit_actual_budget_accuracy_then_ce_and_strict_reload(tmp_path):
    data=dataset(tmp_path/'data');trainer=make_trainer(data)
    result=trainer.fit(tmp_path/'fit')
    assert result['updates']==4 and result['completed_epochs']==2
    assert [row['epoch'] for row in result['history']]==[0,1,2]
    assert [w['cases'] for w in result['history'][1]['training']]==[4,3]
    winner=max(result['history'],key=lambda r:(r['metrics']['correct'],-r['metrics']['ce']))
    assert winner['epoch']==result['selected_epoch']
    assert result['strict_reload_exact'] and result['fixed_state_unchanged']
    assert result['gradient_mode']=='cpu_parameter_dtype'
    assert module.sha(tmp_path/'fit/selected.pt')==result['selected_checkpoint_sha256']
    state=torch.load(tmp_path/'fit/selected.pt',weights_only=True)
    fresh,norm,_,_=components(torch.float32)
    fresh.load_state_dict(state['stack'],strict=True);norm.load_state_dict(state['norm'],strict=True)
    assert same_state(fresh.state_dict(),trainer.stack.state_dict())
    for row in result['history']:
        assert module.sha(tmp_path/f"fit/epoch_{row['epoch']:02d}_development.json")==row['development_sha256']
    with pytest.raises(RuntimeError,match='finished'):trainer.train_window([0])
    trainer.optimizer.close();data.close()


def test_initializer_mismatch_stops_before_update_and_no_test_training(tmp_path):
    data=dataset(tmp_path/'data');trainer=make_trainer(data)
    with pytest.raises(ValueError,match='Training cases'):trainer.train_window([data.development[0]])
    with pytest.raises(RuntimeError,match='before any update'):
        trainer.fit(tmp_path/'fit',initial_reference={'metrics':{},'records':[]})
    assert trainer.updates==0
    trainer.optimizer.close();data.close()


def test_combined_host_budget_and_live_available_guard(monkeypatch):
    p=torch.nn.Parameter(torch.ones(8,dtype=torch.bfloat16))
    with pytest.raises(MemoryError,match='Combined'):module.CPUGradientAdamW([p],max_cpu_state_bytes=111)
    monkeypatch.setattr(module,'available_host_bytes',lambda:100)
    with pytest.raises(MemoryError,match='Host stage'):module.require_host_memory(90,reserve_bytes=11)
    assert module.require_host_memory(90,reserve_bytes=10)['available_bytes']==100


def test_declared_default_recipe_is_unchanged():
    c=module.DEFAULT_CONFIG
    assert (c.epochs,c.microbatch,c.accumulation,c.seed)==(12,4,4,20861)
    assert (c.learning_rate,c.temperature,c.kl_weight,c.ce_weight)==(1.2e-4,2.,.7,.3)
    assert c.max_cpu_state_bytes==48*1024**3
