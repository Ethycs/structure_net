"""Real tiny Qwen entrance/bridge gradients, immutable parents and local fit."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding

from experiments.structure_net import jspace_3b_entrance_distill as study
from experiments.structure_net import jspace_qwen_compact_stack as stack_module
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock


def components():
    torch.manual_seed(11913)
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_attention_heads=2,
        num_key_value_heads=1, num_hidden_layers=36, vocab_size=16, max_position_embeddings=16,
        attention_dropout=0.)
    config._attn_implementation = 'eager'; blocks = []
    for index in range(33,36):
        native = Qwen2DecoderLayer(config, layer_idx=index).bfloat16().eval().requires_grad_(False)
        basis = torch.linalg.qr(torch.randn(8,8)).Q[:,:4]
        block = AffineQwenBlock(native, basis, torch.randn(8,8)/20,
            head_projection=dict(rope_pairs=torch.tensor([[0]]), value_basis=torch.eye(4)[None,:,:2]))
        fit = dict(anchors=[0,1], mean=torch.zeros(12), scale=torch.ones(12),
            coefficients={'2':torch.randn(12,6)/100})
        blocks.append(block.compile_polynomial(fit,[10,11]))
    stack = study.CompactQwenStack(blocks, geometry_mode='bf16', pad_to_multiple=16)
    return stack, Qwen2RMSNorm(8).bfloat16(), torch.randn(16,8).bfloat16(), Qwen2RotaryEmbedding(config=config)


def pairs(path):
    torch.manual_seed(11914); items=[]; offset=0
    for index,length in enumerate((3,5,4,6,3,5)):
        items.append(dict(identity=str(index), split='train' if index<4 else 'development',
            input_ids=list(range(length)), offset=offset, stop=offset+length))
        offset += length
    native = torch.randn(offset,8).bfloat16()
    target = (native.float()+torch.randn_like(native.float())*.3).bfloat16()
    np.save(path/'input.npy',native.view(torch.int16).numpy())
    np.save(path/'target.npy',target.view(torch.int16).numpy())
    return study.EntrancePairs(items,path/'input.npy',path/'target.npy')


def trainer(path, **kwargs):
    config = study.EntranceConfig(epochs=2,microbatch=2,accumulation=2,optimizer_chunk_bytes=128)
    return study.EntranceTrainer(*components(),pairs(path),config=config,**kwargs)


def test_defaults_and_config_reject_invalid_values():
    c=study.EntranceConfig()
    assert (c.epochs,c.microbatch,c.accumulation,c.learning_rate,c.seed)==(12,4,4,2e-5,20861)
    for kwargs in ({'floor_fraction':float('nan')},{'floor_fraction':0},{'max_cpu_state_bytes':0}):
        with pytest.raises(ValueError):study.EntranceConfig(**kwargs)


def test_handoff_is_exact_actual_stack_input_to_parent_without_reconstruction(tmp_path,monkeypatch):
    value=trainer(tmp_path);batch=value.data.batch([0,1],value.device)
    hidden=batch['hidden'].bfloat16();positions=batch['positions'];length=hidden.shape[1]
    allowed=torch.arange(length)[None,:]<=torch.arange(length)[:,None]
    allowed=allowed[None]&batch['valid'][:,None,:]
    mask=torch.zeros(2,1,length,length,dtype=torch.bfloat16).masked_fill(~allowed[:,None],torch.finfo(torch.bfloat16).min)
    rotary=value.rotary(hidden,positions); captured=[]; original=stack_module._run_reduced
    def record(block,z,*args,**kwargs):
        if block is value.stack.blocks[1]:captured.append(z.detach().clone())
        return original(block,z,*args,**kwargs)
    monkeypatch.setattr(stack_module,'_run_reduced',record)
    value.stack(hidden,position_ids=positions,cache_position=positions[0],attention_mask=mask,position_embeddings=rotary)
    actual=study.entrance_coordinates(value.stack,hidden,positions,mask,rotary)
    assert actual.shape==(2,length,4) and len(captured)==1
    assert torch.equal(actual.detach(),captured[0])
    assert torch.equal(value.parent_coordinates(batch['target']),
        torch.nn.functional.linear(batch['target'].bfloat16(),value.stack.blocks[1].basis.T))
    value.data.close()


def test_real_disposable_update_changes_entrance_master_only(tmp_path):
    value=trainer(tmp_path)
    parent=deepcopy(value.stack.blocks[1].state_dict());norm=deepcopy(value.norm.state_dict())
    keys=tuple(value.stack.state_dict()); ids=tuple(id(p) for p in value.stack.parameters())
    proof=value.pilot(value.data.training)
    assert proof['passed'] and proof['discarded'] and proof['systems_only']
    assert proof['fp32_master_before']!=proof['fp32_master_after']
    assert set(proof['fp32_master_changed_by_parameter'])==set(dict(value.block.named_parameters()))
    assert all(proof['fp32_master_changed_by_parameter'].values())
    assert proof['update']['only_entrance_gradients']
    assert proof['update']['optimizer']['unclipped_gradient_norm']>0
    assert study.evidence.state_equal(parent,value.stack.blocks[1].state_dict())
    assert study.evidence.state_equal(norm,value.norm.state_dict())
    assert tuple(value.stack.state_dict())==keys and tuple(id(p) for p in value.stack.parameters())==ids
    assert all(p.grad is None and not p.requires_grad for b in value.stack.blocks[1:] for p in b.parameters())
    assert all(not p.requires_grad for p in value.norm.parameters()) and not value.readout.requires_grad
    assert all(not b.requires_grad for b in value.stack.buffers())
    with pytest.raises(ValueError,match='untouched'):value.fit(tmp_path/'invalid')
    value.data.close()


def test_floor_uses_training_only_and_exact_parent_coordinate_update(tmp_path):
    value=trainer(tmp_path); energies=[]
    for start in range(0,4,2):
        b=value.data.batch([start,start+1],value.device)
        update=value.parent_coordinates(b['target']).float()-value.parent_coordinates(b['hidden']).float()
        energies.append(update.square().mean(-1)[b['valid']])
    expected=float(torch.cat(energies).median())
    assert value.normalization['train_case_count']==4
    assert value.normalization['train_token_count']==18
    assert value.normalization['median_teacher_update_energy']==expected
    assert value.normalization['floor']==expected*1e-4
    value.data.close()
    target=np.load(tmp_path/'target.npy',mmap_mode='r+'); target[18:]=32700; target.flush();del target
    repeated=study.EntrancePairs(value.data.items,tmp_path/'input.npy',tmp_path/'target.npy')
    second=study.EntranceTrainer(*components(),repeated,config=value.config)
    assert second.normalization==value.normalization
    second.data.close()


def test_actual_fit_keeps_initializer_eligible_and_delta_strictly_compatible(tmp_path):
    observed=[]
    value=trainer(tmp_path,quality=lambda stack:observed.append(tuple(stack.layer_indices)) or {'correct':1,'count':2})
    original=deepcopy(value.stack);result=value.fit(tmp_path/'fit')
    assert result['updates']==2 and result['completed_epochs']==2
    assert result['strict_reload_exact'] and result['fixed_state_unchanged']
    assert result['task_quality_guaranteed'] is False and result['confirmed'] is False
    assert [r['epoch'] for r in result['history']]==[0,1,2] and len(observed)==3
    winner=min(result['history'],key=lambda r:r['metrics']['loss'])
    assert result['selected_epoch']==winner['epoch']
    state=torch.load(tmp_path/'fit/selected_entrance.pt',weights_only=True)
    original.blocks[0].load_state_dict(state['block'],strict=True)
    assert tuple(original.state_dict())==tuple(value.stack.state_dict())
    assert study.evidence.state_equal(original.state_dict(),value.stack.state_dict())
    assert result['selected_checkpoint_sha256']==study.evidence.base.sha(tmp_path/'fit/selected_entrance.pt')
    assert value.development()==result['selected']
    with pytest.raises(ValueError):value.train_window(value.data.training)
    value.data.close()


def test_development_rows_cannot_train_and_degenerate_teacher_is_rejected(tmp_path):
    value=trainer(tmp_path)
    with pytest.raises(ValueError,match='training cases'):value.train_window([value.data.development[0]])
    value.data.close()
    (tmp_path/'target.npy').write_bytes((tmp_path/'input.npy').read_bytes())
    data=study.EntrancePairs(value.data.items,tmp_path/'input.npy',tmp_path/'target.npy')
    with pytest.raises(ValueError,match='Degenerate'):study.EntranceTrainer(*components(),data,config=value.config)
    data.close()


def test_quality_callback_must_not_mutate_live_model(tmp_path):
    def bad_quality(stack):
        with torch.no_grad():next(stack.blocks[-1].parameters()).add_(1)
        return {'correct':2}
    value=trainer(tmp_path,quality=bad_quality)
    with pytest.raises(RuntimeError,match='mutated'):value.fit(tmp_path/'fit')
    value.data.close()
