from copy import deepcopy

import numpy as np
import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RMSNorm,Qwen2RotaryEmbedding

from experiments.structure_net import jspace_qwen3b_progressive_train as study
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack


def components(cut=32):
    torch.manual_seed(27013)
    cfg=Qwen2Config(hidden_size=8,intermediate_size=12,num_attention_heads=2,num_key_value_heads=1,
        num_hidden_layers=36,vocab_size=16,max_position_embeddings=32,attention_dropout=0.)
    cfg._attn_implementation='eager';blocks=[];bases=[]
    for layer in range(cut,36):
        native=Qwen2DecoderLayer(cfg,layer_idx=layer).bfloat16().eval().requires_grad_(False)
        u=torch.linalg.qr(torch.randn(8,8)).Q[:,:4];bases.append(u.clone())
        block=AffineQwenBlock(native,u,torch.randn(1,8)/10,
            head_projection=dict(rope_pairs=torch.tensor([[0]]),value_basis=torch.eye(4)[None,:,:2]))
        relation=dict(anchors=[0,1],mean=torch.zeros(12),scale=torch.ones(12),coefficients={'2':torch.randn(12,6)/100})
        blocks.append(block.compile_polynomial(relation,[10,11]))
    parent=CompactQwenStack(deepcopy(blocks[1:]),geometry_mode='bf16',pad_to_multiple=16)
    norm=Qwen2RMSNorm(8).bfloat16()
    with torch.no_grad():
        norm.weight.fill_(3.)
        for p in parent.parameters():p.add_(.015625)
    previous=dict(stack=study.core.recursive_cpu(parent.state_dict()),norm=study.core.recursive_cpu(norm.state_dict()))
    for i,b in enumerate(blocks[1:]):
        prefix=f'blocks.{i}.'
        b.load_state_dict({k[len(prefix):]:v for k,v in previous['stack'].items() if k.startswith(prefix)},strict=True)
        b.basis=bases[i+1]
    stack=CompactQwenStack(blocks,geometry_mode='bf16',pad_to_multiple=16,checkpoint_blocks=True)
    fresh_norm=Qwen2RMSNorm(8).bfloat16()
    proof=study.preserve_inheritance(stack,fresh_norm,previous)
    return stack,fresh_norm,torch.randn(16,8).bfloat16(),Qwen2RotaryEmbedding(config=cfg),previous,proof


def data(path):
    torch.manual_seed(27014);items=[];offset=0
    for index,length in enumerate((4,6,5,4,5,4,6,5)):
        answer=index%2
        items.append(dict(identity=str(index),family='tiny',split='train' if index<4 else 'development',
            input_ids=list(range(length)),choices=['A','B'],answer=answer,targets=[4,5,6,2+answer],
            loss_positions=[0,1,length-2,length-1],offset=offset,stop=offset+length))
        offset+=length
    values=dict(boundary=torch.randn(offset,8).bfloat16(),accepted=torch.randn(8,4,16).bfloat16(),original=torch.randn(8,4,16).bfloat16())
    for name,value in values.items():np.save(path/(name+'.npy'),value.view(torch.int16).numpy())
    return study.core.BoundaryDataset(items,path/'boundary.npy',path/'accepted.npy',
        original_logits_path=path/'original.npy',original_indices=list(range(8)))


def trainer(path,cut=32):
    stack,norm,readout,rotary,*_=components(cut)
    config=study.core.RecoveryConfig(epochs=2,microbatch=2,accumulation=2,seed=20861,optimizer_chunk_bytes=128)
    return study.ProgressiveTrainer(stack,norm,readout,rotary,data(path),[2,3],
        native_correct=3,parent_correct=4,count=4,config=config)


def test_same_proven_recipe_and_parameterized_non_cumulative_native_gate():
    c=study.CONFIG
    assert (c.epochs,c.microbatch,c.accumulation,c.learning_rate,c.seed,c.dtype)==(12,4,4,1.2e-4,20861,'bfloat16')
    assert (c.temperature,c.kl_weight,c.ce_weight)==(2.,.7,.3)
    assert study.gates(48,native_correct=48,parent_correct=51,count=64)['passed']
    assert not study.gates(47,native_correct=48,parent_correct=51,count=64)['passed']
    assert study.gates(45,native_correct=48,parent_correct=47,count=64)['minimum_correct']==45
    assert not study.gates(44,native_correct=48,parent_correct=44,count=64)['passed']
    with pytest.raises(ValueError):study.gates(48,native_correct=48,parent_correct=51,count=64,maximum_drop=.1)
    assert study.selection_key(dict(correct=48,ce=100))>study.selection_key(dict(correct=47,ce=.01))
    assert study.selection_key(dict(correct=48,ce=1))==study.selection_key(dict(correct=48,ce=1))


@pytest.mark.parametrize('cut',[28,32,34])
def test_arbitrary_one_block_expansion_preserves_all_parent_weights_and_bridges(cut):
    stack,norm,_,_,previous,proof=components(cut)
    assert proof['unchanged_layers']==list(range(cut+1,36))
    assert proof['new_layer']==cut and proof['interior_bridges_preserved']==34-cut
    for i,bridge in enumerate(stack.bridges[1:]):
        prefix=f'bridges.{i}.'
        assert study.prior.state_equal(bridge.state_dict(),{k[len(prefix):]:v for k,v in previous['stack'].items() if k.startswith(prefix)})
    with torch.no_grad():next(stack.blocks[-1].parameters()).add_(.5)
    with pytest.raises(ValueError,match='Inherited block'):study.preserve_inheritance(stack,norm,previous)


def test_discarded_update_proves_all_masters_without_requiring_bf16_norm_motion(tmp_path):
    value=trainer(tmp_path)
    proof=value.master_update_probe(value.dataset.training)
    assert proof['passed'] and proof['systems_only'] and proof['discarded']
    assert proof['fp32_master_changed_by_block']==[True]*4
    assert proof['norm_fp32_master_changed'] and not proof['norm_bf16_storage_changed']
    assert proof['fixed_state_unchanged']
    with pytest.raises(ValueError,match='untouched'):value.fit(tmp_path/'fit')
    assert not (tmp_path/'fit').exists()
    value.dataset.close()


def test_real_generic_fit_records_all_epochs_parameterized_gates_and_exact_reload(tmp_path):
    value=trainer(tmp_path,cut=34);initial,rows=value.packed_development();events=[]
    result=value.fit(tmp_path/'fit',initial_reference=dict(initial=initial,initial_records=rows),
        event=lambda kind,**kwargs:events.append((kind,kwargs)))
    assert result['layer_indices']==[34,35] and result['suffix_blocks']==2 and result['cut']==34
    assert result['updates']==2 and result['completed_epochs']==2
    assert [r['epoch'] for r in result['history']]==[0,1,2]
    assert result['strict_reload_exact'] and result['fixed_state_unchanged']
    assert result['controls']==dict(native_correct=3,parent_correct=4,count=4)
    assert result['quality']==study.gates(result['selected']['correct'],native_correct=3,parent_correct=4,count=4)
    chosen=max(result['history'],key=lambda r:study.selection_key(r['metrics']))
    assert chosen['epoch']==result['selected_epoch']
    rates=[v['learning_rate'] for kind,v in events if kind=='update']
    assert rates==pytest.approx([1.2e-4,6e-5])
    assert study.prior.base.sha(tmp_path/'fit/selected.pt')==result['selected_checkpoint_sha256']
    for row in result['history']:
        assert study.prior.base.sha(tmp_path/f"fit/epoch_{row['epoch']:02d}_development.json")==row['development_sha256']
    value.dataset.close()


def test_bad_initializer_stops_before_update_and_development_cannot_train(tmp_path):
    value=trainer(tmp_path);initial,rows=value.packed_development();wrong=deepcopy(rows)
    wrong[0]['logits']=list(reversed(wrong[0]['logits']));wrong[0]['prediction']=1-wrong[0]['prediction']
    with pytest.raises(RuntimeError,match='before training'):
        value.fit(tmp_path/'fit',initial_reference=dict(initial=initial,initial_records=wrong))
    assert value.updates==0
    with pytest.raises(ValueError,match='training cases'):value.train_window(value.dataset.development)
    value.dataset.close()


def test_control_population_mismatch_rejected_before_optimizer_creation(tmp_path):
    stack,norm,readout,rotary,*_=components()
    dataset=data(tmp_path)
    with pytest.raises(ValueError,match='complete development'):
        study.ProgressiveTrainer(stack,norm,readout,rotary,dataset,[2,3],native_correct=48,parent_correct=51)
    dataset.close()
