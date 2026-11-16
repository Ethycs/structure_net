"""Native Qwen functional controls and sequential full-response distillation."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from experiments.structure_net import jspace_3b_ablation32_compile as study
from experiments.structure_net import jspace_3b_ablation32_geometry as geometry
from experiments.structure_net import jspace_3b_packed_response_model as packed


def fixture(dtype=torch.float32):
    torch.manual_seed(220311)
    config=Qwen2Config(hidden_size=8,intermediate_size=12,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,vocab_size=16,
        max_position_embeddings=32,attention_dropout=0.,tie_word_embeddings=True)
    config._attn_implementation='eager'
    native=Qwen2ForCausalLM(config).to(dtype).eval().requires_grad_(False)
    with torch.no_grad():
        for b in native.model.layers:
            b.input_layernorm.weight.copy_(torch.linspace(.5,1.5,8).to(dtype))
            b.post_attention_layernorm.weight.copy_(torch.linspace(1.6,.6,8).to(dtype))
    gs=[geometry.DiscardedSubspace(width=8,retained=4,seed=220101+i).export(
        anchor=torch.randn(8,8)/5)for i in range(2)]
    stack=study.compile_stack(native.model.layers,gs,parameter_dtype=dtype,expected_layers=2)
    return native,gs,stack


def context(native,ids):
    hidden=native.model.embed_tokens(ids)
    positions=torch.arange(ids.shape[1])[None].expand_as(ids)
    mask=torch.zeros(1,1,ids.shape[1],ids.shape[1],dtype=hidden.dtype)
    mask.masked_fill_(~torch.ones(ids.shape[1],ids.shape[1],dtype=torch.bool).tril(),torch.finfo(hidden.dtype).min)
    return hidden,positions,mask,native.model.rotary_emb(hidden,positions)


def test_three_projected_sites_equal_compiler_and_differentiate_discarded_space():
    native,gs,stack=fixture();ids=torch.tensor([[1,3,7,4]])
    hidden,pos,mask,rope=context(native,ids)
    expected=hidden
    for i in range(2):
        proof=study.compiler_parity(native.model.layers[i],stack.blocks[i],gs[i]['discarded'],
            gs[i]['anchor'],expected,position_ids=pos,position_embeddings=rope,attention_mask=mask,
            atol=2e-6,rtol=2e-5)
        assert proof['passed'],proof
        expected=study.projected_native_block(native.model.layers[i],expected,gs[i]['discarded'],
            gs[i]['anchor'],position_ids=pos,position_embeddings=rope,attention_mask=mask)
    actual=stack(hidden,position_ids=pos,position_embeddings=rope,attention_mask=mask,use_cache=False)[0]
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    D=gs[0]['discarded'].float().requires_grad_(True)
    value=study.projected_native_block(native.model.layers[0],hidden,D,gs[0]['anchor'],
        position_ids=pos,position_embeddings=rope,attention_mask=mask)
    value.square().sum().backward()
    assert D.grad is not None and torch.isfinite(D.grad).all() and D.grad.norm()>0
    assert all(p.grad is None for p in native.parameters())
    assert all(b.head_dim==4 and b.intermediate_size==12 and not b.has_polynomial for b in stack.blocks)
    with pytest.raises(ValueError,match='All native blocks'):
        study.compile_stack(native.model.layers[:1],gs[:1],expected_layers=2)


class Dataset:
    def __init__(self,native,layer):
        data=study.response_training.module('distillation_data')
        provenance=dict(source_uri='fixture',source_revision='v1',tokenizer_sha256='t',chat_template_sha256='c',
            prompt_provenance='fixture',response_provenance='reference',assistant_boundary_verified=True)
        self.records=[data.ResponseRecord(example_id=f'case{i}',group_id=f'g{i}',split='train',task='fixture',
            prompt_ids=(1,3+i),response_ids=(6,4)if i%2==0 else(8,9,4),provenance=provenance,
            finish_reason='eos',eos_token_id=4)for i in range(3)]
        self.batches=[]
        with torch.no_grad():
            for r in self.records:
                b=data.response_batch([r],pad_token_id=0,max_length=8)
                h,pos,mask,rope=context(native,b['input_ids'])
                for i,block in enumerate(native.model.layers):
                    h=block(h,position_ids=pos,position_embeddings=rope,attention_mask=mask,use_cache=False)[0]
                    if i==layer:b['next_boundary_hidden']=h.detach().clone()
                b.update(teacher_hidden=native.model.norm(h).detach(),teacher_input_ids=b['input_ids'].clone(),
                    teacher_record_ids=[r.example_id])
                self.batches.append(b)
    def batch(self,indices):
        assert len(indices)==1
        return dict(self.batches[indices[0]])


def make_stage(layer=0,dtype=torch.float32,train_norm=False):
    native,gs,stack=fixture(dtype)
    lookahead=study.FrozenNativeLookahead(native.model.layers[layer+1:],native.model.norm,
        native.model.embed_tokens.weight,native.model.rotary_emb,start=layer+1,expected_layers=2)
    stage=study.SequentialStage(stack,native.model.embed_tokens.weight,
        deepcopy(native.model.rotary_emb),lookahead,layer=layer,train_final_norm=train_norm)
    return native,stage,Dataset(native,layer)


def test_actual_student_prefix_cache_is_bound_to_weights_and_tokens():
    native,stage,dataset=make_stage(layer=1)
    batch=dataset.batch([0]);before=stage.prefix_coordinates(batch).clone()
    with torch.no_grad():stage.stack.blocks[0].mlp_output.add_(.2)
    changed=stage.prefix_coordinates(batch)
    assert not torch.equal(before,changed)
    # Construct the stage after choosing/freezing the earlier block.
    stage=study.SequentialStage(stage.stack,stage.embedding,stage.rotary,stage.lookahead,layer=1)
    coordinates=stage.prefix_coordinates(batch)
    cached=dict(batch,student_input_coordinates=coordinates,
        student_prefix_fingerprint=stage.prefix_fingerprint,student_prefix_input_ids=batch['input_ids'].clone())
    a=stage.forward(batch);b=stage.forward(cached)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=0,rtol=0)
    with pytest.raises(ValueError,match='bound'):
        stage.forward(dict(cached,student_prefix_fingerprint='wrong'))
    with pytest.raises(ValueError,match='bound'):
        stage.forward(dict(cached,student_prefix_input_ids=batch['input_ids']+1))


@pytest.mark.parametrize('layer,train_norm',[(0,False),(1,True)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_real_stage_update_freezes_teacher_and_other_blocks(layer,train_norm,dtype,tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:pytest.fail('CPU test initialized CUDA'))
    native,stage,dataset=make_stage(layer=layer,train_norm=train_norm,dtype=dtype)
    original_norm=native.model.norm.weight.detach().clone()
    cfg=replace(study.DistillationConfig(),epochs=2,accumulation_examples=2,learning_rate=.01,
        token_chunk_size=1,optimizer_chunk_bytes=64)
    trainer=study.StageTrainer(stage,dataset,teacher_id='native',train_indices=[0,1],selection_indices=[2],config=cfg)
    before=[p.detach().clone()for p in trainer.parameters];fixed=trainer.fixed_signature()
    try:
        trainer.begin_epoch(1)
        receipt=trainer.train_window([0,1])
        assert receipt['response_tokens']==5 and receipt['native_suffix_blocks_frozen']
        assert receipt['coordinate_tokens']==9
        assert receipt['exact_frozen_native_lookahead']==(not train_norm)
        assert receipt['optimizer']['updated_scalars']==sum(p.numel()for p in trainer.parameters)
        assert all(not torch.equal(p,x)for p,x in zip(trainer.parameters,before))
        assert trainer.fixed_signature()==fixed
        torch.testing.assert_close(native.model.norm.weight,original_norm,atol=0,rtol=0)
        assert all(p.grad is None for p in native.parameters())
        selected=trainer.development()
        assert selected['response_tokens']==2 and selected['teacher_kl']>=0
        output=trainer.save_checkpoint(tmp_path/'block.pt',identities={'fixture':True})
        payload=torch.load(output['path'],weights_only=True)
        assert 'stack'not in payload and(payload['norm']is not None)==train_norm
        fresh=fixture(dtype)[2]
        study.restore_selected_block(fresh,output['path'],expected_sha256=output['sha256'],norm=deepcopy(native.model.norm))
        for key,value in stage.block.state_dict().items():
            torch.testing.assert_close(fresh.blocks[layer].state_dict()[key],value,atol=0,rtol=0)
        trainer.begin_epoch(2)
        trainer.train_window([0])
        with pytest.raises(ValueError,match='epoch'):trainer.begin_epoch(3)
        with pytest.raises(ValueError,match='TRAIN'):trainer.train_window([2])
    finally:trainer.close()


def test_feature_adjoint_matches_direct_combined_objective():
    _,stage,dataset=make_stage();cfg=replace(study.DistillationConfig(),epochs=1,accumulation_examples=1,
        token_chunk_size=1,optimizer_chunk_bytes=64)
    # Independent direct loss autograd reference before CPU optimizer hooks.
    b=dataset.batch([0]);z,h=stage.forward(b);mask=b['loss_mask'];pos=mask.nonzero()
    delta=z.float()-F.linear(b['next_boundary_hidden'].float(),stage.block.basis.T.float())
    t=F.linear(b['teacher_hidden'][pos[:,0],pos[:,1]],stage.embedding)
    logits=stage.lookahead.readout(h[pos[:,0],pos[:,1]])
    loss_config=study.response_training.nal.DistillationConfig(objective='forward_kl',temperature=2.,ce_weight=0.)
    vectors=study.response_training.nal.loss_vectors(logits,t,b['targets'][mask],loss_config)
    direct=delta.square().mean(-1)[b['attention_mask'].bool()].mean()+vectors['forward_kl'].mean()
    reference=torch.autograd.grad(direct,list(stage.block.parameters()))
    trainer=study.StageTrainer(stage,dataset,teacher_id='native',train_indices=[0],selection_indices=[2],config=cfg)
    observed=[];realstep=trainer.optimizer.step
    def step(**kwargs):
        observed.extend(trainer.optimizer.cpu_gradients[p].clone()for p in trainer.parameters)
        return realstep(**kwargs)
    trainer.optimizer.step=step
    try:
        trainer.begin_epoch(1);trainer.train_window([0])
        for x,y in zip(reference,observed):torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
    finally:trainer.close()


def test_compile_bank_reload_and_packed_geometry_roundtrip(tmp_path):
    native,gs,_=fixture(torch.bfloat16);model_root=tmp_path/'native';native.save_pretrained(model_root)
    geom_root=tmp_path/'geometry';geom_root.mkdir();layers={}
    for i,g in enumerate(gs):
        path=geom_root/f'layer_{i:02d}.pt';torch.save(g,path)
        layers[str(i)]=dict(path=path.name,sha256=study.offload.sha(path),width=8,retained=4)
    (geom_root/'manifest.json').write_text(json.dumps(dict(schema='ablation32.geometry-bank.v1',
        projection_sites=study.PROJECTION_SITES,layers=layers)))
    stack,receipt=study.compile_all(tmp_path/'compiled',model_root,geom_root,arm='learned',expected_layers=2)
    reloaded,_=study.load_compiled(tmp_path/'compiled')
    for k,v in stack.state_dict().items():
        if isinstance(v,torch.Tensor):torch.testing.assert_close(v,reloaded.state_dict()[k],atol=0,rtol=0)
    compact=study.pack_preserving_geometry(reloaded)
    assert compact.geometry_mode=='fp32'
    assert all(b.qkv_shift.dtype==b.gate_up_shift.dtype==torch.float32 for b in compact.blocks)
    assert compact.blocks[0].basis.dtype==compact.bridges[0].weight.dtype==torch.float32
    model=packed.PackedResponseModel(compact,native.model.norm,native.model.embed_tokens.weight,
        native.model.rotary_emb,config=native.config,max_length=8)
    ids=torch.tensor([[1,3,6]])
    expected=model.next_logits(ids)
    proof=packed.export_packed_snapshot(model,tmp_path/'packed',identities={'compiled':receipt['checkpoint_sha256']})
    actual,replay=packed.load_packed_snapshot(tmp_path/'packed')
    torch.testing.assert_close(actual.next_logits(ids),expected,atol=0,rtol=0)
    assert replay['packed_state_exact'] and proof['identities']['compiled']==receipt['checkpoint_sha256']
