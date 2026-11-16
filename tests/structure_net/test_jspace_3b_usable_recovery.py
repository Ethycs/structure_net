"""Real saved-state restoration and one-pass updates; generation-first selection."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RMSNorm,Qwen2RotaryEmbedding

from experiments.structure_net import jspace_3b_usable_recovery as study


class Dataset:
    def __init__(self):
        self.training=[0,1];self.development=[2]
        provenance=dict(source_uri='fixture',source_revision='1',tokenizer_sha256='t',chat_template_sha256='c',
            prompt_provenance='fixture',response_provenance='reference',assistant_boundary_verified=True)
        dm=study.base.training.module('distillation_data')
        self.records=[dm.ResponseRecord(example_id=f'case{i}',group_id=f'g{i}',split='train'if i<2 else'development',
            task='fixture',prompt_ids=(1,3+i),response_ids=(6,4)if i%2==0 else(8,9,4),provenance=provenance,
            finish_reason='eos',eos_token_id=4)for i in range(3)]
        self.teacher=[torch.randn(len(r.input_ids),8).bfloat16()for r in self.records]
        self.boundary=[torch.randn(len(r.input_ids),8).bfloat16()for r in self.records]
    def batch(self,indices):
        i,=indices;r=self.records[i]
        batch=study.base.training.module('distillation_data').response_batch([r],pad_token_id=0,max_length=8)
        return dict(batch,teacher_hidden=self.teacher[i][None],boundary_hidden=self.boundary[i][None],
            teacher_input_ids=batch['input_ids'].clone(),teacher_record_ids=[r.example_id])


def fixture():
    torch.manual_seed(18273)
    cfg=Qwen2Config(hidden_size=8,intermediate_size=12,num_hidden_layers=4,num_attention_heads=2,
        num_key_value_heads=1,max_position_embeddings=32,vocab_size=16,tie_word_embeddings=True,attention_dropout=0.)
    cfg._attn_implementation='eager';blocks=[]
    for layer in range(1,4):
        original=Qwen2DecoderLayer(cfg,layer).bfloat16().eval()
        basis=torch.linalg.qr(torch.randn(8,8)).Q[:,:4]
        block=study.AffineQwenBlock(original,basis,torch.randn(8,8)/10,trainable=True,parameter_dtype=torch.bfloat16)
        block.compile_polynomial(dict(anchors=[0,1],mean=torch.zeros(12),scale=torch.ones(12),
            coefficients={'2':torch.randn(12,6)/20}),[10,11])
        for p in block.parameters():p.data=p.data.bfloat16()
        blocks.append(block)
    stack=study.CompactQwenStack(blocks,geometry_mode='bf16',pad_to_multiple=0,checkpoint_blocks=True)
    return cfg,stack,Qwen2RMSNorm(8).bfloat16(),torch.randn(16,8).bfloat16(),Qwen2RotaryEmbedding(cfg).eval()


def test_raw_restore_uses_saved_tensors_and_preserves_every_geometry_bit(monkeypatch):
    cfg,stack,norm,E,rotary=fixture()
    raw=study.base.offload.recursive_cpu(stack.state_dict())
    packed=study.base.packed_source.PackedResponseModel(study.base.builder.packed_clone(stack),norm,E,rotary,config=cfg,max_length=8)
    description=study.base.packed_source._describe(packed)
    monkeypatch.setattr(study.base.builder.native,'load_native_block',lambda *a,**kw:pytest.fail('Refolded native block'))
    fresh=study.restore_unpacked(raw,description)
    assert study.base.builder.state_equal(fresh.state_dict(),raw)
    assert not fresh.packed and not fresh.stripped and fresh.layer_indices==(1,2,3)
    a=study.base.training.ResponseModel(stack,norm,E,rotary,max_length=8,expected_layers=4)
    b=study.base.training.ResponseModel(fresh,deepcopy(norm),E,deepcopy(rotary),max_length=8,expected_layers=4)
    batch=Dataset().batch([0])
    with torch.no_grad():torch.testing.assert_close(a.hidden_forward(batch),b.hidden_forward(batch),rtol=0,atol=0)


def test_actual_three_block_pilot_all_response_gradients_master_casts_and_reload(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:pytest.fail('CPU test initialized CUDA'))
    cfg,stack,norm,E,rotary=fixture();dataset=Dataset()
    config=study.base.training.ResponseConfig(learning_rate=2e-5,max_supervised_tokens=5,max_length=8,
        token_chunk_size=1,optimizer_chunk_bytes=64)
    trainer=study.base.training.ResponseTrainer(stack,norm,E,rotary,dataset,teacher_id='native',config=config,
        expected_layers=4,host_reserve_bytes=0)
    try:
        receipt=study.pilot_updates(trainer,[[0],[1]])
        assert receipt['finite_gradients']and receipt['master_cast_exact']
        assert receipt['change']['bf16_changed_by_block']==[True]*3
        assert trainer.supervised_tokens==5 and trainer.visited=={0,1}
        assert study.base.offload.fixed_state(stack,E,rotary)==trainer.fixed
        cp=trainer.save_checkpoint(tmp_path/'saved.pt',identities={'fixture':True})
        value=torch.load(cp['path'],map_location='cpu',weights_only=True)
        model=study.base.packed_source.PackedResponseModel(study.base.builder.packed_clone(stack),
            deepcopy(norm),E,deepcopy(rotary),config=cfg,max_length=8)
        description=study.base.packed_source._describe(model)
        restored=study.restore_unpacked(value['stack'],description)
        assert study.base.builder.state_equal(restored.state_dict(),value['stack'])
        batch=dataset.batch([0])
        with torch.no_grad():before=model.readout(model.hidden_forward(batch)[:,-1])
        study.base.packed_source.export_packed_snapshot(model,tmp_path/'export',identities={'fixture':True})
        fresh,_=study.base.packed_source.load_packed_snapshot(tmp_path/'export')
        with torch.no_grad():after=fresh.readout(fresh.hidden_forward(batch)[:,-1])
        assert torch.equal(before,after)
        with pytest.raises(ValueError,match='unvisited'):trainer.train_window([0])
    finally:trainer.close()


def test_generation_quality_ranks_before_nll_or_mc_and_missing_stays_unknown():
    a=dict(label='a',evidence_available=True,qualified=True,useful_completed_tasks=10,reliable_generations=12,nll=2.6,update=71,mc_correct=196)
    b=dict(a,label='b',useful_completed_tasks=9,nll=2.0,mc_correct=198)
    assert study.select_quality([b,a])['selected']['label']=='a'
    failed=dict(a,label='failed',qualified=False,useful_completed_tasks=15)
    assert study.select_quality([failed,a])['selected']['label']=='a'
    result=study.select_quality([dict(a,evidence_available=False,qualified=None)])
    assert result['qualified']is None and result['selected']is None
    result=study.select_quality([failed])
    assert result['status']=='exploratory_unqualified'and not result['qualified']
    early=dict(a,label='early',update=0)
    assert study.select_quality([a,early])['selected']['label']=='early'


def test_reviewed_diagnostic_is_exactly_bound_and_not_a_numerical_gate(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_3b_generation_path_diagnostic as diagnostic
    (tmp_path/'run').mkdir();(tmp_path/'run/result.json').write_text('{}')
    (tmp_path/'plan.json').write_text(json.dumps({'checkpoint_sha256':'c'}))
    result=dict(status='completed',torch_unpacked_greedy_agreement=.6,systems={key:True for key in
        ('packed_reload_exact','generation_entrypoint_exact','shared_original_prefix','shared_original_tied_embedding')})
    monkeypatch.setattr(diagnostic,'check',lambda _:result)
    receipt=dict(status='approved',reviewed=True,checkpoint_sha256='c',backend='packed_torch',
        diagnostic_result_sha256=study.sha(tmp_path/'run/result.json'))
    (tmp_path/'training_admission.json').write_text(json.dumps(receipt))
    p=dict(diagnostic_root=str(tmp_path),checkpoint_sha256='c',backend='packed_torch')
    admitted=study.diagnostic_admission(p)
    assert admitted['admission_sha256']==study.sha(tmp_path/'training_admission.json')
    (tmp_path/'run/result.json').write_text('{"different":true}')
    with pytest.raises(ValueError,match='matching exact'):study.diagnostic_admission(p)


def test_resource_profiles_match_actual_reserved_cards():
    assert study.PROFILES['plain2060']==dict(prefix='GPU-8cf440df-',allocation=3584,limit=3328,prefix_mode='cpu')
    assert study.PROFILES['3060']==dict(prefix='GPU-0370fdb0-',allocation=11264,limit=11008,prefix_mode='resident')
