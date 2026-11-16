"""Actual tiny packed evaluation, all-layer restore and honest measurements."""
from copy import deepcopy
import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen2Config,Qwen2ForCausalLM

from experiments.structure_net import jspace_3b_ablation32_evaluate as study
from experiments.structure_net import jspace_3b_ablation32_geometry as geometry
from experiments.structure_net import jspace_3b_ablation32_search as search


def fixture():
    torch.manual_seed(220422)
    cfg=Qwen2Config(hidden_size=8,intermediate_size=12,num_hidden_layers=2,num_attention_heads=2,
        num_key_value_heads=1,vocab_size=16,max_position_embeddings=512,tie_word_embeddings=True,
        attention_dropout=0.,eos_token_id=15,pad_token_id=0)
    cfg._attn_implementation='eager';native=Qwen2ForCausalLM(cfg).bfloat16().eval().requires_grad_(False)
    gs=[geometry.DiscardedSubspace(8,4,seed=220101+i).export(anchor=torch.randn(416,8)/20)for i in range(2)]
    stack=study.compiler.compile_stack(native.model.layers,gs,parameter_dtype=torch.bfloat16,expected_layers=2)
    return native,gs,stack


def test_final_all_block_restore_keeps_fp32_geometry_and_exact_packed_reload(tmp_path,monkeypatch):
    native,gs,stack=fixture();selected=[]
    for i,block in enumerate(stack.blocks):
        with torch.no_grad():block.mlp_output.add_((i+1)*.01)
        path=tmp_path/f'layer{i}.pt'
        torch.save(dict(schema='jspace.ablation32-stage-checkpoint.v1',layer=i,block=block.state_dict(),
            norm={'weight':native.model.norm.weight.detach()+.125}if i==1 else None),path)
        selected.append(dict(layer=i,checkpoint=str(path),sha256=study.sha(path)))
    result=dict(status='completed',arm='learned',layers=2,compiled_root=str(tmp_path),
        model_revision='native',data_fingerprint='data',selected_blocks=selected)
    original=fixture()[2]
    monkeypatch.setattr(study.compiler,'load_compiled',lambda _: (original,{}))
    monkeypatch.setattr(study.compiler.native,'_config',lambda _:native.config)
    monkeypatch.setattr(study.compiler.native,'load_readout_norm',lambda *a,**kw:(
        native.model.embed_tokens.weight,deepcopy(native.model.norm),deepcopy(native.model.rotary_emb)))
    p=dict(compiled_root=str(tmp_path),model={'path':'fixture','revision':'native'},
        data_fingerprint='data',arm='learned',arm_result=result)
    model=study.build_model(p)
    assert model.cut==0 and model.stack.packed and model.stack.stripped
    assert model.stack.geometry_mode=='fp32'
    assert all(b.basis.dtype==torch.float32 for b in model.stack.blocks)
    assert torch.equal(model.norm.weight,native.model.norm.weight+.125)
    reference=study.timing.packed_probe(model,[1,3,5])
    receipt=study.packed.export_packed_snapshot(model,tmp_path/'export',identities={'test':'selected'})
    fresh,reload=study.packed.load_packed_snapshot(tmp_path/'export')
    after=study.timing.packed_probe(fresh,[1,3,5],prefixes=[r['input_ids']for r in reference])
    assert study.timing.reload_parity(reference,after)['passed']
    assert receipt['packed_state_sha256']==reload['packed_state_sha256']


def test_selected_block_admission_rejects_missing_reordered_and_corrupted(tmp_path):
    path=tmp_path/'block.pt';torch.save(dict(schema='jspace.ablation32-stage-checkpoint.v1',layer=0,
        block={'weight':torch.ones(1)},norm=None),path)
    result=dict(status='completed',arm='random',layers=1,compiled_root=str(tmp_path),model_revision='n',
        data_fingerprint='d',selected_blocks=[dict(layer=0,checkpoint=str(path),sha256=study.sha(path))])
    kw=dict(arm='random',compiled_root=tmp_path,model_revision='n',data_fingerprint='d',layers=1)
    assert study.selected_blocks(result,**kw)==result['selected_blocks']
    with pytest.raises(ValueError,match='Every selected'):
        study.selected_blocks(dict(result,selected_blocks=[]),**kw)
    with pytest.raises(ValueError,match='matching'):
        study.selected_blocks(dict(result,data_fingerprint='other'),**kw)
    path.write_bytes(b'corruption')
    with pytest.raises(ValueError,match='changed'):study.selected_blocks(result,**kw)


def test_actual_projected_reference_comparison_is_diagnostic_and_token_bound():
    native,gs,stack=fixture();anchors=torch.stack([g['anchor']for g in gs]).float()
    projected=search.ProjectedModel(native,anchors,max_length=416)
    projected.discarded=[g['discarded'].float()for g in gs]
    data=search.module('distillation_data')
    record=data.ResponseRecord(example_id='case',group_id='g',split='train',task='summary',
        prompt_ids=(1,3),response_ids=(5,15),finish_reason='eos',eos_token_id=15,
        provenance=dict(source_uri='fixture',source_revision='1',tokenizer_sha256='t',chat_template_sha256='c',
            prompt_provenance='f',response_provenance='reference',assistant_boundary_verified=True))
    ids=torch.tensor([record.input_ids])
    with torch.no_grad():hidden=projected.hidden_forward({'input_ids':ids});logits=projected.readout(hidden[:,-1])
    ref=dict(checkpoint={'sha256':'fixture'},records=[dict(index=0,example_id='case',group_id='g',family='summary',
        input_ids=ids,normalized_hidden=hidden,last_prediction_position=ids.shape[1]-1,last_prediction_logits=logits)])
    model=study.packed.PackedResponseModel(study.compiler.pack_preserving_geometry(stack),native.model.norm,
        native.model.embed_tokens.weight,native.model.rotary_emb,config=native.config,max_length=416)
    proof=study.projection_comparison(model,ref,[record],[0])
    assert proof['diagnostic_only']and not proof['hard_numerical_gate']and proof['count']==1
    assert proof['records'][0]['hidden']['max_abs']>=0
    ref['records'][0]['input_ids']=ids+1
    with pytest.raises(ValueError,match='tokens'):study.projection_comparison(model,ref,[record],[0])


def test_actual_storage_counts_fp32_geometry_and_native_embedding_once(tmp_path):
    native,_,stack=fixture()
    model=study.packed.PackedResponseModel(study.compiler.pack_preserving_geometry(stack),native.model.norm,
        native.model.embed_tokens.weight,native.model.rotary_emb,config=native.config,max_length=416)
    state={k:v.detach().clone()for k,v in native.state_dict().items()}
    save_file(state,tmp_path/'model.safetensors')
    expected=sum(v.numel()*2 for k,v in state.items()if k!='lm_head.weight')
    assert study.native_header_bytes(tmp_path)==expected
    report=study.storage_report(model,tmp_path,{'serialized_bytes':123})
    assert report['packed_resident_tensor_bytes']==study.inventory.tensor_bytes(model)
    assert report['native_resident_tensor_bytes']==expected+study.inventory.tensor_bytes(model.rotary)
    assert report['geometry_precision']=='fp32'and report['tied_embedding_copies']==1


def test_matched_warm_timing_uses_identical_real_call_inputs():
    class Model:
        embedding=torch.zeros(16,8)
        def __init__(self,choice):self.choice,self.calls=choice,[]
        def next_logits(self,ids):
            self.calls.append(ids.tolist());v=torch.zeros(16);v[self.choice]=1;return v
    a,b=Model(2),Model(7)
    ra=study.matched_warm(a,[1,3],5);rb=study.matched_warm(b,[1,3],5)
    assert a.calls==b.calls
    assert a.calls[:7]==[[[1,3]]]*7 and a.calls[7:]==[[[1,3,5]]]*7
    assert ra['identical_timed_inputs']and rb['fixed_extension_token']==5


def test_mc_adapter_forces_real_embedding():
    class Dataset:
        training=[];development=[0]
        def batch(self,indices):return dict(input_ids=torch.tensor([[1,2]]),boundary_hidden=torch.ones(1,2,8))
    adapter=study.EmbeddingOnlyMC(Dataset())
    assert 'boundary_hidden'not in adapter.batch([0])and adapter.development==[0]
