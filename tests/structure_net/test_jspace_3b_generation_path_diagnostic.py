"""Fixed histories, EOS probabilities, and independent native suffix replay."""
from copy import deepcopy

import pytest
import torch

from experiments.structure_net import jspace_3b_generation_path_diagnostic as study
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from test_jspace_3b_ablation32_compile import fixture,context


def generation_fixture():
    canaries=[dict(id=name,input_ids=[1,2],output_ids=list(range(10,42)),stop_reason='token_limit')
        for name in ('addition','replace','invoice')]
    corpus=[dict(example_id='winogrande:revision:train:1336',input_ids=[1,3],output_ids=list(range(20,52)),stop_reason='token_limit')]
    student=dict(canaries={'records':canaries},corpus=corpus);native=deepcopy(student)
    for row in [*native['canaries']['records'],*native['corpus']]:row.update(output_ids=[4,5,99],stop_reason='eos')
    return student,native


def test_fixed_histories_use_exact_student_prefixes_and_native_eos_boundaries():
    student,native=generation_fixture();rows=study.histories(student,native)
    assert len(rows)==28
    assert {r['step']for r in rows if r['history']=='stored_student'}=={0,1,2,4,8,16}
    assert rows[2]['input_ids']==[1,2,10,11]and rows[2]['stored_next_token']==12
    completion=[r for r in rows if r['history']=='native_completion']
    assert len(completion)==4 and all(r['stored_next_token']==99 for r in completion)
    assert completion[0]['input_ids']==[1,2,4,5]
    native['canaries']['records'][0]['input_ids']=[1,8]
    with pytest.raises(ValueError,match='prompt IDs'):study.histories(student,native)


def test_eos_summary_uses_full_vocabulary_and_both_stop_tokens():
    logits=torch.tensor([-2.,0.,4.,1.,2.,3.])
    row=study.logit_summary(logits,[2,4],expected=4)
    assert row['greedy_token']==2 and row['greedy_is_eos']
    assert [r['rank']for r in row['eos']]==[1,3]
    assert row['expected']['rank']==3
    torch.testing.assert_close(torch.tensor(row['eos_mass']),logits.softmax(-1)[[2,4]].sum())
    with pytest.raises(ValueError):study.logit_summary(torch.tensor([float('nan'),1.]),[1])


def test_path_comparison_catches_eos_tail_changes_without_top1_change():
    a=torch.arange(20,dtype=torch.float32)/10;b=a.clone();b[1]-=5
    report=study.compare_logits(b,a)
    assert report['greedy_equal']and not report['bit_exact']and report['reference_to_actual_kl']>0
    assert report['maximum_absolute_error']==5
    assert study.compare_logits(a,a)['bit_exact']
    with pytest.raises(ValueError):study.compare_logits(a,a[:-1])


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_original_suffix_matches_complete_native_model(dtype):
    native,_,_=fixture(dtype);ids=torch.tensor([[1,3,6,8,4]])
    h,pos,mask,rope=context(native,ids)
    with torch.no_grad():
        boundary=native.model.layers[0](h,attention_mask=mask,position_ids=pos,
            position_embeddings=rope,cache_position=pos[0],use_cache=False)[0]
        actual=study.original_suffix(boundary,ids,[native.model.layers[1]],native.model.norm,native.model.rotary_emb)
        expected=native.model(input_ids=ids,use_cache=False).last_hidden_state
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


def test_actual_cpu_weight_prefix_stream_matches_native_reference_without_cuda(tmp_path):
    native,_,stack=fixture(torch.bfloat16)
    model_path=tmp_path/'native';native.save_pretrained(model_path)
    suffix=study.training.CompactQwenStack([stack.blocks[1]],geometry_mode='fp32',enabled=False)
    model=study.training.ResponseModel(suffix,native.model.norm,native.model.embed_tokens.weight,
        native.model.rotary_emb,max_length=8,expected_layers=2)
    ids=torch.tensor([[1,3,6,8,4]]);h,pos,mask,rope=context(native,ids)
    with torch.no_grad():
        expected=native.model.layers[0](h,attention_mask=mask,position_ids=pos,
            position_embeddings=rope,cache_position=pos[0],use_cache=False)[0]
        with study.packed.attach_prefix(model,model_path,mode='cpu')as receipt:
            actual=model.native_prefix(ids,torch.ones_like(ids,dtype=torch.bool))
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            cached=model.readout(model.hidden_forward(dict(input_ids=ids,boundary_hidden=expected))[:,-1])[0]
            torch.testing.assert_close(model.next_logits(ids),cached,atol=0,rtol=0)
        assert receipt['calls']==2 and model.native_prefix is None


def test_candidate_repack_reload_preserves_recorded_state_and_shares_embedding(tmp_path,monkeypatch):
    native,_,full=fixture(torch.bfloat16);model_path=tmp_path/'native';native.save_pretrained(model_path)
    cfg=study.recovery.base.builder.native._config(model_path)
    block=AffineQwenBlock(native.model.layers[1],full.blocks[1].basis,torch.zeros(1,8),parameter_dtype=torch.bfloat16)
    suffix=study.training.CompactQwenStack([block],geometry_mode='fp32',enabled=False)
    builder=study.recovery.base.builder
    reference=study.packed.PackedResponseModel(builder.packed_clone(suffix,device='cpu'),deepcopy(native.model.norm),
        native.model.embed_tokens.weight,deepcopy(native.model.rotary_emb),config=cfg,max_length=416)
    manifest=study.packed.export_packed_snapshot(reference,tmp_path/'source_export',identities={'fixture':True})
    # Keep the production packer/serializer/reloader and actual native suffix;
    # replace only old fit-artifact construction with this already fitted fixture.
    monkeypatch.setattr(builder,'build_stack',lambda *a,**kw:(deepcopy(suffix),deepcopy(native.model.norm),
        native.model.embed_tokens.weight.detach().clone(),deepcopy(native.model.rotary_emb),{'fixture':True}))
    p=dict(source_plan={'fits':{}},model={'path':str(model_path)},cut=1,checkpoint='fixture.pt',
        checkpoint_sha256='fixture',fingerprint='fixture',rotary_export=str(tmp_path/'source_export'),
        candidate={'packed_state_sha256':manifest['packed_state_sha256']})
    out=tmp_path/'run';out.mkdir()
    u,s,blocks,norm,rope,receipt=study.build_models(p,device='cpu',guard=lambda:None,out=out)
    assert u.embedding.data_ptr()==s.embedding.data_ptr()
    assert receipt['reload']['packed_state_exact']
    assert receipt['export']['packed_state_sha256']==manifest['packed_state_sha256']
    assert [b.self_attn.layer_idx for b in blocks]==[1]
