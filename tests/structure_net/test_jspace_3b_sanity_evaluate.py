"""Matched native-prefix/compact-suffix sanity scoring, with sealed confirmation."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_3b_sanity_evaluate as study
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack


@pytest.fixture
def tiny():
    torch.manual_seed(7039)
    config=Qwen2Config(hidden_size=8,intermediate_size=12,num_hidden_layers=4,
        num_attention_heads=2,num_key_value_heads=1,vocab_size=16,max_position_embeddings=32,
        attention_dropout=0.,tie_word_embeddings=True)
    config._attn_implementation="eager"
    native=Qwen2ForCausalLM(config).bfloat16().eval().requires_grad_(False)
    blocks=[]
    for layer in (2,3):
        basis=torch.linalg.qr(torch.randn(8,8)).Q[:,:4]
        block=AffineQwenBlock(native.model.layers[layer],basis,torch.randn(1,8)/20)
        for parameter in block.parameters():parameter.data=parameter.data.bfloat16()
        blocks.append(block)
    compact=CompactQwenStack(blocks,geometry_mode="bf16",pad_to_multiple=16)
    compact.eval().requires_grad_(False).pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16,strip_unused=True)
    norm=deepcopy(native.model.norm)
    with torch.no_grad():norm.weight.copy_(torch.linspace(.5,2.,8).bfloat16())
    cases=[dict(identity="short",family="boolq",input_ids=[1,4,3],choices=["yes","no"],answer=1,split="sanity"),
        dict(identity="long",family="arc",input_ids=[2,3,7,4,5],choices=["a","b","c","d"],answer=2,split="sanity")]
    return native,compact,norm,cases,[3,8,5,12]


def arguments(hidden,rotary):
    length=hidden.shape[1];positions=torch.arange(length)[None]
    blocked=torch.ones(length,length,dtype=torch.bool).triu(1)
    mask=torch.zeros(1,1,length,length,dtype=hidden.dtype).masked_fill(blocked,torch.finfo(hidden.dtype).min)
    return dict(position_ids=positions,position_embeddings=rotary(hidden,positions),
        attention_mask=mask,cache_position=positions[0],use_cache=False)


@torch.inference_mode()
def test_native_boundary_is_before_exact_cut_and_uses_last_prompt_token(tiny):
    native,_,_,cases,labels=tiny
    for case in cases:
        record,boundary=study.native_case(native,case,labels,cut=2)
        hidden=native.model.embed_tokens(torch.tensor([case["input_ids"]]));args=arguments(hidden,native.model.rotary_emb)
        for block in native.model.layers[:2]:hidden=block(hidden,**args)[0]
        assert boundary.device.type=="cpu" and boundary.dtype==torch.bfloat16
        assert torch.equal(boundary,hidden)
        for block in native.model.layers[2:]:hidden=block(hidden,**args)[0]
        expected=torch.nn.functional.linear(native.model.norm(hidden)[0,-1],
            native.model.embed_tokens.weight[labels[:len(case["choices"])]]).float()
        assert record["identity"]==case["identity"] and record["answer"]==case["answer"]
        torch.testing.assert_close(torch.tensor(record["logits"]),expected,atol=0,rtol=0)
        assert record["prediction"]==int(expected.argmax())


@torch.inference_mode()
def test_compact_scores_exact_boundary_with_recovered_norm_and_original_tied_readout(tiny):
    native,stack,norm,cases,labels=tiny;state=deepcopy(stack.state_dict())
    for case in cases:
        _,boundary=study.native_case(native,case,labels,cut=2)
        actual=study.compact_case(stack,norm,native.model.embed_tokens.weight,native.model.rotary_emb,boundary,case,labels)
        hidden=stack(boundary,**arguments(boundary,native.model.rotary_emb))[0]
        expected=torch.nn.functional.linear(norm(hidden)[0,-1],native.model.embed_tokens.weight[labels[:len(case["choices"])]]).float()
        wrong=torch.nn.functional.linear(native.model.norm(hidden)[0,-1],native.model.embed_tokens.weight[labels[:len(case["choices"])]]).float()
        torch.testing.assert_close(torch.tensor(actual["logits"]),expected,atol=0,rtol=0)
        assert not torch.equal(expected,wrong)  # The fixture catches silently using the original norm.
        assert actual["prediction"]==int(expected.argmax()) and len(actual["logits"])==len(case["choices"])
    for k,v in state.items():
        current=stack.state_dict()[k]
        assert torch.equal(current,v) if torch.is_tensor(v) else current==v


@pytest.mark.parametrize("defect",["dtype","shape","value","nonfinite"])
def test_native_boundary_control_rejects_off_by_one_or_precision_drift(defect):
    expected=torch.arange(24).reshape(1,3,8).bfloat16();actual=expected.clone()
    if defect=="dtype":actual=actual.float()
    elif defect=="shape":actual=actual[:,1:]
    elif defect=="value":actual[0,0,0]+=1
    else:actual[0,0,0]=float("nan")
    with pytest.raises(ValueError):study.boundary_control(actual,expected)


def test_boundary_control_accepts_exact_saved_native_prefix():
    expected=torch.arange(40).reshape(1,5,8).bfloat16()
    assert study.boundary_control(expected.clone(),expected)["passed"]


def paired_fixture():
    cases=[dict(identity=f"new-{i}",family=("arc","boolq","hellaswag","winogrande")[i//32],
        choices=["yes","no"],answer=0,split="sanity",input_ids=[1,2,3]) for i in range(128)]
    def rows(correct):
        return [dict(identity=c["identity"],family=c["family"],answer=0,prediction=0 if i in correct else 1,
            logits=[2.,0.] if i in correct else [0.,2.]) for i,c in enumerate(cases)]
    native=set(range(96));compact=(native-{0,33})|{100,101,110}
    return cases,rows(native),rows(compact)


@pytest.mark.parametrize("defect",["order","identity","family","gold","argmax","count","nonfinite"])
def test_paired_population_cannot_be_reordered_or_relabelled(defect):
    cases,native,compact=paired_fixture()
    if defect=="order":compact[0],compact[1]=compact[1],compact[0]
    elif defect=="identity":compact[0]["identity"]="other"
    elif defect=="family":compact[0]["family"]="other"
    elif defect=="gold":compact[0]["answer"]=1
    elif defect=="argmax":compact[0]["prediction"]=1-compact[0]["prediction"]
    elif defect=="count":compact.pop()
    else:compact[0]["logits"][0]=float("inf")
    with pytest.raises(ValueError):study.paired_summary(native,compact,cases)


def test_paired_quality_counts_and_scope_are_descriptive_partial_model_only():
    cases,native,compact=paired_fixture();result=study.paired_summary(native,compact,cases)
    overall=result["quality"]["overall"]
    assert (overall["native_correct"],overall["compact_correct"],overall["wins"],overall["losses"])==(96,97,3,2)
    assert overall["net_lost_correct"]==-1 and overall["accuracy_drop"]==-1/128
    assert result["maximum_net_lost_correct"]==6 and result["descriptive_screen_passed"]
    assert not result["confirmed"] and not result["full36"] and not result["speed_measured"]
    worse=deepcopy(native)
    for row in worse[:7]:row.update(prediction=1,logits=[0.,2.])
    assert not study.paired_summary(native,worse,cases)["descriptive_screen_passed"]


def phase_fixture(tmp_path):
    cases,native,compact=paired_fixture();study.base.write(tmp_path/"items.json",dict(items=cases))
    plan=dict(fingerprint="paired-plan",checkpoint_sha256=study.CHECKPOINT_SHA)
    for phase,rows in (("native",native),("compact",compact)):
        directory=tmp_path/phase;directory.mkdir()
        result=dict(status="completed",phase=phase,fingerprint=plan["fingerprint"],records=rows,count=128,
            correct=sum(r["prediction"]==r["answer"] for r in rows))
        if phase=="native":
            np.save(directory/"boundary_22.npy",np.arange(64,dtype=np.int16).reshape(8,8))
            result["boundary_control"]=dict(passed=True,bit_exact=True,max_abs=0.)
        else:
            result.update(strict_state_load=True,checkpoint_sha256=study.CHECKPOINT_SHA,
                boundary_sha256=study.base.sha(tmp_path/"native/boundary_22.npy"),
                inheritance=dict(passed=True,norm_exact=True,unchanged_layers=list(range(22,36))))
        study.base.write(directory/"result.json",result)
        manifest=dict(status="completed",phase=phase,fingerprint=plan["fingerprint"],
            allocation={"GPU-0370fdb0-fixture":11264},allocator=dict(physical_uuid="GPU-0370fdb0-fixture",
                allocator_limit_bytes=11008*1024**2,lease_bytes=11264*1024**2),peak_reserved_bytes=1024,
            artifacts={p.name:study.base.sha(p) for p in directory.iterdir() if p.is_file()})
        study.base.write(directory/"manifest.json",manifest)
    return plan


def rewrite(path,value):
    import json
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("defect",["checkpoint","boundary","norm","layers","count","correct","reservation","partial"])
def test_completed_compact_receipt_binds_exact_weights_prefix_norm_and_population(tmp_path,defect):
    plan=phase_fixture(tmp_path);assert study.checked(tmp_path,"compact",plan)["correct"]==97
    result=study.read(tmp_path/"compact/result.json");manifest=study.read(tmp_path/"compact/manifest.json")
    if defect=="checkpoint":result["checkpoint_sha256"]="0"*64
    elif defect=="boundary":result["boundary_sha256"]="0"*64
    elif defect=="norm":result["inheritance"]["norm_exact"]=False
    elif defect=="layers":result["inheritance"]["unchanged_layers"]=list(range(36))
    elif defect=="count":result["count"]=512
    elif defect=="correct":result["correct"]=128
    elif defect=="reservation":manifest["peak_reserved_bytes"]=12000*1024**2
    else:manifest["status"]="running"
    rewrite(tmp_path/"compact/result.json",result)
    manifest["artifacts"]["result.json"]=study.base.sha(tmp_path/"compact/result.json")
    rewrite(tmp_path/"compact/manifest.json",manifest)
    with pytest.raises(ValueError):study.checked(tmp_path,"compact",plan)


def test_finalization_preserves_fourteen_layer_scope_and_authenticates_both_phases(tmp_path,monkeypatch):
    plan=phase_fixture(tmp_path);monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(study,"verify",lambda *_:plan)
    value=study.finalize(tmp_path)
    assert value["compressed_layers"]==14 and not value["full36"] and not value["confirmed"]
    assert value["quality"]["overall"]["count"]==128
    assert set(value["phase_manifest_sha256"])=={"native","compact"}


def test_denied_lease_prevents_model_loading_or_any_cuda_initialization(tmp_path,monkeypatch):
    monkeypatch.setattr(study,"verify",lambda *_:{"fingerprint":"p"})
    def denied(*_):raise RuntimeError("reservation denied")
    monkeypatch.setattr(study,"wait_ready",denied)
    monkeypatch.setattr(torch.cuda,"init",lambda:pytest.fail("CUDA before lease"))
    monkeypatch.setattr(study.execution,"load_native_model",lambda *_a,**_k:pytest.fail("model before lease"))
    with pytest.raises(RuntimeError,match="reservation denied"):study.run(tmp_path,"native")


def test_compact_cannot_start_before_authenticated_native_boundary_phase(tmp_path,monkeypatch):
    monkeypatch.setattr(study,"verify",lambda *_:{"fingerprint":"p"})
    def denied(*_):raise ValueError("native phase incomplete")
    monkeypatch.setattr(study,"checked",denied)
    monkeypatch.setattr(study,"wait_ready",lambda *_:pytest.fail("reservation before native admission"))
    with pytest.raises(ValueError,match="native phase incomplete"):study.run(tmp_path,"compact")


def test_native_rejects_untied_readout_before_forward(tiny):
    native,_,_,cases,labels=tiny
    native.lm_head.weight=torch.nn.Parameter(native.lm_head.weight.detach().clone(),requires_grad=False)
    with pytest.raises(ValueError,match="tied"):study.native_case(native,cases[0],labels,cut=2)
