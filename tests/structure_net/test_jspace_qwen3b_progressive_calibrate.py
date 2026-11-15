from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_qwen3b_progressive_calibrate as port
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_progressive_calibration import GroupedTeacherCapture


def fixture(tmp_path):
    torch.manual_seed(831)
    config=Qwen2Config(hidden_size=8,intermediate_size=12,num_attention_heads=2,
        num_key_value_heads=1,num_hidden_layers=3,vocab_size=16,
        attention_dropout=0.,max_position_embeddings=32)
    config._attn_implementation="eager"
    model=Qwen2ForCausalLM(config).bfloat16().eval().requires_grad_(False)
    parent=AffineQwenBlock(model.model.layers[2],torch.eye(8)[:,:4],torch.zeros(1,8),trainable=False)
    parent.pack(dtype=torch.bfloat16,buffer_dtype=torch.bfloat16)
    model.model.layers[2]=parent
    cases=[dict(identity="train0",family="a",split="train",input_ids=[1,2,3,4],loss_positions=[1,3]),
           dict(identity="train1",family="b",split="train",input_ids=[3,1,2],loss_positions=[0,2]),
           dict(identity="dev",family="a",split="development",input_ids=[2,5,1],loss_positions=[1,2])]
    boundaries=[]
    for c in cases:
        with torch.no_grad():
            output=model.model(input_ids=torch.tensor([c["input_ids"]]),use_cache=False,output_hidden_states=True)
            boundaries.append(output.hidden_states[1][0].clone())
    path=tmp_path/"native_boundary.npy";np.save(path,torch.cat(boundaries).view(torch.int16).numpy())
    teacher=port.CachedPrefixTeacher(cases,path,1,model.model.layers[1],parent,
        model.model.norm,model.lm_head.weight,model.model.rotary_emb)
    return model,cases,teacher


def read_bits(path):return torch.from_numpy(np.load(path).copy()).view(torch.bfloat16)


def test_cached_teacher_exactly_replays_native_prefix_and_keeps_suffix_in_gradient(tmp_path):
    model,cases,teacher=fixture(tmp_path)
    fixed={k:v.clone() for k,v in teacher.state_dict().items() if isinstance(v,torch.Tensor)}
    for index,c in enumerate(cases):
        actual=teacher.forward_case(index)
        reference=model.model(input_ids=torch.tensor([c["input_ids"]]),use_cache=False).last_hidden_state
        assert torch.equal(actual,reference)
    captured={}
    def input_hook(_m,args):
        x=args[0].detach().requires_grad_();captured["input"]=x;return (x,*args[1:])
    handle=teacher.model.layers[1].register_forward_pre_hook(input_hook)
    output=teacher.forward_case(0)
    derivative=torch.autograd.grad(output[0,-1].float().square().sum(),captured["input"])[0]
    handle.remove()
    assert torch.isfinite(derivative).all() and derivative[0,0].abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert all(torch.equal(teacher.state_dict()[k],v) for k,v in fixed.items())
    assert not any("embed_tokens" in name for name,_ in teacher.named_modules())
    teacher.close()
    with pytest.raises(ValueError,match="closed"):teacher.forward_case(0)


def test_original_grouped_estimator_roundtrip_is_identical_to_full_prefix_capture(tmp_path):
    model,cases,teacher=fixture(tmp_path)
    root=tmp_path/"cached";events=[]
    result=port.capture(root,teacher,[0,1],teacher_fingerprint="fixed-parent",systems_only=True,progress=events.append)
    original=tmp_path/"full-prefix"
    with GroupedTeacherCapture(original,model,[1],cases,[0,1],teacher_fingerprint="fixed-parent",
            probe_groups=2,probes_per_group=8,activation_positions=16,position_capacity=416,
            seed=port.calibration_seed(1)) as recorder:
        for i,c in enumerate(cases):
            recorder.capture_case(i,lambda c=c:model.model(input_ids=torch.tensor([c["input_ids"]]),use_cache=False).last_hidden_state)
        expected=recorder.finish()
    a,b=result["blocks"]["1"],expected["blocks"]["1"]
    for key in ("boundary","mlp_calibration","mlp_development","mlp_jacobian_bf16"):
        assert torch.equal(read_bits(a[key]),read_bits(b[key]))
    actual_stats=torch.load(a["statistics"],weights_only=True)
    expected_stats=torch.load(b["statistics"],weights_only=True)
    for key in actual_stats:assert torch.equal(actual_stats[key],expected_stats[key])
    assert torch.equal(read_bits(result["accepted_teacher_logits"]),read_bits(expected["accepted_teacher_logits"]))
    wrapper=json.loads((root/"progressive_wrapper.json").read_text())
    assert wrapper["systems_only"] and wrapper["seed"]==2820861
    assert result["probe_groups"]==2 and result["probes_per_group"]==8
    assert len(events)==len(cases)
    teacher.close()


def test_real_basis_and_compiled_interface_preserve_frozen_native_and_fit_contract(tmp_path,monkeypatch):
    model,cases,teacher=fixture(tmp_path)
    root=tmp_path/"capture"
    port.capture(root,teacher,[0,1],teacher_fingerprint="fixed",systems_only=True)
    paths=port.finalize(root,tmp_path/"fit",rank=4,head_dim=2,progress=lambda _:None)
    directory=paths["1"]["directory"]
    before={k:v.clone() for k,v in teacher.model.layers[1].state_dict().items()}
    interface=port.capture_interface(directory,root,1,teacher.model.layers[1],teacher.rotary,progress=lambda _:None)
    fit=torch.load(paths["1"]["fit"],weights_only=True)
    assert fit["basis"].shape==(8,8) and fit["rank"]==4 and fit["head_dim"]==2
    assert all(torch.equal(v,teacher.model.layers[1].state_dict()[k]) for k,v in before.items())
    assert read_bits(interface["compiled_development"]).shape==(16,12)
    weight=torch.load(interface["output_weight"],weights_only=True)
    assert torch.equal(weight,weight.bfloat16().float())
    called=[]
    monkeypatch.setattr(port.corrected_fit,"fit_relation",lambda *a,**kw:called.append((a,kw)) or {"done":True})
    assert port.fit_relation(directory,progress=lambda _:None)=={"done":True}
    args,kw=called[0]
    assert args==(directory,)
    assert {k:kw[k] for k in ("seed","anchors","remove_count","pad_to_multiple","svd_rank","oversample","power_iterations","ridge","eligibility_nrmse","chunk_rows")}=={
        "seed":861,"anchors":24,"remove_count":1024,"pad_to_multiple":128,"svd_rank":64,
        "oversample":16,"power_iterations":1,"ridge":.01,"eligibility_nrmse":.5,"chunk_rows":256}
    teacher.close()


def test_scientific_capture_rejects_small_or_evaluation_sets_and_changed_teacher(tmp_path):
    _,_,teacher=fixture(tmp_path)
    with pytest.raises(ValueError,match="512/64"):
        port.capture(tmp_path/"small",teacher,[0,1],teacher_fingerprint="fixed")
    with pytest.raises(ValueError,match="training cases only"):
        port.capture(tmp_path/"devfit",teacher,[2],teacher_fingerprint="fixed",systems_only=True)
    teacher.model.layers[1].self_attn.q_proj.weight.requires_grad_(True)
    with pytest.raises(ValueError,match="parameters frozen"):
        port.capture(tmp_path/"trainable",teacher,[0,1],teacher_fingerprint="fixed",systems_only=True)
    teacher.close()


def test_alignment_and_native_bf16_inputs_are_required(tmp_path):
    model,cases,teacher=fixture(tmp_path);teacher.close()
    cases=deepcopy(cases);cases[0]["stop"]=100
    with pytest.raises(ValueError,match="offsets"):
        port.CachedPrefixTeacher(cases,teacher.boundary_path,1,model.model.layers[1],model.model.layers[2],
            model.model.norm,model.lm_head.weight,model.model.rotary_emb)
    cases[0].pop("stop");cases[-1]["split"]="evaluation"
    with pytest.raises(ValueError,match="train/development"):
        port.CachedPrefixTeacher(cases,teacher.boundary_path,1,model.model.layers[1],model.model.layers[2],
            model.model.norm,model.lm_head.weight,model.model.rotary_emb)


def test_seed_and_guards_block_cuda_before_initialization(tmp_path,monkeypatch):
    assert port.calibration_seed(28)==120861 and port.calibration_seed(0)==2920861
    for bad in (True,-1,30):
        with pytest.raises(ValueError):port.calibration_seed(bad)
    touched=[];monkeypatch.setattr(torch.cuda,"_lazy_init",lambda:touched.append(True))
    with pytest.raises(ValueError,match="reservation"):
        port.finalize(tmp_path/"missing",tmp_path/"fit",device="cuda")
    assert not touched


def test_metadata_tamper_rejected_before_fit(tmp_path):
    _,_,teacher=fixture(tmp_path);root=tmp_path/"capture"
    port.capture(root,teacher,[0,1],teacher_fingerprint="fixed",systems_only=True)
    path=root/"metadata.json";value=json.loads(path.read_text());value["seed"]+=1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="matching progressive"):
        port.finalize(root,tmp_path/"fit",rank=4,head_dim=2)
    teacher.close()
