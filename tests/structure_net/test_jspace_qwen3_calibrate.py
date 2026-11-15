from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from experiments.structure_net import jspace_qwen3_calibrate as cal
from experiments.structure_net.jspace_qwen3_block import Qwen3AffineBlock


def fixture():
    torch.manual_seed(871)
    cfg = Qwen3Config(hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        vocab_size=32, max_position_embeddings=32, attention_dropout=0., tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    model = Qwen3ForCausalLM(cfg).bfloat16().eval().requires_grad_(False)
    config = cal.CalibrationConfig(families=("a","b"),cases_per_family=2,
        activation_positions=3,position_capacity=8,rank=8,krylov_block_width=8,
        head_dim=2,anchors=2,remove_count=3,svd_rank=6,oversample=4,chunk_rows=4)
    cases = []
    for family in config.families:
        for n in range(4):
            length = 4+n%2
            ids = torch.randint(0,32,(length,)).tolist()
            cases.append(dict(identity=f"{family}/{n}",family=family,
                split="train" if n<3 else "development",input_ids=ids,
                choices=["x","y"] if family=="a" else ["w","x","y","z"],
                answer=n%2,loss_positions=[0,1,length-2,length-1]))
    banks = [model.model.embed_tokens(torch.tensor(c["input_ids"])).detach().clone() for c in cases]
    def read(index,device): return banks[index].to(device)
    def call(block,x,case):
        length=x.shape[1];positions=torch.arange(length,device=x.device)[None]
        mask=torch.full((1,1,length,length),torch.finfo(x.dtype).min,device=x.device,dtype=x.dtype).triu(1)
        output=block(x,attention_mask=mask,position_ids=positions,
            position_embeddings=model.model.rotary_emb(x,positions),cache_position=positions[0])
        return output[0] if isinstance(output,tuple) else output
    def forward(x,case):
        for block in model.model.layers: x=call(block,x,case)
        return model.model.norm(x)
    modules=(model.model.layers[1],model.model.norm,model.model.rotary_emb)
    return model,config,cases,banks,read,call,forward,modules


def capture(tmp_path,name="capture",*,development_shift=0):
    model,config,cases,banks,read,call,forward,modules=fixture()
    for i,c in enumerate(cases):
        if c["split"]=="development": banks[i].add_(development_shift)
    root=tmp_path/name
    meta=cal.calibrate(cases,read,model.model.layers[0],forward,model.lm_head.weight[:4],root,
        teacher_modules=modules,teacher_fingerprint="frozen-tiny-native",config=config)
    return root,meta,(model,config,cases,banks,read,call,forward,modules)


def assert_nested_equal(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: assert_nested_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_nested_equal(x,y)
    else:assert a==b


def test_train_only_selection_is_balanced_disjoint_and_deterministic():
    _,config,cases,*_=fixture()
    selected=cal.select_calibration(cases,config)
    assert selected==cal.select_calibration(cases,config)
    assert len(selected)==4
    assert [cases[i]["family"] for i in selected].count("a")==2
    assert all(cases[i]["split"]=="train" for i in selected)
    bad=deepcopy(cases);bad[-1]["split"]="evaluation"
    with pytest.raises(ValueError,match="Evaluation"):cal.select_calibration(bad,config)
    bad=deepcopy(cases);bad[-1]["identity"]=bad[0]["identity"]
    with pytest.raises(ValueError,match="unique and disjoint"):cal.select_calibration(bad,config)
    with pytest.raises(ValueError):replace(config,ridge=float("nan"))


def test_post_norm_pair_mask_derivative_sums_queries_tokens_and_coordinates_before_square():
    torch.manual_seed(412)
    q=torch.randn(1,5,4,4);k=torch.randn(1,5,2,4)
    gq=torch.randn_like(q);gk=torch.randn_like(k)
    mask=torch.ones(2,2,requires_grad=True)
    columns=torch.cat((mask,mask),-1)
    objective=(q*columns.repeat_interleave(2,0)*gq).sum()+(k*columns*gk).sum()
    expected=torch.autograd.grad(objective,mask)[0]
    actual=cal.pair_mask_derivative(q,k,gq,gk,query_heads=4,kv_heads=2,head_dim=4)
    torch.testing.assert_close(actual,expected)
    # Shared mask contributions can cancel: squaring each token first is wrong.
    q=torch.ones(1,2,1,2);gq=torch.tensor([[[[1.,0.]], [[-1.,0.]]]])
    zero=torch.zeros_like(q)
    assert cal.pair_mask_derivative(q,zero,gq,zero,query_heads=1,kv_heads=1,head_dim=2).item()==0


def test_real_bf16_capture_matches_input_jacobian_and_all_training_means(tmp_path):
    root,meta,args=capture(tmp_path)
    model,config,cases,banks,read,call,forward,modules=args
    stats=torch.load(root/"statistics.pt",weights_only=True)
    gram=torch.zeros(16,16)
    for index in meta["calibration_indices"]:
        case=cases[index];x=banks[index][None].clone().requires_grad_()
        logits=F.linear(forward(x,case)[0,-1],model.lm_head.weight[:len(case["choices"])]).float()
        others=[j for j in range(len(case["choices"])) if j!=case["answer"]]
        grad=torch.autograd.grad(logits[case["answer"]]-logits[others].mean(),x)[0][0].float()
        gram+=grad.T@grad
    assert torch.equal(stats["input_gram"],gram/4)
    alltrain=torch.cat([banks[i].float() for i,c in enumerate(cases) if c["split"]=="train"])
    torch.testing.assert_close(stats["global_mean"],alltrain.mean(0),atol=1e-9,rtol=1e-6)
    torch.testing.assert_close(stats["means"][0],torch.stack([banks[i][0].float() for i,c in enumerate(cases) if c["split"]=="train"]).mean(0))
    assert torch.equal(stats["means"][7],stats["global_mean"])
    jac=np.load(root/"mlp_jacobian.npy")
    assert jac.dtype==np.float32 and len(jac)==sum(len(cases[i]["input_ids"]) for i in meta["calibration_indices"])
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert not model.model.layers[0]._forward_pre_hooks
    assert not model.model.layers[0].self_attn.q_norm._forward_hooks
    assert cal.verify_capture(root)["teacher_fingerprint"]=="frozen-tiny-native"


def fitted(tmp_path,name,shift):
    root,meta,args=capture(tmp_path,name,development_shift=shift)
    model,config,cases,banks,read,call,*_=args
    target=tmp_path/(name+"_fit")
    fit=cal.fit_basis(root,target)
    block=Qwen3AffineBlock(model.model.layers[0],fit["basis"],fit["means"],
        trainable=False,head_projection={k:fit[k] for k in ("rope_pairs","value_basis")}).eval()
    interface=tmp_path/(name+"_interface")
    cal.capture_compiled_development(root,target,read,block,call,interface)
    result=cal.fit_relations(root,target,interface)
    return fit,torch.load(target/"relation.pt",weights_only=True),result,root,target,interface


def test_real_compiled_interface_and_relations_never_fit_development(tmp_path):
    a=fitted(tmp_path,"ordinary",0)
    b=fitted(tmp_path,"changed_development",2)
    for key in ("basis","means","rope_pairs","value_basis"):assert torch.equal(a[0][key],b[0][key])
    for key in ("anchors","mean","scale","coefficients","removed"):
        assert_nested_equal(a[1][key],b[1][key])
    assert a[2]["fixed_anchor_width"]==2 and a[2]["removed_channels"]==3
    assert a[2]["jacobian_rows_read"]>0
    assert a[2]["metrics"]["candidates"] != b[2]["metrics"]["candidates"]
    assert a[2]["runtime_padding"]["padded_width"]%128==0
    assert a[2]["anchor_selection"]["rank_admission"]["rule_id"]=="actual_sketch_svd_shape_fp32"


def test_last_block_margin_keeps_zero_context_derivatives_explicit(tmp_path):
    model,config,cases,banks,read,call,forward,modules=fixture()
    for i,c in enumerate(cases):banks[i]=call(model.model.layers[0],banks[i][None],c)[0].detach()
    last=model.model.layers[1]
    root=tmp_path/"last"
    meta=cal.calibrate(cases,read,last,lambda x,c:model.model.norm(call(last,x,c)),
        model.lm_head.weight[:4],root,teacher_modules=(model.model.norm,model.model.rotary_emb),
        teacher_fingerprint="last",config=config)
    jac=np.load(root/"mlp_jacobian.npy")
    for row in meta["jacobian_case_ranges"]:
        assert np.count_nonzero(jac[row["start"]:row["stop"]-1])==0
        assert np.count_nonzero(jac[row["stop"]-1])>0


def test_fp32_stream_has_owned_bounded_rows_and_rejects_wrong_precision(tmp_path):
    path=tmp_path/"rows.npy";expected=np.arange(48,dtype=np.float32).reshape(8,6)/12345
    np.save(path,expected)
    with cal.StreamedFloat32Rows(path,max_rows=3) as rows:
        a=rows[1:3];assert torch.equal(a,torch.from_numpy(expected[1:3]))
        a.fill_(999);assert torch.equal(rows[1:3],torch.from_numpy(expected[1:3]))
        with pytest.raises(ValueError,match="bounded"):rows[:4]
        with pytest.raises(ValueError,match="Contiguous"):rows[::2]
    np.save(path,expected.astype(np.float16))
    with pytest.raises(ValueError,match="FP32"):cal.StreamedFloat32Rows(path)


def test_tampered_capture_stops_before_basis_fit(tmp_path):
    root,_,_=capture(tmp_path)
    with (root/"statistics.pt").open("ab") as stream:stream.write(b"changed")
    with pytest.raises(ValueError,match="Changed calibration"):cal.fit_basis(root,tmp_path/"fit")


def test_requires_frozen_normal_teacher_and_reservation_before_cuda(tmp_path,monkeypatch):
    model,config,cases,banks,read,call,forward,modules=fixture()
    kwargs=dict(teacher_modules=modules,teacher_fingerprint="teacher",config=config)
    model.model.layers[0].self_attn.q_proj.weight.requires_grad_(True)
    with pytest.raises(ValueError,match="frozen and eval"):
        cal.calibrate(cases,read,model.model.layers[0],forward,model.lm_head.weight[:4],tmp_path/"bad",**kwargs)
    model.requires_grad_(False)
    with torch.inference_mode(),pytest.raises(ValueError,match="inference_mode"):
        cal.calibrate(cases,read,model.model.layers[0],forward,model.lm_head.weight[:4],tmp_path/"infer",**kwargs)
    touched=[];monkeypatch.setattr(torch.cuda,"_lazy_init",lambda:touched.append(True))
    with pytest.raises(ValueError,match="reservation guard"):
        cal.fit_basis(tmp_path/"missing",tmp_path/"fit",device="cuda")
    assert not touched
