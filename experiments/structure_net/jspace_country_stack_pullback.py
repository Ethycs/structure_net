"""Exact linear-readout pullback through a real tiny Qwen attention module, on CPU."""
import json
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from .jspace_country_stack import DEFAULT_ROOT, PARENT, verify
from .jspace_causal_components import cpu_only
from .jspace_relation_study import file_hash, write_json


def check_pullback():
    cpu_only();torch.set_num_threads(2);torch.manual_seed(713)
    config=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=48,num_hidden_layers=1,
                       num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
    config._attn_implementation="eager"
    model=Qwen2ForCausalLM(config).double().eval().requires_grad_(False)
    module=model.model.layers[0].self_attn
    state={}
    def before(m,args,kwargs):state["input"]=kwargs["hidden_states"]
    def after(m,args,output):state["output"],state["attention"]=output[:2]
    handles=[module.register_forward_pre_hook(before,with_kwargs=True),module.register_forward_hook(after)]
    embeddings=torch.randn(2,9,32,dtype=torch.double,requires_grad=True)
    try:
        model.model(inputs_embeds=embeddings,use_cache=False,output_attentions=True)
    finally:
        for handle in handles:handle.remove()
    h,attention=state["input"],state["attention"]
    qheads,kvheads,dim,rank=4,2,8,3
    readout=torch.randn(rank,32,dtype=torch.double)
    original=state["output"]@readout.T
    value_weight=module.v_proj.weight.reshape(kvheads,dim,32)
    value_bias=module.v_proj.bias.reshape(kvheads,dim)
    out_blocks=module.o_proj.weight.reshape(32,qheads,dim)
    fused=torch.zeros_like(original)
    for head in range(qheads):
        pullback=readout@out_blocks[:,head,:]
        kv=head//(qheads//kvheads)
        fused_weight=pullback@value_weight[kv]
        fused_bias=pullback@value_bias[kv]
        small_values=h@fused_weight.T+fused_bias
        fused+=attention[:,head]@small_values
    if module.o_proj.bias is not None:fused+=module.o_proj.bias@readout.T
    polynomial=lambda z:(z[...,0]**2+z[...,1]*z[...,2]+.3*z[...,0]).sum()
    original_gradient=torch.autograd.grad(polynomial(original),embeddings,retain_graph=True)[0]
    fused_gradient=torch.autograd.grad(polynomial(fused),embeddings)[0]
    error=float((original-fused).abs().max())
    derivative_error=float((original_gradient-fused_gradient).abs().max())
    if error>1e-10 or derivative_error>1e-10:
        raise ValueError("Attention readout pullback failed value/derivative equivalence")
    return {"scope":"actual_tiny_Qwen_attention_algebra_control_not_a_3B_speed_or_accuracy_test",
            "dtype":"float64","query_heads":qheads,"kv_heads":kvheads,"head_width":dim,"readout_rank":rank,
            "value_max_absolute_error":error,"input_gradient_max_absolute_error":derivative_error,
            "uses_actual_RoPE_causal_attention_weights":True,"qk_and_softmax_unchanged":True,
            "polynomial_observable":"z0**2 + z1*z2 + 0.3*z0", "passed":True}


def main(root=DEFAULT_ROOT):
    cpu_only();verify(root)
    checked=check_pullback()
    parent=json.loads((PARENT/"plan.json").read_text())
    config_path=Path(parent["model"]["path"])/"config.json"
    cfg=json.loads(config_path.read_text())
    width=cfg["hidden_size"];qheads=cfg["num_attention_heads"];kvheads=cfg["num_key_value_heads"]
    dim=cfg.get("head_dim",width//qheads);rank=8
    normal=width*kvheads*dim+width*qheads*dim
    fused=width*qheads*rank
    value={"status":"completed","confirmed":False,"control":checked,
           "attention_readout_identity":"C o = sum_h A_h(X) applied to values with fused projection C W_O,h W_V,h (column convention)",
           "model_config_sha256":file_hash(config_path),"source_sha256":file_hash(__file__),
           "3B_config_arithmetic_only":{"hidden_size":width,"query_heads":qheads,"kv_heads":kvheads,"head_dim":dim,"rank":rank,
               "original_V_plus_O_macs_per_token":normal,"fused_readout_value_macs_per_token":fused,"suboperation_ratio":normal/fused,
               "original_attention_value_channels":qheads*dim,"fused_attention_value_channels":qheads*rank,
               "original_GQA_value_channels":kvheads*dim,"fused_value_channels":qheads*rank},
           "limitations":["no_3B_fused_kernel_run","no_runtime_speedup_measured","remaining_transformer_needs_full_state_unless_closure_is_established",
                          "finite_country_label_basis_does_not_supply_a_sufficient_continuous_attention_state",
                          "linear_projection_identity_does_not_itself_require_Groebner_methods"]}
    write_json(root/"attention_pullback_audit.json",value)
    return value


if __name__=="__main__":print(json.dumps(main(),allow_nan=False))
