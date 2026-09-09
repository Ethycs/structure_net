"""Prefix architecture and readout-gate contracts independent of large weights."""
import pytest
import torch
from transformers import Qwen3Config

from experiments.structure_net import qwen_moba_prefix as q
from experiments.structure_net.moba_head_benchmark import sdpa_call


def test_layer_selection_and_full_coverage():
    assert [q.MoBASetting(layers="early5").enabled(i,1024) for i in range(6)] == [True]*5+[False]
    assert [q.MoBASetting(layers="last").enabled(i,1024) for i in range(6)] == [False]*5+[True]
    assert not q.MoBASetting(top_k=8).enabled(0,1024)
    assert not q.FULL.enabled(0,8192)
    with pytest.raises(ValueError):
        q.MoBASetting(top_k=1)


def test_prefix_matches_hf_graph_and_skips_only_sixth_mlp(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(15)
    monkeypatch.setattr(q,"flash_call",sdpa_call)
    config=Qwen3Config(vocab_size=256,hidden_size=32,intermediate_size=64,num_hidden_layers=6,
                       num_attention_heads=4,num_key_value_heads=2,head_dim=8)
    config._attn_implementation="sdpa"
    model=q.QwenPrefix(config).eval()
    ids=torch.arange(32)[None,:]
    calls=[0]*6
    def hook(index):
        def record(*args):
            calls[index]+=1
        return record
    handles=[layer.mlp.register_forward_hook(hook(i)) for i,layer in enumerate(model.layers)]
    with torch.no_grad():
        actual=model(ids,q.FULL,segment_size=8,capture=True)
    assert calls == [1]*5+[0]
    for handle in handles:
        handle.remove()
    with torch.no_grad():
        x=model.embed_tokens(ids)
        positions=torch.arange(32)[None,:]
        rope=model.rotary_emb(x,positions)
        for layer in model.layers[:5]:
            x=layer(x,attention_mask=None,position_ids=positions,position_embeddings=rope)[0]
        projected=model.layers[5].self_attn(model.layers[5].input_layernorm(x),position_embeddings=rope,attention_mask=None)[0]
    torch.testing.assert_close(actual["hidden_before_final"],x,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(actual["ov"],projected.reshape(1,4,8,32).mean(2),atol=1e-6,rtol=1e-5)
    assert actual["heads"].shape == (1,4,32)
    assert actual["ov"].dtype == torch.float32


def test_readout_identity_and_real_changes():
    torch.manual_seed(75)
    value={"heads":torch.randn(1,8,16),"ov":torch.randn(1,8,16),"changes":torch.arange(7).float()[None,:]}
    identity=q.readout_metrics(value,value)
    assert identity["heads_cosine_mean"] == pytest.approx(1)
    assert identity["ov_relative_l2"] == 0
    assert identity["change_pearson"] == pytest.approx(1)
    assert identity["top_quarter_change_f1"] == 1
    reversed_value={**value,"changes":value["changes"].flip(-1)}
    changed=q.readout_metrics(reversed_value,value)
    assert changed["change_pearson"] == pytest.approx(-1)
    assert changed["top_quarter_change_f1"] == 0


def test_feasibility_gate_cannot_substitute_one_endpoint():
    metrics={"heads_cosine_mean":.95,"ov_cosine_mean":.95,"change_pearson":.95,"top_quarter_change_f1":.90}
    assert q.feasibility_gate(metrics,1.05,True)
    assert not q.feasibility_gate(metrics,1.049,True)
    assert not q.feasibility_gate(metrics,2,False)
    for key in metrics:
        assert not q.feasibility_gate({**metrics,key:metrics[key]-.001},2,True)
    assert not q.feasibility_gate({**metrics,"change_pearson":None},2,True)


def test_tokenwise_mlp_chunking_preserves_prefix(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(q,"flash_call",sdpa_call)
    config=Qwen3Config(vocab_size=256,hidden_size=32,intermediate_size=64,num_hidden_layers=6,
                       num_attention_heads=4,num_key_value_heads=2,head_dim=8)
    model=q.QwenPrefix(config).eval()
    ids=torch.arange(32)[None,:]
    with torch.no_grad():
        full=model(ids,segment_size=8)
        model.mlp_chunk_tokens=8
        chunked=model(ids,segment_size=8)
    for key in full:
        torch.testing.assert_close(full[key],chunked[key],atol=2e-6,rtol=2e-5)


def test_head_chunking_preserves_gqa_order(monkeypatch):
    from experiments.structure_net.moba_head_benchmark import reference
    monkeypatch.setattr(q,"official_call",lambda query,key,value,block,k,cu:reference(query,key,value,block,k))
    torch.manual_seed(721)
    query=torch.randn(1,8,64,16)
    key,value=[torch.randn(1,2,64,16) for _ in range(2)]
    setting=q.MoBASetting(block_size=16)
    whole=q.moba_attention(query,key,value,setting,None,8)
    chunked=q.moba_attention(query,key,value,setting,None,4)
    torch.testing.assert_close(whole,chunked,atol=1e-6,rtol=1e-5)
    with pytest.raises(ValueError,match="GQA"):
        q.moba_attention(query,key,value,setting,None,3)
