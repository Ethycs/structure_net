"""Packing arithmetic and actual Phi3 selective-loading controls; CPU only."""
import json

import numpy as np
import pytest
from safetensors.torch import save_file
import torch
from transformers import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer

from experiments.structure_net import jspace_phi_weights as weights


def packed(codes, zero_points, scales):
    """Independent uint32 scalar-bit insertion, matching published GPTQ pack."""
    codes = np.asarray(codes, dtype=np.uint32)
    zero_points = np.asarray(zero_points, dtype=np.uint32)
    n, m = codes.shape
    qw = np.zeros((n//8, m), dtype=np.uint32)
    qz = np.zeros((n//128, m//8), dtype=np.uint32)
    for i in range(n):
        qw[i//8] |= codes[i] << np.uint32(4*(i % 8))
    for j in range(m):
        qz[:, j//8] |= (zero_points[:, j]-1) << np.uint32(4*(j % 8))
    return (torch.from_numpy(qw.view(np.int32)), scales,
            torch.from_numpy(qz.view(np.int32)), torch.arange(n, dtype=torch.int32)//128)


def official_eager_reference(qw, scales, qz, groups):
    # Independent unchunked shape sequence from AutoGPTQ qlinear_cuda.forward.
    shifts = torch.tensor(list(range(0,32,4)), dtype=torch.int32).unsqueeze(0)
    zeros = torch.bitwise_right_shift(qz.unsqueeze(2).expand(-1,-1,8), shifts.unsqueeze(0)).to(torch.int8)
    zeros = (torch.bitwise_and(zeros,15)+1).reshape(scales.shape)
    codes = torch.bitwise_right_shift(qw.unsqueeze(1).expand(-1,8,-1), shifts.unsqueeze(-1)).to(torch.int8)
    codes = torch.bitwise_and(codes,15).reshape(qw.shape[0]*8,qw.shape[1])
    return (scales[groups.long()] * (codes-zeros[groups.long()])).T.contiguous()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_known_codes_zero_offset_signed_words_and_chunking(dtype):
    rng = np.random.default_rng(42)
    codes = rng.integers(0,16,(256,24),dtype=np.uint32)
    zeros = np.tile(np.array([1,8,16],dtype=np.uint32), (2,8))
    scales = torch.linspace(-.2,.2,48,dtype=torch.float16).reshape(2,24)
    scales[0,0] = 0  # Literal signed/zero scales are allowed by the format.
    tensors = packed(codes,zeros,scales)
    assert (tensors[0] < 0).any() and (tensors[2] < 0).any()
    expected = torch.from_numpy(codes.astype(np.int32)-zeros[np.arange(256)//128].astype(np.int32))
    expected = (expected.float()*scales.float()[torch.arange(256)//128]).half().T.to(dtype)
    actual = weights.dequantize_gptq(*tensors,dtype=dtype,chunk_rows=24)
    assert actual.is_contiguous() and torch.equal(actual,expected)
    assert torch.equal(actual,official_eager_reference(*tensors).to(dtype))
    assert torch.equal(actual,weights.dequantize_gptq(*tensors,dtype=dtype,chunk_rows=128))
    x = torch.linspace(-.1,.1,256).reshape(1,-1)
    torch.testing.assert_close(torch.nn.functional.linear(x,actual.float()),
        torch.nn.functional.linear(x,expected.float().contiguous()),rtol=0,atol=0)


@pytest.mark.parametrize("mutation", ["groups","scale","shape","dtype","chunk"])
def test_malformed_quantization_fails(mutation):
    t = list(packed(np.zeros((128,16),np.uint32),np.ones((1,16),np.uint32),torch.ones(1,16).half()))
    kwargs = {}
    if mutation == "groups": t[3][0] = 1
    if mutation == "scale": t[1][0,0] = float("nan")
    if mutation == "shape": t[2] = t[2][:,:1]
    if mutation == "dtype": t[0] = t[0].long()
    if mutation == "chunk": kwargs["chunk_rows"] = 3
    with pytest.raises(ValueError): weights.dequantize_gptq(*t,**kwargs)


@pytest.fixture
def tiny_source(tmp_path):
    config = Phi3Config(hidden_size=128,intermediate_size=256,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,vocab_size=64,max_position_embeddings=32,
        original_max_position_embeddings=32,bos_token_id=0,eos_token_id=1,pad_token_id=1,
        tie_word_embeddings=False,rope_scaling=None,resid_pdrop=0.,embd_pdrop=0.,attention_dropout=0.)
    config._attn_implementation = "eager"
    raw = config.to_dict()
    raw["quantization_config"] = dict(quant_method="gptq",bits=4,group_size=128,desc_act=False,sym=True)
    (tmp_path/"config.json").write_text(json.dumps(raw))
    index, expected = {}, {}
    rng = np.random.default_rng(117)
    for layer in range(2):
        with torch.device("meta"): native = Phi3DecoderLayer(config,layer_idx=layer)
        tensors, decoded = {}, {}
        for key,value in native.state_dict().items():
            name = f"model.layers.{layer}.{key}"
            if key.endswith("_proj.weight"):
                m,n = value.shape
                codes = rng.integers(0,16,(n,m),dtype=np.uint32)
                zeros = rng.integers(1,17,(n//128,m),dtype=np.uint32)
                scales = torch.full((n//128,m),.002,dtype=torch.float16)
                parts = packed(codes,zeros,scales)
                for suffix,tensor in zip(("qweight","scales","qzeros","g_idx"),parts):
                    tensors[name.removesuffix(".weight")+"."+suffix] = tensor
                decoded[key] = official_eager_reference(*parts).float()
            else:
                tensors[name] = torch.ones(value.shape,dtype=torch.float16)
                decoded[key] = tensors[name].float()
        filename = f"layer{layer}.safetensors"
        save_file(tensors,tmp_path/filename)
        index.update({key:filename for key in tensors}); expected[layer] = decoded
    dense = {"model.embed_tokens.weight":torch.arange(64*128).reshape(64,128).half()/8192,
             "lm_head.weight":torch.arange(64*128).reshape(64,128).half()/16384,
             "model.norm.weight":torch.ones(128,dtype=torch.float16)}
    save_file(dense,tmp_path/"rest.safetensors")
    index.update({key:"rest.safetensors" for key in dense})
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps(dict(weight_map=index)))
    return tmp_path,config,expected


def test_actual_phi_block_selective_load_forward_backward_and_inventory(tiny_source,monkeypatch):
    path,config,expected = tiny_source
    opened = []; original = weights.safe_open
    def recording(filename,**kwargs):
        opened.append(str(filename)); return original(filename,**kwargs)
    monkeypatch.setattr(weights,"safe_open",recording)
    loaded = weights.load_native_block(path,1,dtype=torch.float32,config=config)
    assert all("layer1" in x for x in opened)
    assert loaded.source_kind == weights.SOURCE_KIND
    assert all(not p.requires_grad and p.device.type == "cpu" for p in loaded.parameters())
    reference = Phi3DecoderLayer(config,layer_idx=1)
    reference.load_state_dict(expected[1],strict=True)
    for key,value in loaded.state_dict().items(): assert torch.equal(value,expected[1][key])
    torch.manual_seed(16)
    x = torch.randn(2,5,128,requires_grad=True)
    positions = torch.arange(5).expand(2,-1)
    rotary = weights.load_rotary(path,config=config,dtype=torch.float32)
    mask = torch.full((5,5),float("-inf")).triu(1)[None,None]
    args = dict(position_ids=positions,position_embeddings=rotary(x,positions),attention_mask=mask,use_cache=False)
    actual = loaded(x,**args)[0]; baseline = reference(x,**args)[0]
    assert torch.equal(actual,baseline)
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    inventory = weights.inspect_checkpoint(path)
    assert inventory["decoded_fp32_bytes"] == 4*inventory["dense_parameter_count"]
    assert inventory["layers"][1]["decoded_parameter_count"] == sum(v.numel() for v in expected[1].values())
    assert inventory["payload_hashes_verified"] is False
    assert inventory["source_kind"] == weights.SOURCE_KIND


def test_untied_head_and_rope_and_bf16_block(tiny_source):
    path,config,_ = tiny_source
    embedding = weights.load_embedding(path,dtype=torch.float32)
    head,norm,rotary = weights.load_readout_norm(path,dtype=torch.float32)
    assert not torch.equal(head,embedding)
    assert torch.equal(head,embedding/2)
    assert rotary.inv_freq.dtype == torch.float32 and norm.weight.shape == (128,)
    block = weights.load_native_block(path,0,dtype=torch.bfloat16)
    x = embedding[:3].unsqueeze(0).bfloat16().requires_grad_(True)
    positions = torch.arange(3).unsqueeze(0)
    y = block(x,position_ids=positions,position_embeddings=rotary(x,positions),
              attention_mask=torch.full((3,3),float("-inf")).triu(1)[None,None])[0]
    y.float().square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()


def test_reservation_denied_before_cuda_and_config_rejection(tiny_source,monkeypatch):
    path,config,_ = tiny_source
    def forbidden(*a,**k): raise AssertionError("CUDA was initialized")
    monkeypatch.setattr(torch.cuda,"_lazy_init",forbidden)
    with pytest.raises(ValueError,match="reservation"):
        weights.load_native_block(path,0,device="cuda:0")
    def denied(): raise RuntimeError("lease denied")
    with pytest.raises(RuntimeError,match="lease denied"):
        weights.load_readout_norm(path,device="cuda:0",guard=denied)
    raw = json.loads((path/"config.json").read_text())
    raw["quantization_config"]["checkpoint_format"] = "gptq_v2"
    (path/"config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError,match="GPTQ-v1"):
        weights.load_config(path)


def test_supplied_config_cannot_override_actual_geometry(tiny_source):
    path,config,_ = tiny_source
    config.rope_theta *= 2
    with pytest.raises(ValueError,match="rope_theta"):
        weights.load_config(path,config)
