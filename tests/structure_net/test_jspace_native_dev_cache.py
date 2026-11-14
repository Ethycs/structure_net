from copy import deepcopy

import numpy as np
import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding

from experiments.structure_net import jspace_native_dev_cache as cache


def tiny():
    torch.manual_seed(392)
    config = Qwen2Config(hidden_size=16, intermediate_size=24, num_attention_heads=4,
        num_key_value_heads=2, num_hidden_layers=3, vocab_size=31, attention_dropout=0., max_position_embeddings=32)
    config._attn_implementation = "eager"
    blocks = [Qwen2DecoderLayer(config, layer_idx=i).bfloat16().float().eval().requires_grad_(False) for i in range(3)]
    rotary = Qwen2RotaryEmbedding(config=config)
    norm = Qwen2RMSNorm(16).bfloat16().float().eval().requires_grad_(False)
    embedding = torch.randn(31, 16).bfloat16().float()/3
    ids = [[1, 2, 3], [1, 2, 8, 9, 10, 11, 12], [7, 6, 5, 4, 3]]
    cases = [dict(identity=str(i), family="tiny", input_ids=tokens, choices=["A", "B"], answer=i%2)
        for i, tokens in enumerate(ids)]
    return blocks, rotary, norm, embedding, cases


def test_all_boundaries_match_causal_unpadded_native_and_stream_score(tmp_path):
    blocks, rotary, norm, embedding, cases = tiny()
    result = cache.capture(cases, input_embedding=embedding, block_loader=lambda i:deepcopy(blocks[i]),
        final_readout=(norm, embedding), label_ids=[29, 30], directory=tmp_path/"cache",
        rotary=rotary, layer_count=3, microbatch=2)
    offsets = np.cumsum([0, *[len(c["input_ids"]) for c in cases]])
    banks = [torch.from_numpy(np.load(tmp_path/"cache"/f"boundary_{i:02d}.npy")) for i in range(4)]
    assert result["shape"] == [15, 16] and result["boundaries"] == 4
    with torch.no_grad():
        for j, case in enumerate(cases):
            hidden = embedding[case["input_ids"]][None]
            length = hidden.shape[1]; positions = torch.arange(length)[None]
            mask = torch.zeros(1, 1, length, length).masked_fill(
                torch.ones(length, length, dtype=torch.bool).triu(1), torch.finfo(torch.float32).min)
            torch.testing.assert_close(banks[0][offsets[j]:offsets[j+1]], hidden[0], atol=0, rtol=0)
            for layer, block in enumerate(blocks):
                hidden = block(hidden, attention_mask=mask, position_ids=positions,
                    position_embeddings=rotary(hidden, positions), use_cache=False)[0]
                torch.testing.assert_close(banks[layer+1][offsets[j]:offsets[j+1]], hidden[0], atol=2e-6, rtol=2e-6)
    # Different suffix tokens and padding cannot affect their shared prefix.
    for bank in banks:
        torch.testing.assert_close(bank[:2], bank[offsets[1]:offsets[1]+2], atol=2e-6, rtol=2e-6)
    scored = cache.scoring.stream_score(cases, input_embedding=embedding, block_loader=lambda i:deepcopy(blocks[i]),
        final_readout=(norm, embedding), label_ids=[29, 30], scratch_dir=tmp_path/"score",
        rotary=rotary, layer_count=3, microbatch=2)
    assert result["records"] == scored["records"]
    assert len(list((tmp_path/"cache").glob("boundary_*.npy"))) == 4


def test_arbitrary_layer_native_substitution_reproduces_cached_final_logits(tmp_path):
    blocks, rotary, norm, embedding, cases = tiny()
    original = cache.capture(cases, input_embedding=embedding, block_loader=lambda i:deepcopy(blocks[i]),
        final_readout=(norm, embedding), label_ids=[29, 30], directory=tmp_path/"cache",
        rotary=rotary, layer_count=3, microbatch=2)
    offsets = np.cumsum([0, *[len(c["input_ids"]) for c in cases]])
    virtual = [dict(c, input_ids=list(range(int(offsets[i]), int(offsets[i+1])))) for i,c in enumerate(cases)]
    for cut in (1, 2):
        bank = torch.from_numpy(np.load(tmp_path/"cache"/f"boundary_{cut:02d}.npy"))
        replay = cache.scoring.stream_score(virtual, input_embedding=bank,
            block_loader=lambda local:deepcopy(blocks[cut+local]), final_readout=(norm, embedding),
            label_ids=[29, 30], scratch_dir=tmp_path/f"replay{cut}", rotary=rotary,
            layer_count=3-cut, microbatch=2)
        assert replay["records"] == original["records"]


def test_dev_only_offsets_and_cpu_prepare_guard(tmp_path, monkeypatch):
    source = [dict(identity="train", split="train", offset=0, stop=2, input_ids=[1, 2]),
        dict(identity="dev0", split="development", offset=2, stop=5, input_ids=[3, 4, 5]),
        dict(identity="dev1", split="development", offset=5, stop=7, input_ids=[6, 7])]
    layout = cache.development_items(source)
    assert [(c["offset"], c["stop"]) for c in layout["items"]] == [(0, 3), (3, 5)]
    assert [c["source_offset"] for c in layout["items"]] == [2, 5]
    assert [c["identity"] for c in layout["items"]] == ["dev0", "dev1"]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(cache.recipe, "verify", lambda *_:pytest.fail("Read before CPU guard"))
    with pytest.raises(ValueError, match="Hide CUDA"):
        cache.prepare(tmp_path, tmp_path, tmp_path)
