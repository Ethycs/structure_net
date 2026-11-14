"""Localization uses dev hidden states and an original suffix with exact positions."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net.jspace_prefix_localize import reconstruct_development, validate_cuts
from experiments.structure_net.jspace_quadratic_evaluate import stream_score


def data(tmp_path):
    cases = [dict(identity="dev_a", family="toy", input_ids=[7, 8, 9], choices=["A", "B"], answer=0),
             dict(identity="dev_b", family="toy", input_ids=[4, 5, 6, 7, 8], choices=["A", "B"], answer=1)]
    items = [dict(identity="train", split="train", input_ids=[2, 3], family="toy", answer=0, choices=["A", "B"], offset=0, stop=2)]
    offset = 2
    for case in cases:
        items.append(dict(case, split="development", offset=offset, stop=offset+len(case["input_ids"])))
        offset += len(case["input_ids"])
    torch.manual_seed(19)
    z = torch.randn(offset, 4); z[:2] = torch.nan
    basis = torch.linalg.qr(torch.randn(8, 4)).Q
    anchor = torch.randn(1, 8)
    path = tmp_path / "prefix.npy"; np.save(path, z.numpy())
    return cases, items, path, z, basis, anchor


def test_reconstruction_reads_dev_only_and_preserves_lengths_labels_and_positions(tmp_path):
    cases, items, path, z, basis, anchor = data(tmp_path)
    original = deepcopy(cases)
    hidden, virtual, provenance = reconstruct_development(cases, items, path, basis, anchor, chunk_tokens=2)
    torch.testing.assert_close(hidden, z[2:] @ basis.T + anchor, rtol=1e-6, atol=1e-6)
    assert torch.isfinite(hidden).all() and hidden.device.type == "cpu"
    assert cases == original
    assert virtual[0]["input_ids"] == [0, 1, 2] and virtual[1]["input_ids"] == [3, 4, 5, 6, 7]
    assert [row["original_length"] for row in provenance["cases"]] == [3, 5]
    assert [(r["identity"], r["answer"]) for r in virtual] == [(r["identity"], r["answer"]) for r in cases]


@pytest.mark.parametrize("fault", ["training", "changed_content", "duplicate", "offset"])
def test_rejects_non_development_or_mismatched_cached_rows(tmp_path, fault):
    cases, items, path, z, basis, anchor = data(tmp_path)
    if fault == "training": cases[0] = deepcopy(items[0])
    elif fault == "changed_content": cases[0]["input_ids"] = [8, 8, 9]
    elif fault == "duplicate": cases.append(deepcopy(cases[0]))
    else: items[1]["stop"] += 1
    with pytest.raises(ValueError):
        reconstruct_development(cases, items, path, basis, anchor)


def test_virtual_hidden_table_streaming_matches_direct_original_suffix(tmp_path):
    cases, items, path, z, basis, anchor = data(tmp_path)
    hidden, virtual, _ = reconstruct_development(cases, items, path, basis, anchor)
    cfg = Qwen2Config(hidden_size=8, intermediate_size=12, num_attention_heads=2,
        num_key_value_heads=1, num_hidden_layers=3, vocab_size=16, attention_dropout=0.)
    cfg._attn_implementation = "eager"
    model = Qwen2ForCausalLM(cfg).float().eval().requires_grad_(False)
    cut = 1
    actual = stream_score(virtual, input_embedding=hidden,
        block_loader=lambda index: deepcopy(model.model.layers[cut+index]),
        final_readout=(model.model.norm, model.lm_head.weight), label_ids=[0, 1],
        scratch_dir=tmp_path / "scratch", rotary=model.model.rotary_emb,
        layer_count=3-cut, microbatch=2)
    with torch.inference_mode():
        for case, record in zip(virtual, actual["records"]):
            x = hidden[case["input_ids"]][None]
            length = x.shape[1]; pos = torch.arange(length)[None]
            mask = torch.zeros(1, 1, length, length).masked_fill(
                ~torch.ones(length, length, dtype=torch.bool).tril(), torch.finfo(torch.float32).min)
            rope = model.model.rotary_emb(x, pos)
            for block in model.model.layers[cut:]:
                x = block(x, attention_mask=mask, position_ids=pos, cache_position=pos[0], position_embeddings=rope)[0]
            expected = model.lm_head(model.model.norm(x[:, -1]))[0, :2]
            torch.testing.assert_close(torch.tensor(record["logits"]), expected, rtol=1e-5, atol=1e-6)
    assert not (tmp_path / "scratch").exists()


@pytest.mark.parametrize("cuts", [[], [0], [36], [1, 1], [True], [2.5]])
def test_only_nonempty_native_suffix_cuts_are_allowed(cuts):
    with pytest.raises(ValueError): validate_cuts(cuts)
