import pytest
import torch
from transformers import Qwen3Config

from experiments.structure_net import qwen_moba_prefix as q
from experiments.structure_net import qwen_nonattention_fast as f
from experiments.structure_net.moba_head_benchmark import sdpa_call


def test_fast_float32_graph_matches_original_and_retains_original_parameters(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    monkeypatch.setattr(q, "flash_call", sdpa_call)
    config = Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = q.QwenPrefix(config).eval().requires_grad_(False)
    before = q.frozen_hash(model)
    optimized = f.FastPrefix(model, compile_kernels=False)
    ids = torch.arange(64)[None, :]
    with torch.inference_mode():
        expected = model(ids, q.FULL, segment_size=8)
        actual = optimized(ids, q.FULL, segment_size=8)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-5)
    assert before == q.frozen_hash(model)


def test_fidelity_gates_include_magnitude_and_ranking():
    m = {"heads_cosine_mean": .995, "ov_cosine_mean": .995, "heads_relative_l2": .05,
         "ov_relative_l2": .05, "change_pearson": .995, "top_quarter_change_f1": .95}
    assert f.fidelity_gate(m, True)
    assert not f.fidelity_gate(m, False)
    assert not f.fidelity_gate({**m, "heads_relative_l2": .051}, True)
    assert not f.fidelity_gate({**m, "top_quarter_change_f1": .94}, True)
    assert not f.fidelity_gate({**m, "change_pearson": None}, True)
