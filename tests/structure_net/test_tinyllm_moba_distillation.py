"""Numerical, gradient, decision, and provenance contracts for MoBA conversion."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.structure_net import tinyllm_moba_distillation as m


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return m.StudyConfig(device="cpu", shakedown=True, synthetic=True, seeds=(997,),
                         sequence_length=32, block_size=4, steps=4,
                         validation_tokens=2048, monitor_windows=1, test_windows=2)


@pytest.fixture
def model(config):
    return m.TinyLLMModel(m.TinyLLMConfig(block_size=32, vocab_size=64, n_layer=2,
                                         n_head=2, n_embd=16)).eval()


def test_config_locks_and_invalid_values(config):
    with pytest.raises(ValueError, match="frozen"):
        m.StudyConfig(steps=64)
    with pytest.raises(ValueError, match="CUDA"):
        m.StudyConfig(device="cpu")
    with pytest.raises(ValueError):
        replace(config, block_size=0)
    with pytest.raises(ValueError):
        replace(config, steps=3)


@pytest.mark.parametrize("length", [7, 16, 32])
def test_full_budget_equals_sdpa_and_prefix_causality(length):
    torch.manual_seed(51)
    q, k, v = [torch.randn(1, 2, length, 8) for _ in range(3)]
    output, _, _, selected, _ = m.attention(q, k, v, 4, 8)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
    sparse = m.attention(q, k, v, 4, 2)[0]
    for prefix in range(1, length):
        actual = m.attention(q[..., :prefix, :], k[..., :prefix, :], v[..., :prefix, :], 4, 2)[0]
        torch.testing.assert_close(sparse[..., :prefix, :], actual, atol=1e-6, rtol=1e-5)


def test_block_counts_and_no_future_selection():
    scores = torch.zeros(1, 2, 16, 4)
    selected = m.select_blocks(scores, 4, 2)
    local = torch.arange(16) // 4
    assert torch.all(selected.sum(-1) == (local + 1).clamp_max(2))
    assert not (selected & (torch.arange(4)[None, :] > local[:, None])).any()
    assert torch.all(selected[0, :, torch.arange(16), local])


def test_model_parity_and_no_future_tokens(model, config):
    ids = torch.randint(0, 64, (1, 32))
    native = model(ids, return_full_logits=True)[0]
    full = m.forward(model, ids, config, 8)[0]
    torch.testing.assert_close(native, full, atol=1e-6, rtol=1e-5)
    sparse = m.forward(model, ids, config, 2)[0]
    changed = ids.clone()
    changed[:, 13:] = torch.randint(0, 64, (1, 19))
    altered = m.forward(model, changed, config, 2)[0]
    torch.testing.assert_close(sparse[:, :13], altered[:, :13], atol=0, rtol=0)


def test_projected_contribution_matches_explicit_projection():
    torch.manual_seed(12)
    p = torch.randn(1, 2, 8, 8).softmax(-1)
    v = torch.randn(1, 2, 8, 4)
    w = torch.randn(8, 8)
    mass, contribution = m.teacher_targets(p, v, w, 4)
    for head in range(2):
        for block in range(2):
            c = p[0, head, :, block * 4:(block + 1) * 4] @ v[0, head, block * 4:(block + 1) * 4]
            expected = (c @ w[:, head * 4:(head + 1) * 4].T).norm(dim=-1)
            torch.testing.assert_close(contribution[0, head, :, block], expected)
    torch.testing.assert_close(mass.sum(-1), torch.ones(1, 2, 8))


def test_zero_contribution_fallback_and_route_gradient():
    scores = torch.randn(1, 2, 16, 4, requires_grad=True)
    eligible = m.past_blocks(16, 4, scores.device)
    mass = torch.rand_like(scores)
    target = m.normalize_target(torch.zeros_like(mass), eligible, mass)
    torch.testing.assert_close(target[..., 4:, :].sum(-1), torch.ones(1, 2, 12))
    loss = m.route_kl(scores, target, 4)
    assert loss.item() >= 0
    loss.backward()
    assert scores.grad[..., :4, :].abs().sum() == 0
    assert scores.grad.masked_select(~eligible).abs().sum() == 0
    assert scores.grad.abs().sum() > 0
    identical_scores = target.clamp_min(1e-30).log()
    assert abs(m.route_kl(identical_scores, target, 4).item()) < 1e-6


def test_qk_only_stage_preserves_fused_value_and_frozen_parameters(model, config):
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = m.configure_stage(model, 0, config)
    ids = torch.randint(0, 64, (1, 32))
    logits, _ = m.forward(model, ids, config, 8)
    F.cross_entropy(logits.flatten(0, 1), torch.randint(0, 64, (32,))).backward()
    m.mask_value_gradients(model)
    optimizer.step()
    changed = False
    for name, p in model.named_parameters():
        if ".attn.c_attn." in name:
            torch.testing.assert_close(p[32:], before[name][32:], atol=0, rtol=0)
            changed |= not torch.equal(p[:32], before[name][:32])
        else:
            torch.testing.assert_close(p, before[name], atol=0, rtol=0)
    assert changed


def test_objective_sign_and_teacher_free_stage(config):
    terms = {key: torch.tensor(float(i + 1), requires_grad=True) for i, key in enumerate(
        ["lm", "logits", "hidden", "attention", "route_mass", "route_contribution"])}
    loss = m.objective(terms, "combined_mass", 1, config)
    loss.backward()
    assert terms["route_mass"].grad == 1
    assert terms["route_contribution"].grad is None
    assert terms["hidden"].grad == pytest.approx(0.1)
    assert m.objective({"lm": terms["lm"]}, "combined_mass", 3, config) == terms["lm"]
    logits = torch.randn(1, 3, 5, requires_grad=True)
    teacher = torch.randn_like(logits, requires_grad=True)
    m.logits_kl(logits, teacher).backward()
    assert teacher.grad is None
    assert logits.grad.abs().sum() > 0


def test_paired_schedules_and_split_disjointness(config):
    first = m.make_schedules(8192, 7, config)
    repeated = m.make_schedules(8192, 7, config)
    other = m.make_schedules(8192, 17, config)
    assert all(np.array_equal(first[k], repeated[k]) for k in first)
    assert not np.array_equal(first["train"], other["train"])
    split = 8192 - config.validation_tokens
    assert first["train"].max() + 33 <= split
    assert first["monitor"].min() >= split
    assert first["monitor"].max() + 33 <= split + 1024
    assert first["test"].min() >= split + 1024
    assert first["test"].max() + 33 <= 8192


def test_recall_excludes_local_and_joint_gate_inclusive():
    scores = torch.randn(1, 1, 16, 4)
    assert m.recall(scores, scores, 4, 2) == 1
    assert m.recall(scores, scores, 4, 1) is None
    c = m.StudyConfig()
    gate = m.seed_gate({"nll": 4}, {"mass_recall": 0.5}, {"logits_kl": 1},
                       {"nll": 4.2, "mass_recall": 0.55, "logits_kl": 0.8}, c)
    assert gate["joint"]
    assert m.classify([gate] * 4 + [{"kl": False, "joint": False}], c) == "routing_distillation_supported"
    assert m.classify([{**gate, "joint": False}] * 5, c) == "behavioral_gain_without_full_joint_gate"
    assert m.classify([gate] * 3, c) == "incomplete"


def test_strict_json_fingerprint_and_reuse(tmp_path):
    with pytest.raises(ValueError):
        m.write_json(tmp_path / "bad.json", {"x": float("nan")})
    assert not (tmp_path / "bad.json").exists()
    assert m.digest({"seed": 1}) != m.digest({"seed": 2})
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"checkpoint")
    result = {"status": "completed", "schema_version": m.SCHEMA_VERSION, "scientific_fingerprint": "abc",
              "artifacts": {"checkpoint": {"path": str(artifact), "sha256": m.sha256(artifact)}}}
    path = tmp_path / "result.json"
    m.write_json(path, result)
    assert m.reuse_result(path, "abc") == result
    with pytest.raises(RuntimeError):
        m.reuse_result(path, "def")
    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError):
        m.reuse_result(path, "abc")
