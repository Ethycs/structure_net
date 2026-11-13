import copy

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net.jspace_attention_runtime import boundaries, compact_delta, fit_dictionary, pooled
from experiments.structure_net.jspace_attention_transport import polynomial, regress, predict, rollout, error_summary
from experiments.structure_net.jspace_attention_tasks import render, validate_tasks
from experiments.structure_net import jspace_attention_tasks as task_module


def tiny():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).double().eval().requires_grad_(False)


class Tokens:
    def __call__(self, *args, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])}


def test_real_attention_vjp_matches_finite_difference_and_addition_identity():
    model, tok = tiny(), Tokens()
    dictionary = fit_dictionary(model, tok, ["calibration"], [9, 10], start=0)
    for layer in range(3):
        assert torch.equal(dictionary["raw"][f"{layer}.attention"], dictionary["raw"][f"{layer}.residual"])
    def measured(epsilon):
        def edit(key, h):
            if key == "0.attention":
                value = h.clone()
                value[:, 1:-1, 3] += epsilon
                return value
        with torch.no_grad(), boundaries(model, 0, edit):
            state = {}
            handle = model.model.layers[-1].register_forward_hook(lambda m, i, o: state.update(h=o[0] if isinstance(o, tuple) else o))
            try:
                model.model(**tok(), use_cache=False)
            finally:
                handle.remove()
        return float((state["h"][:, 1:-1] @ model.lm_head.weight[9]).sum()/6)
    finite = (measured(1e-5)-measured(-1e-5))/2e-5
    assert finite == pytest.approx(float(dictionary["raw"]["0.attention"][0, 3]), rel=2e-4, abs=2e-6)


def test_real_output_projection_transports_covector_exactly():
    model = tiny()
    projection = model.model.layers[1].self_attn.o_proj
    x = torch.randn(7, 16, dtype=torch.double)
    b = torch.randn(4, 16, dtype=torch.double)
    assert torch.allclose(projection(x) @ b.T, x @ (b @ projection.weight).T, atol=1e-12, rtol=1e-12)


def test_hooks_removed_after_exception():
    model = tiny()
    def fail(key, h):
        raise RuntimeError("deliberate")
    with pytest.raises(RuntimeError, match="deliberate"), boundaries(model, 0, fail):
        model.model(**Tokens()(), use_cache=False)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_projection_and_remainder_preserve_declared_components():
    torch.manual_seed(3)
    bank = torch.randn(8, 24)
    source, donor = torch.randn(10, 24), torch.randn(10, 24)
    delta = compact_delta(source, donor, bank, "projected_donor", 0, 1)
    rest = compact_delta(source, donor, bank, "remainder", 0, 1)
    random = compact_delta(source, donor, bank, "random_projected_donor", 0, 1)
    assert torch.allclose((source+delta) @ bank.T, donor @ bank.T, atol=2e-5)
    assert torch.allclose(rest @ bank.T, torch.zeros(10, 8), atol=2e-5)
    assert torch.allclose(delta+rest, donor-source, atol=1e-6)
    assert random.norm().item() == pytest.approx(delta.norm().item(), rel=1e-6)


def test_concept_swap_changes_only_selected_lexical_coefficients():
    bank = torch.eye(12)[:8]
    source = torch.arange(36).float().reshape(3, 12)
    edited = source+compact_delta(source, source, bank, "concept_swap", 0, 2)
    expected = source.clone()
    expected[:, [0, 1, 4, 5]] = source[:, [4, 5, 0, 1]]
    assert torch.equal(edited, expected)


def test_pooling_keeps_final_token_separate():
    x = torch.arange(30).reshape(10, 3).float()
    assert torch.equal(pooled(x, 4), torch.stack((x[4:9].mean(0), x[9])))
    with pytest.raises(ValueError):
        pooled(x, 9)


def test_quadratic_transport_recovers_known_nonlinear_action():
    rng = np.random.default_rng(9)
    x, d = rng.normal(size=(400, 2)), rng.normal(size=(80, 2))
    action = lambda a: np.stack((a[:, 0]+a[:, 1], a[:, 0]*a[:, 1]), 1)
    q, _ = regress(polynomial(x, True), action(x), polynomial(d, True), action(d))
    l, _ = regress(x, action(x), d, action(d))
    e = rng.normal(size=(80, 2))
    qe = np.mean((predict(q, polynomial(e, True))-action(e))**2)
    le = np.mean((predict(l, e)-action(e))**2)
    assert qe < 1e-5 and qe < le/1000


def test_rollout_and_divergence_are_not_scored_on_survivors():
    step = {"mean": torch.zeros(2, dtype=torch.double), "prior": torch.zeros(1, dtype=torch.double),
            "weight": torch.tensor([[2.], [0.]], dtype=torch.double)}
    model = {"steps": {"linear": [step, step]}}
    x, masks = rollout(model, np.array([[1.], [3000.]]), np.zeros((2, 1)), "linear")
    assert x[0, :, 0].tolist() == [1., 2., 4.]
    assert not masks[1, 2]
    result = error_summary(x[:, 2], np.array([[4.], [12000.]]), masks[:, 2])
    assert result["status"] == "diverged" and result["nrmse"] is None


def test_holdout_validation_rejects_leakage_and_duplicate_worlds(monkeypatch):
    monkeypatch.setattr(task_module, "COUNTS", {"train": 1, "evaluation": 1})
    mapping = {"Alice": "China", "Ben": "France", "Clara": "Germany", "David": "Russia"}
    source = render(mapping, list(mapping), "Alice", "capital", "case", "France")
    donor = render(mapping, list(mapping), "Ben", "capital", "donor", "China")
    pair = {"source": source, "donor": donor, "world_id": "one"}
    data = {"pairs": {"train": [pair], "evaluation": [{**copy.deepcopy(pair), "world_id": "two"}]}}
    validate_tasks(data)
    data["pairs"]["evaluation"][0]["world_id"] = "one"
    with pytest.raises(ValueError, match="overlaps"):
        validate_tasks(data)
    data["pairs"].pop("evaluation")
    data["pairs"]["train"][0]["source"] = render(mapping, list(mapping), "Alice", "nationality", "case", "France")
    with pytest.raises(ValueError, match="holdout leaked"):
        validate_tasks(data)
