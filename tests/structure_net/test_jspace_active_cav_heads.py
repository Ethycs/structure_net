"""Known-mechanism and actual tiny-Qwen controls for the prospective study."""
import io

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net.jspace_active_math import (
    task_dictionary, component_delta, make_patch, quadratic, head_scores, tensor_bytes,
)
from experiments.structure_net.jspace_active_heads import paired_concepts, polynomial_features, common_encoder
from experiments.structure_net.jspace_active_benchmark import truncate_model
from experiments.structure_net.jspace_active_study import answer_donor
from experiments.structure_net.jspace_relation_tasks import COUNTRIES, COLOURS, _make_split


class Tokenizer:
    def __call__(self, prompt, **kwargs):
        return {"input_ids": torch.tensor([[1, 3, 4, 5, 7, 8, 6]]), "attention_mask": torch.ones(1, 7, dtype=torch.long)}


def tiny_model():
    torch.manual_seed(31)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=3,
                        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).double().eval().requires_grad_(False)


def test_position_vjp_matches_actual_decoder_finite_difference():
    model, tokenizer = tiny_model(), Tokenizer()
    dictionary = task_dictionary(model, tokenizer, ["prompt"], [7, 9], {0: "question", 1: "final"}, suffix_tokens=4)
    assert dictionary[0].shape == (2, 48)
    assert dictionary[1].shape == (2, 16)

    def value(offset):
        captured = {}
        def patch(module, inputs, output):
            state = output[0].clone()
            state[0, 4, 2] += offset
            return (state, *output[1:])
        def observe(module, inputs, output):
            captured["h"] = output[0]
        h1 = model.model.layers[0].register_forward_hook(patch)
        h2 = model.model.layers[2].register_forward_hook(observe)
        try:
            model.model(**tokenizer("prompt"), use_cache=False)
            return float(captured["h"][0, -1]@model.get_output_embeddings().weight[7])
        finally:
            h1.remove()
            h2.remove()
    # Qwen RMSNorm internally casts to FP32 even on a float64 test model.
    epsilon = 1e-4
    measured = (value(epsilon)-value(-epsilon))/(2*epsilon)
    assert float(dictionary[0][0, 18]) == pytest.approx(measured, rel=1e-3, abs=1e-7)
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_actual_prefix_output_is_pre_norm_cut_state():
    model, tokenizer = tiny_model(), Tokenizer()
    captured = {}
    def hook(module, inputs, output):
        captured["state"] = output[0].detach().clone()
    handle = model.model.layers[1].register_forward_hook(hook)
    model.model(**tokenizer("prompt"), use_cache=False)
    handle.remove()
    before = sum(p.numel() for p in model.parameters())
    truncate_model(model, 1)
    output = model.model(**tokenizer("prompt"), use_cache=False)
    assert len(model.model.layers) == 2
    assert torch.equal(output.last_hidden_state, captured["state"])
    assert sum(p.numel() for p in model.parameters()) < before


def components(width):
    full = torch.eye(width)[:4]
    return {"full": full, "J": .4*full, "remainder": .6*full, "country_inverse": torch.linalg.pinv(full),
            "colour": full.flip(1), "colour_inverse": torch.linalg.pinv(full.flip(1)),
            "permutation": torch.arange(width-1, -1, -1), "sign": torch.ones(width)}


def test_shared_coefficients_reconstruct_delta_and_controls_match_norm():
    source, donor = torch.randn(12), torch.randn(12)
    fitted = components(12)
    full = component_delta(source, donor, fitted, "full")
    j = component_delta(source, donor, fitted, "J")
    r = component_delta(source, donor, fitted, "remainder")
    assert torch.allclose(full, j+r, atol=1e-6)
    assert component_delta(source, donor, fitted, "random_full").norm() == pytest.approx(float(full.norm()))
    assert component_delta(source, donor, fitted, "remainder_norm_J").norm() == pytest.approx(float(j.norm()))


def test_position_patch_preserves_unedited_state_and_dense_copy():
    hidden = torch.arange(32).reshape(1, 8, 4).to(torch.bfloat16)
    donor = torch.full((12,), -3., dtype=torch.bfloat16)
    patch, diagnostics = make_patch(donor, components(12), "question", "dense", suffix_tokens=4)
    result = patch(hidden)
    assert torch.equal(result[:, :4], hidden[:, :4])
    assert torch.equal(result[:, -1:], hidden[:, -1:])
    assert torch.equal(result[0, 4:7].flatten(), donor)
    assert diagnostics["positions"] == 3
    bad, _ = make_patch(torch.ones(8), components(12), "question", "dense", suffix_tokens=4)
    with pytest.raises(ValueError, match="dimensions"):
        bad(hidden)


def synthetic_pairs():
    pairs = []
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    pairs.append({"source": {"intermediate": COUNTRIES[a], "answer": COLOURS[c]},
                                  "donor": {"intermediate": COUNTRIES[b], "answer": COLOURS[d]}})
    return pairs


def test_paired_fit_separates_known_country_and_colour_effects():
    pairs = synthetic_pairs()
    rng = np.random.default_rng(7)
    country, colour = rng.normal(size=(4, 12)), rng.normal(size=(4, 12))
    country -= country.mean(0)
    colour -= colour.mean(0)
    source = rng.normal(size=(len(pairs), 12))
    donor = source+np.array([country[COUNTRIES.index(p["donor"]["intermediate"])]-country[COUNTRIES.index(p["source"]["intermediate"])]+
                            colour[COLOURS.index(p["donor"]["answer"])]-colour[COLOURS.index(p["source"]["answer"])] for p in pairs])
    c, a, diagnostics = paired_concepts(source, donor, pairs)
    assert np.allclose(c, country, atol=1e-12)
    assert np.allclose(a, colour, atol=1e-12)
    assert diagnostics["paired_difference_energy_explained"] == pytest.approx(1.)
    encoder, diagnostic = common_encoder(source, c, a)
    assert encoder["basis"].shape == (8, 12)
    assert torch.allclose(encoder["basis"]@encoder["basis"].T, torch.eye(8), atol=1e-6)


def test_confounded_country_and_colour_design_rejected():
    pairs = synthetic_pairs()
    for p in pairs:
        for side in ("source", "donor"):
            p[side]["answer"] = COLOURS[COUNTRIES.index(p[side]["intermediate"]) ]
    with pytest.raises(ValueError, match="identifiable"):
        paired_concepts(np.zeros((256, 8)), np.ones((256, 8)), pairs)


def test_gradient_normalized_polynomials_recover_class_circles():
    theta = np.linspace(0, 2*np.pi, 80, endpoint=False)
    x = np.concatenate([radius*np.column_stack((np.cos(theta), np.sin(theta))) for radius in (1, 2, 3, 4)])
    labels = np.repeat(np.arange(4), 80)
    polynomials, diagnostics = polynomial_features(x, labels, per_class=1, candidates=2)
    features = quadratic(torch.from_numpy(x).float())
    for label in range(4):
        assert float((features[labels == label]@polynomials[label]).abs().max()) < 2e-6
        assert diagnostics[label]["class_gradient_rms"] == pytest.approx(1., abs=1e-8)
        assert diagnostics[label]["other_class_residual_rms"] > .1


@pytest.mark.parametrize("kind,width", [("linear_full", 10), ("linear", 8), ("quadratic", 45), ("vanishing", 16), ("mlp", 4)])
def test_every_head_survives_weights_only_round_trip(kind, width):
    torch.manual_seed(13)
    model = {"kind": kind, "encoder": {"mean": torch.zeros(10), "basis": torch.eye(10)[:8], "scale": torch.ones(8)}}
    if kind == "vanishing":
        model["polynomials"] = torch.randn(16, 45)
    if kind == "mlp":
        model["state"] = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.GELU(), torch.nn.Linear(64, 4)).state_dict()
    else:
        model.update({"feature_mean": torch.zeros(width), "weight": torch.randn(width, 4), "prior": torch.zeros(4)})
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    values = torch.randn(3, 10)
    assert torch.equal(head_scores(model, values), head_scores(restored, values))
    assert tensor_bytes(model) > 0


def test_answer_donor_holds_country_and_changes_only_colour_assignment():
    source = _make_split("shakedown", 1, 43, set())[0]
    donor = answer_donor(source)
    assert source["person_to_country"] == donor["person_to_country"]
    assert source["query_person"] == donor["query_person"]
    assert source["intermediate"] == donor["intermediate"]
    assert source["answer"] != donor["answer"] == source["swap_answer"]
    assert source["world_id"] != donor["world_id"]
