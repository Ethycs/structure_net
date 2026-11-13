from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.structure_net import jspace_relations as relations


def parabola(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    return np.column_stack((x, x*x))


def frozen(values, *, groups=None):
    return relations.validate_relation_bank(relations.fit_relation_bank(values[:256], groups=groups),
                                            values[256:384], controls=8)


def test_monomial_counts_and_reject_underdetermined_nonfinite_inputs():
    assert len(relations.monomial_exponents(4, 2)) == 15
    assert len(relations.monomial_exponents(8, 2)) == 45
    with pytest.raises(ValueError, match="twice"):
        relations.fit_relation_bank(np.ones((20, 4)))
    with pytest.raises(ValueError, match="finite"):
        relations.fit_relation_bank(np.full((512, 4), np.nan))


def test_continuous_quadratic_relation_generalizes_without_discretization():
    values = parabola(512, 8)
    bank = frozen(values)
    result = relations.evaluate_relation_bank(bank, values[384:], controls=8)
    assert result["accepted_count"] >= 1
    best = min(result["relations"], key=lambda r: r["normalized_rms"])
    assert best["normalized_rms"] < 1e-12
    assert best["passes_frozen_test_gate"]
    assert best["coordinate_shuffle"]["median_rms"] > .5
    assert bank["train_rows"] == 256 and bank["dev_rows"] == 128
    json.dumps(bank, allow_nan=False)
    # Evaluation cannot silently update normalizers or thresholds.
    original = json.dumps(bank, sort_keys=True)
    shifted = values[384:].copy()
    shifted[:, 1] += 1
    broken = relations.evaluate_relation_bank(bank, shifted, controls=8)
    assert min(r["normalized_rms"] for r in broken["relations"]) > .3
    assert json.dumps(bank, sort_keys=True) == original


def test_independent_gaussians_and_constant_coordinates_do_not_invent_relations():
    values = np.random.default_rng(19).normal(size=(512, 4))
    assert relations.evaluate_relation_bank(frozen(values), values[384:], controls=8)["accepted_count"] == 0
    values[:, 1] = 1000 + 1e-10 * values[:, 1]
    bank = frozen(values)
    assert bank["excluded_low_variation_variables"] == ["z1"]
    assert not bank["accepted_ids"]
    with pytest.raises(ValueError, match="already validated"):
        relations.validate_relation_bank(bank, values[384:])


def test_pairing_shuffle_distinguishes_transition_from_marginal_relations():
    rng = np.random.default_rng(12)
    source = rng.normal(size=(512, 2))
    destination = np.column_stack((source[:, 0] + source[:, 1], rng.normal(size=512)))
    values = np.column_stack((source, destination))
    bank = frozen(values, groups=2)
    result = relations.evaluate_relation_bank(bank, values[384:], controls=8)
    assert result["accepted_count"] > 0
    best = min(result["relations"], key=lambda r: r["normalized_rms"])
    assert best["pairing_shuffle"]["median_rms"] > .5
    broken = values[384:].copy()
    broken[:, 2:] = broken[::-1, 2:]
    assert min(r["normalized_rms"] for r in relations.evaluate_relation_bank(bank, broken, controls=8)["relations"]) > .1
    # A shared source-only relation is invariant to pairing and must be rejected.
    marginal = np.column_stack((parabola(512, 4), rng.normal(size=(512, 2))))
    marginal_bank = frozen(marginal, groups=2)
    assert not marginal_bank["accepted_ids"]
    # SVD can mix independent marginal constraints into a polynomial with
    # coefficients on both groups. Tiny paired null residuals cannot validate it.
    independent_marginal = np.column_stack((parabola(512, 4), parabola(512, 5)))
    mixed_bank = frozen(independent_marginal, groups=2)
    assert any(c["mixed"] for c in mixed_bank["candidates"])
    assert not mixed_bank["accepted_ids"]


def test_quadratic_multiples_of_linear_relations_are_not_new_structure():
    rng = np.random.default_rng(47)
    x, noise = rng.normal(size=(2, 512))
    values = np.column_stack((x, 2*x+1, noise))
    bank = frozen(values)
    accepted = [c for c in bank["candidates"] if c["accepted"]]
    assert len(accepted) == 1 and accepted[0]["degree"] == 1
    redundant = [c for c in bank["candidates"] if c["degree"] == 2 and c["passes_dev"]]
    assert redundant
    assert all(c["redundant_at_degree_bound"] for c in redundant)
    assert bank["selection"]["max_rms"] == .10
    assert bank["selection"]["max_control_ratio"] == pytest.approx(1/3)


def test_quadratic_regression_is_reported_as_baseline_and_beats_linear_on_square():
    source = np.random.default_rng(2).uniform(-2, 2, (512, 1))
    destination = source**2
    model = relations.fit_transition_baselines(source[:256], destination[:256])
    result = relations.evaluate_transition_baselines(model, source[256:], destination[256:])
    assert result["baselines"]["quadratic"]["normalized_rmse"] < 1e-12
    assert result["baselines"]["linear"]["normalized_rmse"] > .9


def test_bounded_exact_elimination_positive_negative_and_unit_ideal_controls():
    # x^2=1 and y=x implies y^2=1. These are exact CPU test inputs,
    # not constraints imposed on the neural capture.
    payload = {"width": 2, "source_width": 1, "relations": [
        {"exponents": [[2, 0], [0, 0]], "rational_coefficients": ["1", "-1"]},
        {"exponents": [[0, 1], [1, 0]], "rational_coefficients": ["1", "-1"]}]}
    result = relations._bounded_groebner(payload, 5)
    assert result["status"] == "ok" and not result["empty_elimination"]
    assert result["consequences"][0]["polynomial"] == "v1**2 - 1"
    payload["relations"][1]["rational_coefficients"] = ["1", "-2"]
    changed = relations._bounded_groebner(payload, 5)
    assert changed["consequences"][0]["polynomial"] == "v1**2 - 4"
    payload["relations"] = [{"exponents": [[1, 0]], "rational_coefficients": ["1"]},
                            {"exponents": [[1, 0], [0, 0]], "rational_coefficients": ["1", "1/1000"]}]
    inconsistent = relations._bounded_groebner(payload, 5)
    assert inconsistent["status"] == "unit_ideal"
    assert inconsistent["unit_ideal"] and not inconsistent["consequences"]


def test_empty_elimination_and_hard_timeout_are_honest_results():
    payload = {"width": 2, "source_width": 1, "relations": [
        {"exponents": [[0, 1], [1, 0]], "rational_coefficients": ["1", "-1"]}]}
    result = relations._bounded_groebner(payload, 5)
    assert result["status"] == "ok" and result["empty_elimination"]
    assert relations._bounded_groebner(payload, .00001)["status"] == "timeout"
    with pytest.raises(ValueError, match="five"):
        relations._bounded_groebner(payload, 6)
    payload["relations"] = [{"exponents": [[0, 13], [0, 0]], "rational_coefficients": ["1", "-1"]}]
    assert relations._bounded_groebner(payload, 5)["status"] == "degree_explosion"


def test_rationalization_failure_and_no_accepted_inputs_do_not_claim_neural_proof():
    source = parabola(512, 2)
    destination = source + np.array([.123456, .432198])
    source_bank = frozen(source)
    joint_bank = frozen(np.column_stack((source, destination)), groups=2)
    result = relations.groebner_propagation(source_bank, joint_bank, source[256:384], destination[256:384],
                                           max_denominator=1, rationalization_tolerance=1e-12)
    assert result["status"] == "rationalization_rejected"
    assert result["exact_neural_claim"] is False
    assert not result["consequences"]
    no_relations = dict(source_bank, candidates=[])
    skipped = relations.groebner_propagation(no_relations, joint_bank, source[256:384], destination[256:384])
    assert skipped["status"] == "insufficient_accepted_relations"
    with pytest.raises(ValueError, match="seven"):
        relations.groebner_propagation(source_bank, joint_bank, source[256:384], destination[256:384], max_relations=8)


def test_propagated_consequence_evaluation_uses_frozen_scale():
    result = {"destination_mean": [0., 0.], "destination_scale": [1., 1.], "consequences": [
        {"polynomial": "y1-y0**2", "exponents": [[0, 1], [2, 0]],
         "coefficients": [1., -1.], "normalizer_dev": 2.}]}
    values = parabola(128, 7)
    score = relations.evaluate_propagated_relations(result, values, controls=8)
    assert score["propagated_relation_count"] == 1
    assert score["relations"][0]["normalized_rms"] < 1e-12
    assert score["relations"][0]["passes_frozen_test_gate"]
    assert score["relations"][0]["coordinate_shuffle"]["median_rms"] > .5
    assert score["exact_neural_claim"] is False


def test_exact_control_passes_full_numeric_rationalization_and_propagation_path():
    # Explicit exact controls exercise the glue around the symbolic worker.
    # They do not replace discovery from continuous neural scores.
    def exact_bank(width, polynomial_terms, groups=None):
        exponents = relations.monomial_exponents(width, 2)
        candidates = []
        for i, terms in enumerate(polynomial_terms):
            coefficients = [terms.get(tuple(p), 0.) for p in exponents]
            candidates.append({"id": str(i), "coefficients": coefficients,
                               "accepted": True, "mixed": groups is not None, "dev_rms": 0.})
        return {"variable_names": [f"z{i}" for i in range(width)], "groups": groups,
                "mean": [0.]*width, "scale": [1.]*width, "exponents": exponents,
                "validated": True, "candidates": candidates,
                "selection": {"max_rms": .10, "max_control_ratio": 1/3}}

    source_bank = exact_bank(2, [{(0, 1): 1., (2, 0): -1.}])
    joint_bank = exact_bank(4, [{(0, 0, 1, 0): 1., (1, 0, 0, 0): -1.},
                                {(0, 0, 0, 1): 1., (0, 1, 0, 0): -1.}], groups=2)
    dev, test = parabola(128, 101), parabola(128, 102)
    result = relations.groebner_propagation(source_bank, joint_bank, dev, dev)
    assert result["status"] == "ok"
    assert len(result["consequences"]) == 1
    assert all(c["change_dev_rms"] == 0 for c in result["rationalization"])
    score = relations.evaluate_propagated_relations(result, test, controls=8)
    assert score["relations"][0]["passes_frozen_test_gate"]
    assert score["relations"][0]["passes_dev_residual_gate"]
    assert result["exact_neural_claim"] is False
    json.dumps(result, allow_nan=False)
    test[:, 1] += 2
    broken = relations.evaluate_propagated_relations(result, test, controls=8)
    assert not broken["relations"][0]["passes_frozen_test_gate"]
