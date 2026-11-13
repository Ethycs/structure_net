"""Positive and adversarial algebra cases; no transformer/model download needed."""
import json
import numpy as np
import pytest
import sympy as sp

from experiments.structure_net.jspace_groebner_controls import coordinate_payloads, remainder, run_controls
from experiments.structure_net.jspace_groebner_classification import aggregate_audit, graph_consequences
from experiments.structure_net import jspace_relations as relations


@pytest.fixture(scope="module")
def controls():
    return {row["id"]: row for row in run_controls()["controls"]}


def test_positive_and_negative_controls_all_classified(controls):
    assert len(controls) == 11
    assert all(row["passed"] for row in controls.values())


def test_control_artifact_is_strict_json_and_has_no_neural_accuracy():
    artifact = json.loads(json.dumps(run_controls(), allow_nan=False))
    assert artifact["neural_speedup"] is None
    assert artifact["neural_compression_tested"] is False
    assert "accuracy" not in artifact


def test_payload_savings_require_round_trip_and_branch_information():
    payload = coordinate_payloads()
    assert payload["original_bytes"] == 1040
    assert payload["retain_x_bytes"] == 520
    assert payload["retain_y_and_branch_bytes"] == 529
    assert payload["retain_x_bitwise_roundtrip"]
    assert payload["retain_y_and_branch_bitwise_roundtrip"]


def test_reconstruction_does_not_imply_future_closure(controls):
    witness = controls["nonclosed_projection"]["witness"]
    assert witness[0]["retained_x"] == witness[1]["retained_x"]
    assert witness[0]["next_retained_x"] != witness[1]["next_retained_x"]
    assert controls["closed_reduced_dynamics"]["mismatches"] == 0


def test_constraint_breaking_transition_is_detected():
    x, y = sp.symbols("x y")
    assert remainder(y-x*x, [y-x*x], (y, x)) == 0
    assert remainder((y+1)-(x+1)**2, [y-x*x], (y, x)) == -2*x


def test_exact_inconsistency_and_real_domain_are_separate(controls):
    assert controls["near_relations_can_generate_unit_ideal"]["exact_basis"] == ["1"]
    assert controls["real_domain_not_encoded_by_ideal"]["destination_basis"] == []
    assert controls["real_domain_not_encoded_by_ideal"]["real_preimage"] == "EmptySet"


def graph_fixture():
    # The graph creates v=u^2 despite having no source constraint. Use explicit
    # known equations; relation-discovery quality is not the subject of this test.
    x = np.linspace(-1, 1, 100)
    values = np.column_stack((x, x, x*x))
    exponents = relations.monomial_exponents(3)
    coefficients = []
    for terms in ({(1, 0, 0): -1, (0, 1, 0): 1}, {(2, 0, 0): -1, (0, 0, 1): 1}):
        coefficients.append([terms.get(tuple(p), 0) for p in exponents])
    bank = {"validated": True, "groups": 1, "mean": [0]*3, "scale": [1]*3,
            "variable_names": ["x", "u", "v"], "exponents": exponents,
            "candidates": [{"id": str(i), "accepted": True, "mixed": True, "dev_rms": 0,
                            "coefficients": c, "degree": i+1} for i, c in enumerate(coefficients)]}
    return bank, values


def test_graph_only_elimination_finds_created_relation():
    bank, values = graph_fixture()
    graph = graph_consequences(bank, values[:, :1], values[:, 1:])
    assert graph["source_constraints_used"] == 0
    assert graph["status"] == "ok"
    assert len(graph["consequences"]) == 1
    good = relations.evaluate_propagated_relations(graph, values[:, 1:], controls=8)
    assert good["relations"][0]["passes_frozen_test_gate"]
    broken = values[:, 1:].copy()
    broken[:, 1] += 1
    bad = relations.evaluate_propagated_relations(graph, broken, controls=8)
    assert not bad["relations"][0]["passes_frozen_test_gate"]


def test_unfrozen_bank_rejected():
    bank, values = graph_fixture()
    bank["validated"] = False
    with pytest.raises(ValueError, match="frozen"):
        graph_consequences(bank, values[:, :1], values[:, 1:])


def test_rationalization_cannot_drop_a_critical_small_coefficient():
    bank, values = graph_fixture()
    bank["candidates"] = bank["candidates"][:1]
    bank["candidates"][0]["coefficients"][1] = -1e-5
    values[:, 0] *= 1e5
    result = graph_consequences(bank, values[:, :1], values[:, 1:])
    assert result["status"] == "rationalization_rejected"
    assert not result["symbolic_attempted"]


def test_timeout_remains_unresolved_in_aggregate():
    frozen = {"jacobian": {"0->1": {"graph": {"status": "timeout", "symbolic_attempted": True,
                                               "consequences": []}}}}
    evaluated = {"jacobian": {"0->1": {"relations": []}}}
    result = aggregate_audit(frozen, evaluated)
    assert result["unresolved_symbolic_cells"] == 1
    assert result["classification"] == "no_validated_graph_consequences_with_unresolved_cells"
