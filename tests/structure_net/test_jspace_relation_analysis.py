from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from experiments.structure_net import jspace_relation_analysis as analysis


def arrays(seed=7, linked=False):
    rng = np.random.default_rng(seed)
    result = {}
    for split, count in (("train", 128), ("development", 64), ("evaluation", 64)):
        values = rng.normal(size=(count, 2, 4))
        if linked:
            values[:, :, 1] = values[:, :, 0]
            values[:, 1] = values[:, 0]
        result[split] = values
    return result


def test_fit_cannot_access_evaluation_and_freeze_has_fixed_protocol():
    class Guarded(dict):
        def __getitem__(self, key):
            if key == "evaluation":
                raise AssertionError("Heldout accessed during fitting")
            return super().__getitem__(key)

    frozen = analysis.fit_models({"jacobian": Guarded(arrays())}, layers=[5, 6], edges=[(5, 6)])
    assert frozen["symbolic_attempts"] == 1
    assert frozen["protocol"]["max_normalized_rms"] == .10
    assert frozen["protocol"]["joint_relations"] == 4
    assert frozen["protocol"]["shuffle_controls"] == 32
    assert frozen["status"] == "frozen_before_evaluation"
    assert frozen["exact_neural_claim"] is False
    json.dumps(frozen, allow_nan=False)


def test_wrapper_finishes_all_fits_before_first_heldout_access(monkeypatch):
    events = []
    original = analysis.fit_models

    def recorded_fit(*args, **kwargs):
        frozen = original(*args, **kwargs)
        events.append("all_models_frozen")
        return frozen

    class Tracked(dict):
        def __getitem__(self, key):
            if key == "evaluation":
                assert events and events[0] == "all_models_frozen"
                events.append("evaluation_read")
            return super().__getitem__(key)

    monkeypatch.setattr(analysis, "fit_models", recorded_fit)
    result = analysis.analyze_scores({"jacobian": Tracked(arrays()), "random": Tracked(arrays(19))},
                                     layers=[5, 6], edges=[(5, 6)])
    assert events == ["all_models_frozen", "evaluation_read", "evaluation_read"]
    assert result["frozen_models"]["symbolic_attempts"] == 2
    assert result["evaluation"]["interventions"]["status"] == "not_supplied"


def test_frozen_models_are_unchanged_and_tampering_is_rejected():
    scores = arrays()
    frozen = analysis.fit_models({"jacobian": scores}, layers=[5, 6], edges=[(5, 6)])
    original = copy.deepcopy(frozen)
    evaluated = analysis.evaluate_models(frozen, {"jacobian": scores["evaluation"]})
    assert frozen == original
    assert evaluated["banks"]["jacobian"]["summary"]["layer_candidates_accepted_on_dev"] == 0
    frozen["seed"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        analysis.evaluate_models(frozen, {"jacobian": scores["evaluation"]})


def test_interventions_keep_all_rows_and_show_source_compatible_subset():
    scores = arrays(linked=True)
    frozen = analysis.fit_models({"jacobian": scores}, layers=[5, 6], edges=[(5, 6)])
    source_bank = frozen["banks"]["jacobian"]["layers"]["5"]
    compatible, _ = analysis._source_compatibility(source_bank, scores["development"][:, 0])
    assert compatible is not None and compatible.any()
    sample = scores["development"][np.flatnonzero(compatible)[0]]
    interventions = np.stack((sample, sample.copy(), sample.copy()))
    interventions[1, 0, 1] += 10
    records = [{"case_id": "same-case", "arm": "jacobian", "source_site": 5,
                "original_correct": True, "counterfactual_correct": False},
               {"case_id": "broken-source", "arm": "jacobian", "source_site": 5,
                "original_correct": False, "counterfactual_correct": True},
               {"case_id": "other-arm", "arm": "random", "source_site": 5,
                "original_correct": False, "counterfactual_correct": False}]
    result = analysis.evaluate_models(frozen, {"jacobian": scores["evaluation"]},
                                      {"jacobian": interventions}, records)
    report = result["interventions"]
    assert report["unmatched_arm_row_indices"] == [2]
    assert len(report["records"]) == 3
    group = report["groups"][0]
    assert group["rows"] == 2
    assert group["task_scores"]["original_correct"]["accuracy"] == .5
    assert group["task_scores"]["counterfactual_correct"]["accuracy"] == .5
    edge = group["edges"]["5->6"]
    assert edge["all_rows"]["rows"] == 2
    assert edge["source_compatible_subset"]["rows"] == 1
    assert edge["source_compatible_subset"]["row_indices"] == [0]
    assert edge["source_compatibility"]["incompatible_rows"] == 1
    json.dumps(result, allow_nan=False)


def test_no_source_relation_means_undefined_compatibility_not_vacuous_success():
    scores = arrays()
    frozen = analysis.fit_models({"jacobian": scores}, layers=[5, 6], edges=[(5, 6)])
    record = {"case_id": "one", "arm": "jacobian", "source_site": 5}
    result = analysis.evaluate_models(frozen, {"jacobian": scores["evaluation"]},
                                      {"jacobian": scores["evaluation"][:1]}, [record])
    edge = result["interventions"]["groups"][0]["edges"]["5->6"]
    assert edge["all_rows"]["rows"] == 1
    assert edge["source_compatibility"]["defined"] is False
    assert edge["source_compatible_subset"]["status"] == "undefined"


def test_layout_bounds_alignment_and_nonfinite_scores_are_rejected():
    with pytest.raises(ValueError, match="six"):
        analysis.fit_models({"jacobian": arrays()}, layers=list(range(8)), edges=[(i, i+1) for i in range(7)])
    mismatch = arrays(9)
    mismatch["development"] = mismatch["development"][:30]
    with pytest.raises(ValueError, match="align"):
        analysis.fit_models({"jacobian": arrays(), "random": mismatch}, layers=[5, 6], edges=[(5, 6)])
    bad = arrays()
    bad["train"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        analysis.fit_models({"jacobian": bad}, layers=[5, 6], edges=[(5, 6)])


def test_propagated_polynomial_counts_require_same_dev_and_test_success():
    scores = arrays()
    frozen = analysis.fit_models({"jacobian": scores}, layers=[5, 6], edges=[(5, 6)])
    edge = copy.deepcopy(frozen["banks"]["jacobian"]["edges"]["5->6"])
    # Exact continuous control: x1=x0^2 and identity bridge imply y1=y0^2.
    # The underlying worker's elimination is tested in test_jspace_relations;
    # this checks the separate development/test accounting layer.
    polynomial = "v5 - v4**2"
    edge["propagation"] = {"destination_mean": [0.]*4, "destination_scale": [1.]*4,
                           "consequences": [{"polynomial": polynomial,
                                             "exponents": [[0, 1, 0, 0], [2, 0, 0, 0]],
                                             "coefficients": [1., -1.], "normalizer_dev": 1.,
                                             "passes_dev_residual_gate": True}]}
    edge["dev_validated_polynomials"] = []
    destination = scores["evaluation"][:, 1].copy()
    destination[:, 1] = destination[:, 0]**2
    source = destination.copy()
    unvalidated = analysis._evaluate_edge(edge, source, destination, seed=7)
    assert unvalidated["propagated_relations"]["relations"][0]["passes_frozen_test_gate"]
    assert unvalidated["validated_propagation_count"] == 0
    edge["dev_validated_polynomials"] = [polynomial]
    validated = analysis._evaluate_edge(edge, source, destination, seed=7)
    assert validated["validated_propagation_count"] == 1
    destination[:, 1] += 2
    broken = analysis._evaluate_edge(edge, source, destination, seed=7)
    assert broken["validated_propagation_count"] == 0
    assert broken["exact_neural_claim"] is False


def test_interventions_before_no_primary_edge_keep_records_without_retrocausal_scoring():
    scores = arrays()
    frozen = analysis.fit_models({"jacobian": scores}, layers=[5, 6], edges=[(5, 6)])
    record = {"case_id": "late", "arm": "jacobian", "source_site": 6}
    result = analysis.evaluate_models(frozen, {"jacobian": scores["evaluation"]},
                                      {"jacobian": scores["evaluation"][:1]}, [record])
    assert result["interventions"]["rows"] == 1
    assert result["interventions"]["groups"][0]["edges"] == {}
    with pytest.raises(ValueError, match="duplicate|Duplicate"):
        analysis.evaluate_models(frozen, {"jacobian": scores["evaluation"]},
                                  {"jacobian": scores["evaluation"][:2]}, [record, record])
