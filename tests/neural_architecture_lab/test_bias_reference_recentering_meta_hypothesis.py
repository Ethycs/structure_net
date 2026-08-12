import json
from pathlib import Path

import pytest

from neural_architecture_lab.bias_reference_recentering_meta_hypothesis import (
    build_bias_reference_recentering_experiment_results,
    build_bias_reference_recentering_meta_hypothesis,
    store_bias_reference_recentering_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_bias_reference_recentering/"
    "20260810_d10_preregistered_v2/campaign_results.json"
)


def test_meta_preserves_specific_exact_pilot_repair() -> None:
    record = build_bias_reference_recentering_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_specific_observed_bias_repair"
    )
    assert hypothesis["evidence_count"] == 10
    assert hypothesis["integrity_valid_experiment_count"] == 10
    assert hypothesis["subclaims"] == {
        "correct_recentering_utility": "supported_five_of_five_both_arms",
        "wrong_sign_specificity": (
            "supported_analytic_one_of_five_learned_zero_of_five"
        ),
        "target_changing_specificity": "supported_zero_of_five_both_arms",
        "repaired_centered_equivalence": (
            "supported_feature_and_posterior_contracts_all_systems"
        ),
        "exact_pilot_interface_repair": "supported",
        "finite_noisy_pilot_acquisition": "not_tested",
        "same_scope_retraining": "not_licensed",
    }


def test_experiment_records_preserve_repair_and_control_counts() -> None:
    record = build_bias_reference_recentering_meta_hypothesis(RESULTS)
    experiments = build_bias_reference_recentering_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 10
    assert all(item.metrics["validity"] == 1.0 for item in experiments)
    assert sum(
        item.metrics["recenter_correct_seed_pass"] for item in experiments
    ) == 10.0
    assert sum(
        item.metrics["recenter_wrong_sign_seed_pass"] for item in experiments
    ) == 1.0
    assert sum(
        item.metrics["recenter_target_changing_seed_pass"] for item in experiments
    ) == 0.0
    assert sum(item.primary_metric for item in experiments) == 9.0
    assert max(
        item.metrics[f"{regime}_feature_equivalence_error"]
        for item in experiments
        for regime in ("composition", "extrapolation")
    ) < 1e-6
    assert max(
        item.metrics[f"{regime}_posterior_equivalence_error"]
        for item in experiments
        for regime in ("composition", "extrapolation")
    ) < 2e-6


def test_campaign_cannot_rewrite_specific_confirmation(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["primary_hypothesis_pass"] = False
    campaign["aggregates"]["classification"] = (
        "algebraic_repair_without_specificity"
    )
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="bias-reference campaign"):
        build_bias_reference_recentering_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_bias_reference_recentering_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_bias_reference_recentering_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-bias-reference-recentering-v2",
        "experiment_count": 10,
    }
