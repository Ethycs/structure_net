import json
from pathlib import Path

import pytest

from neural_architecture_lab.noise_law_observed_twirl_meta_hypothesis import (
    build_noise_law_observed_twirl_experiment_results,
    build_noise_law_observed_twirl_meta_hypothesis,
    store_noise_law_observed_twirl_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_noise_law_observed_twirl/"
    "20260810_d10_preregistered/campaign_results.json"
)


def test_meta_preserves_invalid_primary_and_narrow_mechanics() -> None:
    record = build_noise_law_observed_twirl_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_invalid_isotropic_positive_control"
    )
    assert hypothesis["evidence_count"] == 10
    assert hypothesis["direct_quality_experiment_count"] == 0
    assert hypothesis["integrity_valid_experiment_count"] == 10
    assert hypothesis["subclaims"] == {
        "asymmetric_noise_law_quotient_robustness": (
            "not_tested_invalid_isotropic_positive_control"
        ),
        "analytic_isotropic_natural_utility": "failed_zero_of_five",
        "learned_isotropic_natural_utility": "passed_four_of_five",
        "learned_anisotropic_natural_utility": "failed_three_of_five",
        "learned_biased_natural_utility": "failed_zero_of_five",
        "conditional_correct_action_sufficiency": (
            "supported_all_thirty_system_law_cells"
        ),
        "conditional_correct_twirl_sufficiency": (
            "supported_all_thirty_system_law_cells"
        ),
        "orthogonal_action_specificity": (
            "supported_zero_of_thirty_system_law_cells"
        ),
        "analytic_feature_action_invariance": "supported_all_laws_shifts",
        "same_scope_retraining": "not_licensed",
    }


def test_experiment_records_keep_primary_zero_and_cell_metrics() -> None:
    record = build_noise_law_observed_twirl_meta_hypothesis(RESULTS)
    experiments = build_noise_law_observed_twirl_experiment_results(record, RESULTS)
    assert len(experiments) == 10
    assert all(item.primary_metric == 0.0 for item in experiments)
    assert all(item.model_architecture == [8, 512] for item in experiments)
    assert all(item.metrics["validity"] == 1.0 for item in experiments)
    assert all(
        item.metrics["isotropic_control_seed_pass"] == 0.0
        for item in experiments
    )
    assert max(
        item.metrics[f"{law}_maximum_action_posterior_js"]
        for item in experiments
        for law in ("isotropic", "lab_anisotropic", "lab_biased")
    ) < 0.001
    assert max(
        item.metrics[f"{law}_maximum_twirl_posterior_js"]
        for item in experiments
        for law in ("isotropic", "lab_anisotropic", "lab_biased")
    ) < 0.00025


def test_campaign_cannot_rewrite_invalid_primary_as_confirmation(
    tmp_path: Path,
) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["valid"] = True
    campaign["aggregates"]["primary_hypothesis_pass"] = True
    campaign["aggregates"]["classification"] = (
        "observed_quotient_closed_under_asymmetric_noise"
    )
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="noise-law observed-twirl campaign"):
        build_noise_law_observed_twirl_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_noise_law_observed_twirl_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_noise_law_observed_twirl_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-noise-law-observed-twirl-v1",
        "experiment_count": 10,
    }
