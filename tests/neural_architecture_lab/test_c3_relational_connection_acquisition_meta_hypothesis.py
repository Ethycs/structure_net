from __future__ import annotations

import json
from pathlib import Path

from neural_architecture_lab.c3_relational_connection_acquisition_meta_hypothesis import (
    build_c3_relational_connection_acquisition_experiment_results,
    build_c3_relational_connection_acquisition_meta_hypothesis,
    store_c3_relational_connection_acquisition_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_relational_connection_acquisition/"
    "20260811_preregistered/campaign_results.json"
)


def test_meta_rejects_population_acquisition_without_losing_supported_subclaims() -> None:
    record = build_c3_relational_connection_acquisition_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "rejected_population_acquisition_one_of_five_despite_exact_class"
    )
    assert hypothesis["subclaims"]["analytic_connection_ceiling"] == (
        "supported_5_of_5"
    )
    assert hypothesis["subclaims"]["learned_population_acquisition"] == (
        "rejected_1_of_5"
    )
    assert hypothesis["subclaims"]["information_specificity"] == (
        "supported_all_controls_0_of_5"
    )
    assert hypothesis["subclaims"]["posthoc_public_scale_readability"] == (
        "corrective_4_of_5_with_three_new_repairs"
    )
    assert hypothesis["subclaims"]["persistent_wrong_winding_failure"] == (
        "seed_1453"
    )
    assert hypothesis["subclaims"]["unrestricted_tinyllm_training"] == (
        "not_licensed"
    )


def test_experiment_results_preserve_five_seed_outcomes_and_zero_tinyllm() -> None:
    record = build_c3_relational_connection_acquisition_meta_hypothesis(RESULTS)
    experiments = build_c3_relational_connection_acquisition_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 5
    assert sum(item.primary_metric for item in experiments) == 1.0
    assert sum(
        item.metrics["posthoc_scalar_affine_joint_pass"] for item in experiments
    ) == 4.0
    assert all(item.model_parameters == 187 for item in experiments)
    assert all(item.metrics["all_lifecycle_valid"] == 1.0 for item in experiments)
    assert all(item.metrics["tinyllm_models_instantiated"] == 0.0 for item in experiments)


def test_meta_contains_primary_and_corrective_evidence_without_rescue() -> None:
    record = build_c3_relational_connection_acquisition_meta_hypothesis(RESULTS)
    assert len(record["evidence"]["direct_tests"]) == 5
    corrective = record["evidence"]["corrective_artifact_audit"]
    assert corrective["aggregates"]["primary_classification_unchanged"] is True
    assert corrective["aggregates"]["further_optimizer_tuning_licensed"] is False
    assert record["result"]["primary_seed_passes"] == 1
    assert record["result"]["successful_direct_experiments"] == 1
    assert record["result"]["failed_direct_experiments"] == 4


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_relational_connection_acquisition_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_relational_connection_acquisition_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c3-relational-connection-acquisition-v1",
        "experiment_count": 5,
    }
