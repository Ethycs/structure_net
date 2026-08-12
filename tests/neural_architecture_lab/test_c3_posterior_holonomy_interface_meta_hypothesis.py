from __future__ import annotations

import json
from pathlib import Path

from neural_architecture_lab.c3_posterior_holonomy_interface_meta_hypothesis import (
    build_c3_posterior_holonomy_interface_experiment_results,
    build_c3_posterior_holonomy_interface_meta_hypothesis,
    store_c3_posterior_holonomy_interface_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_posterior_holonomy_interface/"
    "20260811_preregistered/result.json"
)


def test_meta_confirms_exact_soft_interface_without_licensing_training() -> None:
    record = build_c3_posterior_holonomy_interface_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_exact_c3_soft_holonomy_moment_interface"
    )
    assert hypothesis["subclaims"]["c3_posterior_moment_invertibility"] == (
        "supported_2080_of_2080"
    )
    assert hypothesis["subclaims"]["bayes_mean_and_risk_factorization"] == (
        "supported_4259840_posterior_phase_cells"
    )
    assert hypothesis["subclaims"]["frozen_soft_interface"] == (
        "supported_10_of_10"
    )
    assert hypothesis["subclaims"]["posterior_estimator_training"] == (
        "not_yet_licensed"
    )
    assert hypothesis["subclaims"]["unrestricted_tinyllm_training"] == (
        "not_licensed"
    )


def test_experiment_results_preserve_five_frozen_seed_replays() -> None:
    record = build_c3_posterior_holonomy_interface_meta_hypothesis(RESULTS)
    experiments = build_c3_posterior_holonomy_interface_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 5
    assert sum(item.primary_metric for item in experiments) == 5.0
    assert all(item.metrics["frozen_replay_maximum_error"] <= 1e-6 for item in experiments)
    assert all(item.metrics["optimizer_steps"] == 0.0 for item in experiments)
    assert all(item.metrics["tinyllm_models_instantiated"] == 0.0 for item in experiments)


def test_meta_preserves_exhaustive_and_frozen_evidence() -> None:
    record = build_c3_posterior_holonomy_interface_meta_hypothesis(RESULTS)
    assert len(record["evidence"]["direct_tests"]) == 5
    assert record["evidence"]["exhaustive_simplex"]["simplex_point_count"] == 2_080
    assert record["evidence"]["exhaustive_mean_and_risk"][
        "simplex_phase_cell_count"
    ] == 4_259_840
    assert record["result"]["successful_direct_experiments"] == 5
    assert record["result"]["failed_direct_experiments"] == 0


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_posterior_holonomy_interface_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_posterior_holonomy_interface_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c3-posterior-holonomy-interface-v1",
        "experiment_count": 5,
    }
