from __future__ import annotations

import json
from pathlib import Path

from neural_architecture_lab.c3_connection_observation_identifiability_meta_hypothesis import (
    build_c3_connection_observation_identifiability_experiment_results,
    build_c3_connection_observation_identifiability_meta_hypothesis,
    store_c3_connection_observation_identifiability_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_connection_observation_identifiability/"
    "20260811_preregistered/result.json"
)


def test_meta_confirms_minimal_holonomy_and_closes_unlicensed_training() -> None:
    record = build_c3_connection_observation_identifiability_meta_hypothesis(
        RESULTS
    )
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_total_holonomy_minimal_erasure_nonidentifiable_"
        "known_noise_closed"
    )
    assert hypothesis["subclaims"]["total_holonomy_sufficiency"] == (
        "supported_10_of_10_exact_zero_error"
    )
    assert hypothesis["subclaims"]["total_holonomy_minimality"] == (
        "supported_7_of_7_exact_collisions"
    )
    assert hypothesis["subclaims"]["missing_or_partial_connection_training"] == (
        "not_licensed"
    )
    assert hypothesis["subclaims"]["known_symmetric_noise_training"] == (
        "not_licensed"
    )
    assert hypothesis["subclaims"]["unrestricted_tinyllm_training"] == (
        "not_licensed"
    )


def test_experiment_results_preserve_five_frozen_seed_replays() -> None:
    record = build_c3_connection_observation_identifiability_meta_hypothesis(
        RESULTS
    )
    experiments = build_c3_connection_observation_identifiability_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 5
    assert sum(item.primary_metric for item in experiments) == 5.0
    assert all(item.model_parameters == 187 for item in experiments)
    assert all(item.metrics["full_to_total_maximum_error"] == 0.0 for item in experiments)
    assert all(item.metrics["optimizer_steps"] == 0.0 for item in experiments)
    assert all(item.metrics["tinyllm_models_instantiated"] == 0.0 for item in experiments)


def test_meta_preserves_exact_and_population_evidence() -> None:
    record = build_c3_connection_observation_identifiability_meta_hypothesis(
        RESULTS
    )
    assert len(record["evidence"]["direct_tests"]) == 5
    assert record["evidence"]["exact_erasure_witnesses"]["pass_count"] == 7
    assert record["evidence"]["known_noise_enumeration"]["enumeration_pass"] is True
    assert record["result"]["successful_direct_experiments"] == 5
    assert record["result"]["failed_direct_experiments"] == 0


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_connection_observation_identifiability_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_connection_observation_identifiability_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c3-connection-observation-identifiability-v1",
        "experiment_count": 5,
    }
