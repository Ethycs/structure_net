import json
from pathlib import Path

import pytest

from neural_architecture_lab.joint_interface_gradient_attribution_meta_hypothesis import (
    build_joint_interface_gradient_attribution_experiment_results,
    build_joint_interface_gradient_attribution_meta_hypothesis,
    store_joint_interface_gradient_attribution_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_joint_interface_gradient_attribution/"
    "20260811_d6_d10_registered_v2/campaign_results.json"
)


def test_meta_rejects_population_wide_gradient_mechanism() -> None:
    record = build_joint_interface_gradient_attribution_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_no_population_wide_registered_gradient_mechanism"
    )
    assert hypothesis["subclaims"]["population_wide_registered_mechanism"] == (
        "contradicted"
    )


def test_meta_preserves_architecture_specific_counts() -> None:
    record = build_joint_interface_gradient_attribution_meta_hypothesis(RESULTS)
    strata = record["result"]["descriptive_metrics"]["strata"]
    assert strata["d6/learned_calibrated_equivariant"][
        "initial_cross_block_starvation_count"
    ] == 5
    assert strata["d10/learned_calibrated_equivariant"][
        "initial_cross_block_starvation_count"
    ] == 2
    assert strata["d6/learned_calibrated_equivariant"][
        "persistent_learned_state_conflict_count"
    ] == 0
    assert strata["d10/learned_calibrated_equivariant"][
        "persistent_learned_state_conflict_count"
    ] == 2


def test_experiment_results_preserve_twenty_units() -> None:
    record = build_joint_interface_gradient_attribution_meta_hypothesis(RESULTS)
    experiments = build_joint_interface_gradient_attribution_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 20
    assert sum(item.primary_metric for item in experiments) == 7
    assert sum(item.metrics["persistent_learned_state_conflict"] for item in experiments) == 2
    assert all(item.metrics["validity"] == 1 for item in experiments)


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["classification"] = (
        "initial_cross_block_clip_starvation_only"
    )
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="gradient-attribution campaign"):
        build_joint_interface_gradient_attribution_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_joint_interface_gradient_attribution_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 20


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_joint_interface_gradient_attribution_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-joint-interface-gradient-attribution-v2",
        "experiment_count": 20,
    }

