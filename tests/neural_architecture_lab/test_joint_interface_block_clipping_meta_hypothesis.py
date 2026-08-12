import json
from pathlib import Path

import pytest

from neural_architecture_lab.joint_interface_block_clipping_meta_hypothesis import (
    build_joint_interface_block_clipping_experiment_results,
    build_joint_interface_block_clipping_meta_hypothesis,
    store_joint_interface_block_clipping_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_joint_interface_block_clipping/"
    "20260811_d6_d10_preregistered/campaign_results.json"
)


def test_meta_rejects_parameter_block_clipping_repair() -> None:
    record = build_joint_interface_block_clipping_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_physical_true_zero_of_five_both_presets"
    )
    assert hypothesis["subclaims"]["architecture_family_block_clipping_repair"] == (
        "contradicted_zero_of_ten"
    )


def test_meta_preserves_population_and_control_counts() -> None:
    record = build_joint_interface_block_clipping_meta_hypothesis(RESULTS)
    strata = record["result"]["descriptive_metrics"]["strata"]
    for preset in ("d6", "d10"):
        assert strata[preset]["physical_true_pass_count"] == 0
        assert strata[preset]["pair_shuffled_pass_count"] == 0
        assert strata[preset]["valid_count"] == 5


def test_experiment_results_preserve_ten_seed_units() -> None:
    record = build_joint_interface_block_clipping_meta_hypothesis(RESULTS)
    experiments = build_joint_interface_block_clipping_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 10
    assert sum(item.primary_metric for item in experiments) == 0
    assert all(item.metrics["validity"] == 1 for item in experiments)
    assert all(item.metrics["pair_shuffled_joint_seed_pass"] == 0 for item in experiments)


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["classification"] = (
        "parameter_block_clipping_repairs_physical_interface"
    )
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="block-clipping campaign"):
        build_joint_interface_block_clipping_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_joint_interface_block_clipping_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_joint_interface_block_clipping_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-joint-interface-block-clipping-v1",
        "experiment_count": 10,
    }

