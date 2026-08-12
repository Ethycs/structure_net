import json
from pathlib import Path

import pytest

from neural_architecture_lab.frozen_cyclic_answer_projection_meta_hypothesis import (
    build_frozen_cyclic_answer_projection_experiment_results,
    build_frozen_cyclic_answer_projection_meta_hypothesis,
    store_frozen_cyclic_answer_projection_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_frozen_cyclic_answer_projection/"
    "20260811_d10_learned_seed29_registered/result.json"
)


def test_meta_rejects_complete_cyclic_head_repair() -> None:
    record = build_frozen_cyclic_answer_projection_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_no_strict_order_completes_or_naturally_repairs_chart"
    )
    assert hypothesis["subclaims"]["complete_deployable_chart_repair"] == (
        "contradicted_zero_of_three_orders"
    )


def test_meta_preserves_partial_order_two_result() -> None:
    record = build_frozen_cyclic_answer_projection_meta_hypothesis(RESULTS)
    direct = record["evidence"]["direct_tests"][0]
    order_two = direct["orders"]["2"]
    assert order_two["composition"]["radius_1_oracle_pass"] is True
    assert order_two["extrapolation"]["radius_1_oracle_pass"] is True
    assert order_two["composition"]["natural_pass"] is False
    assert order_two["composition"]["radius_1_reachable_bins"] == list(range(2, 14))


def test_experiment_result_is_one_artifact_only_unit() -> None:
    record = build_frozen_cyclic_answer_projection_meta_hypothesis(RESULTS)
    experiments = build_frozen_cyclic_answer_projection_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 1
    assert experiments[0].primary_metric == 0
    assert experiments[0].metrics["complete_order_count"] == 0
    assert experiments[0].metrics["validity"] == 1


def test_result_tampering_is_rejected(tmp_path: Path) -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    result["summary"]["complete_order_count"] = 1
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen cyclic answer-projection result"):
        build_frozen_cyclic_answer_projection_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_frozen_cyclic_answer_projection_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 1


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_frozen_cyclic_answer_projection_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-frozen-cyclic-answer-projection-v1",
        "experiment_count": 1,
    }
