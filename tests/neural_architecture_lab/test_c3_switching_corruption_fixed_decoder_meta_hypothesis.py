import json
from pathlib import Path

import pytest

from neural_architecture_lab.c3_switching_corruption_fixed_decoder_meta_hypothesis import (
    build_c3_switching_corruption_fixed_decoder_experiment_results,
    build_c3_switching_corruption_fixed_decoder_meta_hypothesis,
    store_c3_switching_corruption_fixed_decoder_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/"
    "tinyllm_c3_switching_corruption_fixed_decoder_corrective/"
    "20260811_preregistered/result.json"
)


def test_meta_preserves_valid_negative_and_narrow_license() -> None:
    record = build_c3_switching_corruption_fixed_decoder_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_fixed_decoder_full_gate_one_of_five_"
        "oracle_recoverable_five_of_five"
    )
    assert hypothesis["subclaims"]["oracle_recoverability"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["fixed_decoder_closure"] == (
        "not_supported_three_of_five"
    )
    assert hypothesis["subclaims"]["oracle_fidelity"] == (
        "not_supported_one_of_five"
    )
    assert hypothesis["subclaims"]["compact_typed_continuation_comparison"] == (
        "licensed"
    )
    assert hypothesis["subclaims"]["unrestricted_tinyllm_training"] == (
        "not_licensed"
    )


def test_direct_evidence_preserves_exact_boundary_and_fresh_cells() -> None:
    record = build_c3_switching_corruption_fixed_decoder_meta_hypothesis(RESULTS)
    pairs = record["evidence"]["direct_tests"]
    assert len(pairs) == 5
    for pair in pairs:
        for regime in ("composition", "extrapolation"):
            cell = pair[regime]
            assert cell["valid"] is True
            assert cell["dynamics_material"] is True
            assert cell["corruption_material"] is True
            assert cell["operators"]["oracle_switch_drop"][
                "fixed_ceiling_pass"
            ] is True
            assert cell["continuous_future_equivalent_count"] == 4_096


def test_experiment_results_preserve_one_complete_seed_pass() -> None:
    record = build_c3_switching_corruption_fixed_decoder_meta_hypothesis(RESULTS)
    experiments = build_c3_switching_corruption_fixed_decoder_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 5
    assert sum(item.primary_metric for item in experiments) == 1.0
    assert sum(item.metrics["oracle_both_shifts"] for item in experiments) == 5.0
    assert sum(item.metrics["fixed_both_shifts"] for item in experiments) == 3.0
    assert sum(
        item.metrics["oracle_fidelity_both_shifts"] for item in experiments
    ) == 1.0
    assert all(item.model_parameters == 0 for item in experiments)


def test_result_tampering_is_rejected(tmp_path: Path) -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    result["aggregates"]["fixed_switch_drop_seed_pass_count"] = 4
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="C3 switching-corruption result"):
        build_c3_switching_corruption_fixed_decoder_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_switching_corruption_fixed_decoder_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_switching_corruption_fixed_decoder_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": (
            "tinyllm-c3-switching-corruption-fixed-decoder-corrective-v1"
        ),
        "experiment_count": 5,
    }
