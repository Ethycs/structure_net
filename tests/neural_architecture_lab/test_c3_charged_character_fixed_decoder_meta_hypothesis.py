import json
from pathlib import Path

import pytest

from neural_architecture_lab.c3_charged_character_fixed_decoder_meta_hypothesis import (
    build_c3_charged_character_fixed_decoder_experiment_results,
    build_c3_charged_character_fixed_decoder_meta_hypothesis,
    store_c3_charged_character_fixed_decoder_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/"
    "tinyllm_c3_charged_character_fixed_decoder_corrective/"
    "20260811_preregistered/result.json"
)


def test_meta_confirms_exact_charged_closure_and_training_stop() -> None:
    record = build_c3_charged_character_fixed_decoder_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_exact_charged_closure_five_of_five_"
        "invariant_comparator_three_of_five"
    )
    assert hypothesis["subclaims"]["fixed_invariant_closure"] == (
        "not_supported_three_of_five"
    )
    assert hypothesis["subclaims"]["fixed_charged_closure"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["charged_oracle_fidelity"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["exact_charged_deck_closure"] == (
        "supported_zero_action_error"
    )
    assert hypothesis["subclaims"][
        "compact_typed_continuation_comparison"
    ] == "not_licensed"
    assert hypothesis["subclaims"]["unrestricted_tinyllm_training"] == (
        "not_licensed"
    )


def test_direct_evidence_preserves_fresh_cells_and_exact_action() -> None:
    record = build_c3_charged_character_fixed_decoder_meta_hypothesis(RESULTS)
    pairs = record["evidence"]["direct_tests"]
    assert len(pairs) == 5
    for pair in pairs:
        for regime in ("composition", "extrapolation"):
            cell = pair[regime]
            assert cell["valid"] is True
            assert cell["operators"]["oracle_charged_switch_drop"][
                "fixed_ceiling_pass"
            ] is True
            assert cell["operators"]["fixed_charged_switch_drop"][
                "fixed_ceiling_pass"
            ] is True
            assert cell["charged_oracle_fidelity"]["pass"] is True
            for arm in (
                "oracle_charged_switch_drop",
                "fixed_charged_switch_drop",
            ):
                assert cell["operators"][arm][
                    "maximum_deck_action_error"
                ] == 0.0


def test_experiment_results_preserve_five_charged_and_three_invariant() -> None:
    record = build_c3_charged_character_fixed_decoder_meta_hypothesis(RESULTS)
    experiments = build_c3_charged_character_fixed_decoder_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 5
    assert sum(item.primary_metric for item in experiments) == 5.0
    assert sum(item.metrics["charged_both_shifts"] for item in experiments) == 5.0
    assert sum(item.metrics["invariant_both_shifts"] for item in experiments) == 3.0
    assert sum(
        item.metrics["charged_oracle_fidelity_both_shifts"]
        for item in experiments
    ) == 5.0
    assert all(item.metrics["exact_charged_action_error"] == 0.0 for item in experiments)
    assert all(item.model_parameters == 0 for item in experiments)


def test_result_tampering_is_rejected(tmp_path: Path) -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    result["aggregates"]["charged_fixed_seed_pass_count"] = 4
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="charged-character corrective result"):
        build_c3_charged_character_fixed_decoder_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_charged_character_fixed_decoder_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_charged_character_fixed_decoder_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": (
            "tinyllm-c3-charged-character-fixed-decoder-corrective-v1"
        ),
        "experiment_count": 5,
    }
