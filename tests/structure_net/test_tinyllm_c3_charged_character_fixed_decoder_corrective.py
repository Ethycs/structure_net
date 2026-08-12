from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net import (
    tinyllm_c3_charged_character_fixed_decoder_corrective as study,
)


PRIMARY_RESULT_SHA256 = (
    "d8076bd7bfc42819a17438e188300e8ea0131dcc72d9ceac0bee39ede72fcaaf"
)
RUNNER_SHA256 = "5fcd691d3fd910a619bd61aa8dd0432e0050efc44699657ec98f5d7e2e01de97"


def test_sources_preserve_invalid_primary() -> None:
    assert study.validate_sources() == {
        "preregistration": study.PREREGISTRATION_SHA256,
        "invalid_preregistration": study.INVALID_PREREGISTRATION_SHA256,
        "invalid_runner": study.INVALID_RUNNER_SHA256,
        "invalid_result": study.INVALID_RESULT_SHA256,
    }
    invalid = json.loads(
        study.INVALID_RESULT_PATH.read_text(encoding="utf-8")
    )
    assert invalid["status"] == "invalid"
    assert invalid["aggregates"]["classification"] == (
        "invalid_charged_character_fixed_decoder"
    )
    assert sum(not cell["valid"] for cell in invalid["cells"]) == 7


def test_corrective_seeds_and_streams_are_fresh() -> None:
    used = {
        *study.invalid.SEEDS,
        *study.invalid.base.SEEDS,
        *study.invalid.parent.SEEDS,
        *study.invalid.base.acceleration.SEEDS,
        991,
        997,
        study.PILOT_SEED,
    }
    assert set(study.SEEDS).isdisjoint(used)
    stream_bases = [
        *study.DATASET_SEED_BASES.values(),
        *study.PILOT_DATASET_SEED_BASES.values(),
        *study.DONOR_SEED_BASES.values(),
        *study.FRAME_SEED_BASES.values(),
        *study.SHUFFLE_SEED_BASES.values(),
    ]
    assert len(set(stream_bases)) == len(stream_bases)


def test_eisenstein_pairs_are_exactly_invariant_after_neutral_products() -> None:
    dataset = study.generate_dataset(
        "composition",
        study.PILOT_SEED,
        sample_count=64,
        allow_pilot=True,
    )
    donor, frame = study.corruption_plan(
        "composition", study.PILOT_SEED, 64
    )
    corrupted = study.base.corruption.corrupt_frames(
        dataset.tokens, donor, frame
    )
    for element in (1, 2):
        transformed = study.base.corruption.corrupt_frames(
            study.base.source.apply_deck_action(dataset.tokens, element),
            donor,
            frame,
        )
        contract = study.exact_pair_contract(corrupted, transformed)
        assert contract == {
            "relative_integer_error_count": 0,
            "anchor_cube_integer_error_count": 0,
            "pass": True,
        }


def test_exact_decoder_predictions_are_bitwise_deck_invariant() -> None:
    dataset = study.generate_dataset(
        "composition",
        study.PILOT_SEED,
        sample_count=64,
        allow_pilot=True,
    )
    donor, frame = study.corruption_plan(
        "composition", study.PILOT_SEED, 64
    )
    corrupted = study.base.corruption.corrupt_frames(
        dataset.tokens, donor, frame
    )
    invariant, _ = study.base.source.analytic_carrier(
        corrupted, dataset.calibration
    )
    natural, natural_selection, _ = study.decoder_predictions(
        invariant, corrupted, dataset.switch, frame
    )
    for element in (1, 2):
        transformed_tokens = study.base.corruption.corrupt_frames(
            study.base.source.apply_deck_action(dataset.tokens, element),
            donor,
            frame,
        )
        transformed_invariant, _ = study.base.source.analytic_carrier(
            transformed_tokens, dataset.calibration
        )
        transformed, transformed_selection, _ = study.decoder_predictions(
            transformed_invariant,
            transformed_tokens,
            dataset.switch,
            frame,
        )
        for arm in (
            "oracle_charged_switch_drop",
            "fixed_charged_switch_drop",
        ):
            assert torch.equal(natural[arm], transformed[arm])
        assert torch.equal(
            natural_selection["charged"], transformed_selection["charged"]
        )


def test_corrective_lifecycle_cell_passes_all_validity_contracts() -> None:
    cell = study.analyze_cell(
        "composition",
        study.PILOT_SEED,
        study.base.identifiability_audit(),
        sample_count=64,
        allow_pilot=True,
    )
    assert cell["valid"] is True
    assert all(
        item["pass"] for item in cell["exact_integer_pair_contracts"].values()
    )
    assert max(
        operator["maximum_deck_action_error"]
        for operator in cell["operators"].values()
    ) <= 2e-12
    assert max(cell["continuous_prediction_maximum_errors"].values()) <= 1e-10
    assert cell["stabilization_displacements"]["maximum"] <= 1e-12
    assert cell["operators"]["fixed_charged_switch_drop"][
        "fixed_ceiling_pass"
    ] is True
    assert cell["charged_oracle_fidelity"]["pass"] is True


def _classification_cells(
    *,
    invariant_oracle: int,
    invariant_fixed: int,
    charged_oracle: int,
    charged_fixed: int,
    charged_fidelity: int,
) -> list[dict]:
    cells = []
    for index, seed in enumerate(study.SEEDS):
        for regime in study.REGIMES:
            cells.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "operators": {
                        "oracle_invariant_switch_drop": {
                            "fixed_ceiling_pass": index < invariant_oracle
                        },
                        "fixed_invariant_switch_drop": {
                            "fixed_ceiling_pass": index < invariant_fixed
                        },
                        "oracle_charged_switch_drop": {
                            "fixed_ceiling_pass": index < charged_oracle
                        },
                        "fixed_charged_switch_drop": {
                            "fixed_ceiling_pass": index < charged_fixed
                        },
                    },
                    "charged_oracle_fidelity": {
                        "pass": index < charged_fidelity
                    },
                    "invariant_oracle_fidelity": {"pass": False},
                }
            )
    return cells


def test_corrective_classifications_are_locked() -> None:
    closes_gap = _classification_cells(
        invariant_oracle=5,
        invariant_fixed=3,
        charged_oracle=5,
        charged_fixed=5,
        charged_fidelity=5,
    )
    decision = study.classify(closes_gap, True)
    assert decision["classification"] == (
        "exact_charged_character_corrective_closes_invariant_quantization_gap"
    )
    assert decision["compact_typed_continuation_comparison_licensed"] is False
    assert decision["tinyllm_training_licensed"] is False

    both = _classification_cells(
        invariant_oracle=5,
        invariant_fixed=4,
        charged_oracle=5,
        charged_fixed=4,
        charged_fidelity=4,
    )
    assert study.classify(both, True)["classification"] == (
        "both_exact_charged_and_invariant_fixed_decoders_close_"
        "fresh_switching_scope"
    )

    exceeds = _classification_cells(
        invariant_oracle=5,
        invariant_fixed=3,
        charged_oracle=5,
        charged_fixed=3,
        charged_fidelity=3,
    )
    decision = study.classify(exceeds, True)
    assert decision["classification"] == (
        "recoverable_switching_exceeds_exact_charged_fixed_decoder"
    )
    assert decision["compact_typed_continuation_comparison_licensed"] is True
    assert decision["tinyllm_training_licensed"] is False

    oracle_fails = _classification_cells(
        invariant_oracle=5,
        invariant_fixed=3,
        charged_oracle=3,
        charged_fixed=3,
        charged_fidelity=3,
    )
    assert study.classify(oracle_fails, True)["classification"] == (
        "fresh_corrective_switching_scope_not_oracle_recoverable"
    )
    assert study.classify(closes_gap, False)["classification"] == (
        "invalid_exact_charged_character_corrective"
    )


def test_authoritative_corrective_closes_invariant_decoder_gap() -> None:
    result = json.loads(study.PRIMARY_RESULT_PATH.read_text(encoding="utf-8"))
    assert study._sha256(study.PRIMARY_RESULT_PATH) == PRIMARY_RESULT_SHA256
    assert result["status"] == "completed"
    assert result["schema_version"] == study.SCHEMA_VERSION
    assert result["source_hashes"]["runner"] == RUNNER_SHA256
    assert result["aggregates"] == {
        "classification": (
            "exact_charged_character_corrective_closes_"
            "invariant_quantization_gap"
        ),
        "valid": True,
        "required_seed_passes": 4,
        "invariant_oracle_seed_pass_count": 5,
        "invariant_fixed_seed_pass_count": 3,
        "charged_oracle_seed_pass_count": 5,
        "charged_fixed_seed_pass_count": 5,
        "charged_oracle_fidelity_seed_pass_count": 5,
        "invariant_oracle_fidelity_seed_pass_count": 3,
        "compact_typed_continuation_comparison_licensed": False,
        "tinyllm_training_licensed": False,
    }
    assert result["accounting"] == {
        "checkpoints_loaded": 0,
        "continuous_validation_candidate_fits": 1_966_080,
        "fresh_base_examples": 40_960,
        "fresh_corrupted_evaluations": 40_960,
        "invalid_primary_examples_pooled": 0,
        "models_instantiated": 0,
        "observation_only_candidate_fits": 1_966_080,
        "optimizer_steps": 0,
        "parameters_changed": 0,
        "reusable_parameter_fits": 0,
        "target_using_fits": 0,
    }
    assert len(result["cells"]) == 10
    assert all(cell["valid"] for cell in result["cells"])
    assert all(
        contract["pass"]
        for cell in result["cells"]
        for contract in cell["exact_integer_pair_contracts"].values()
    )
    assert max(
        cell["operators"][arm]["maximum_deck_action_error"]
        for cell in result["cells"]
        for arm in (
            "oracle_charged_switch_drop",
            "fixed_charged_switch_drop",
        )
    ) == 0.0
    assert max(
        cell["continuous_prediction_maximum_errors"][arm]
        for cell in result["cells"]
        for arm in (
            "oracle_charged_switch_drop",
            "fixed_charged_switch_drop",
        )
    ) <= 1e-10
    invariant_failures = {
        cell["seed"]
        for cell in result["cells"]
        if not cell["operators"]["fixed_invariant_switch_drop"][
            "fixed_ceiling_pass"
        ]
    }
    assert invariant_failures == {577, 593}
    assert all(
        cell["operators"]["fixed_charged_switch_drop"][
            "fixed_ceiling_pass"
        ]
        for cell in result["cells"]
    )
    assert all(cell["charged_oracle_fidelity"]["pass"] for cell in result["cells"])


def test_main_writes_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "status": "completed",
        "aggregates": {
            "classification": "test_classification",
            "invariant_fixed_seed_pass_count": 4,
            "charged_fixed_seed_pass_count": 5,
            "charged_oracle_fidelity_seed_pass_count": 5,
            "compact_typed_continuation_comparison_licensed": False,
        },
    }
    monkeypatch.setattr(study, "build_result", lambda: expected)
    output = tmp_path / "result.json"
    assert study.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
