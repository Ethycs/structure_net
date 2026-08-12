from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from experiments.structure_net import (
    tinyllm_c3_charged_character_fixed_decoder as study,
)


def test_sources_and_preregistration_are_frozen() -> None:
    assert study.validate_sources() == {
        "preregistration": study.PREREGISTRATION_SHA256,
        "switching_corrective_runner": study.PARENT_RUNNER_SHA256,
        "switching_corrective_result": study.PARENT_RESULT_SHA256,
        "switching_report": study.PARENT_REPORT_SHA256,
    }


def test_primary_seeds_are_fresh_and_streams_are_distinct() -> None:
    used = {
        *study.base.SEEDS,
        *study.parent.SEEDS,
        *study.base.acceleration.SEEDS,
        991,
        997,
    }
    assert set(study.SEEDS).isdisjoint(used)
    assert len(set(study.SEEDS)) == len(study.SEEDS)
    stream_bases = [
        *study.DATASET_SEED_BASES.values(),
        *study.PILOT_DATASET_SEED_BASES.values(),
        *study.DONOR_SEED_BASES.values(),
        *study.FRAME_SEED_BASES.values(),
        *study.SHUFFLE_SEED_BASES.values(),
    ]
    assert len(set(stream_bases)) == len(stream_bases)


def test_charged_character_has_declared_c3_type() -> None:
    dataset = study.generate_dataset(
        "composition", 997, sample_count=64, allow_pilot=True
    )
    charged, magnitude = study.charged_character(
        dataset.tokens, dataset.calibration
    )
    assert float(magnitude.min()) > 1e-12
    for element in (1, 2):
        transformed, _ = study.charged_character(
            study.base.source.apply_deck_action(dataset.tokens, element),
            dataset.calibration,
        )
        root = torch.polar(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(
                -2.0 * math.pi * element / study.base.source.CHANNELS,
                dtype=torch.float64,
            ),
        ).to(torch.complex128)
        assert float((transformed - root * charged).abs().max()) <= 2e-12


def test_lifecycle_generation_is_deterministic_and_group_typed() -> None:
    for regime in study.REGIMES:
        first = study.generate_dataset(
            regime, 997, sample_count=64, allow_pilot=True
        )
        second = study.generate_dataset(
            regime, 997, sample_count=64, allow_pilot=True
        )
        assert study.base.dataset_hash(first) == study.base.dataset_hash(second)
        assert first.dataset_seed == second.dataset_seed
        assert first.saturation_count == 0
        assert study.base.group_contracts(first)["pass"] is True


def test_lifecycle_cell_exercises_charged_joint_path() -> None:
    cell = study.analyze_cell(
        "composition",
        997,
        study.base.identifiability_audit(),
        sample_count=64,
        allow_pilot=True,
    )
    assert cell["valid"] is True
    assert cell["base_dataset_deterministic"] is True
    assert cell["corruption_deterministic"] is True
    assert cell["donor_fixed_points"] == 0
    assert cell["maximum_charged_equivariance_error"] <= 2e-12
    assert cell["stabilization_displacements"]["maximum"] <= 1e-12
    assert max(cell["continuous_prediction_maximum_errors"].values()) <= 1e-10
    assert all(arm["action_pass"] for arm in cell["operators"].values())
    assert cell["operators"]["oracle_charged_switch_drop"][
        "fixed_ceiling_pass"
    ] is True
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
    invariant_fidelity: int = 0,
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
                    "invariant_oracle_fidelity": {
                        "pass": index < invariant_fidelity
                    },
                }
            )
    return cells


def test_classifications_and_training_stop_are_locked() -> None:
    closes_gap = _classification_cells(
        invariant_oracle=5,
        invariant_fixed=3,
        charged_oracle=5,
        charged_fixed=5,
        charged_fidelity=5,
    )
    decision = study.classify(closes_gap, True)
    assert decision["classification"] == (
        "charged_character_fixed_decoder_closes_invariant_quantization_gap"
    )
    assert decision["compact_typed_continuation_comparison_licensed"] is False
    assert decision["tinyllm_training_licensed"] is False

    both_close = _classification_cells(
        invariant_oracle=5,
        invariant_fixed=4,
        charged_oracle=5,
        charged_fixed=4,
        charged_fidelity=4,
    )
    assert study.classify(both_close, True)["classification"] == (
        "both_fixed_decoders_close_fresh_switching_scope"
    )

    exceeds_fixed = _classification_cells(
        invariant_oracle=5,
        invariant_fixed=3,
        charged_oracle=5,
        charged_fixed=3,
        charged_fidelity=3,
    )
    decision = study.classify(exceeds_fixed, True)
    assert decision["classification"] == (
        "recoverable_switching_exceeds_charged_fixed_decoder"
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
        "fresh_switching_scope_not_oracle_recoverable"
    )
    assert study.classify(closes_gap, False)["classification"] == (
        "invalid_charged_character_fixed_decoder"
    )


def test_main_writes_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "status": "completed",
        "aggregates": {
            "classification": "test_classification",
            "invariant_fixed_seed_pass_count": 3,
            "charged_fixed_seed_pass_count": 5,
            "charged_oracle_fidelity_seed_pass_count": 5,
            "compact_typed_continuation_comparison_licensed": False,
        },
    }
    monkeypatch.setattr(study, "build_result", lambda: expected)
    output = tmp_path / "result.json"
    assert study.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
