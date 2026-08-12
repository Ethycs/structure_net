from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from experiments.structure_net import (
    tinyllm_c3_hidden_order_corruption_fixed_estimator as study,
)


PRIMARY_RESULT_SHA256 = (
    "88e44e76c44d654331ea647ccedd8dad505030a65ab7c4ac241d8b98d98bd02e"
)
RUNNER_SHA256 = "318cf81497960d37f86b1be58af3d819076e7dc9b620fe2ba73ae215ae0adea7"


def test_sources_and_preregistration_are_frozen() -> None:
    hashes, speed_hashes, acceleration_hashes = study.validate_sources()
    assert hashes == {
        "preregistration": study.PREREGISTRATION_SHA256,
        "constant_speed_runner": study.SPEED_RUNNER_SHA256,
        "constant_speed_result": study.SPEED_RESULT_SHA256,
        "single_frame_corruption_runner": study.CORRUPTION_RUNNER_SHA256,
        "single_frame_corruption_result": study.CORRUPTION_RESULT_SHA256,
    }
    assert set(speed_hashes) == {
        (pair[0], regime) for pair in study.PAIR_SEEDS for regime in study.REGIMES
    }
    assert set(acceleration_hashes) == {
        (pair[1], regime)
        for pair in study.PAIR_SEEDS
        for regime in study.REGIMES
    }


def test_degree_one_is_exactly_nested_in_common_quadratic_chart() -> None:
    count = 128
    generator = torch.Generator(device="cpu").manual_seed(7193)
    phase = 2.0 * math.pi * torch.rand(
        count, dtype=torch.float64, generator=generator
    )
    velocity = torch.empty(count, dtype=torch.float64).uniform_(
        -0.18, 0.18, generator=generator
    )
    time = torch.arange(8, dtype=torch.float64)[None]
    angle = 3.0 * (phase[:, None] + velocity[:, None] * time)
    carrier = torch.polar(torch.ones_like(angle), angle).to(torch.complex128)
    frame = torch.randint(
        0,
        8,
        (count,),
        generator=torch.Generator(device="cpu").manual_seed(7199),
    )
    coefficients = study._true_deletion_coefficients(carrier, frame)
    assert float(coefficients[:, 2].abs().max()) <= 1e-12


def test_primary_robust_prediction_does_not_branch_on_law_family() -> None:
    phase = torch.linspace(-2.0, 2.0, 64, dtype=torch.float64)
    velocity = torch.linspace(-0.12, 0.12, 64, dtype=torch.float64)
    acceleration = torch.linspace(-0.02, 0.02, 64, dtype=torch.float64)
    time = torch.arange(8, dtype=torch.float64)[None]
    angle = 3.0 * (
        phase[:, None]
        + velocity[:, None] * time
        + 0.5 * acceleration[:, None] * time * (time - 1.0)
    )
    clean = torch.polar(torch.ones_like(angle), angle).to(torch.complex128)
    donor = study.speed.sattolo_derangement(64, 7211)
    frame = torch.arange(64) % 8
    corrupted = study.corruption.corrupt_frames(clean, donor, frame)
    speed_predictions, speed_selected = study.common_predictions(
        "constant_speed", clean, corrupted, frame
    )
    acceleration_predictions, acceleration_selected = study.common_predictions(
        "constant_acceleration", clean, corrupted, frame
    )
    assert torch.equal(speed_selected, acceleration_selected)
    for arm in ("oracle_drop_one_quadratic", "robust_drop_one_quadratic"):
        assert torch.equal(speed_predictions[arm], acceleration_predictions[arm])


def test_disjoint_pilot_exercises_both_laws_and_actual_mixture() -> None:
    cell = study.analyze_cell(
        1, "composition", sample_count=64, allow_pilot=True
    )
    assert cell["valid"] is True
    assert cell["common_robust_endpoint_pass"] is True
    assert cell["materiality_pass"] is True
    assert cell["repair_pass"] is True
    assert cell["oracle_fidelity_pass"] is True
    assert cell["populations"]["mixture"]["sample_count"] == 128
    for family in study.FAMILIES:
        population = cell["populations"][family]
        assert population["valid"] is True
        assert population["continuous_law_inclusion_maximum_error"] <= 1e-10
        assert population["continuous_corruption_index_recovery_count"] == 64
        assert max(population["continuous_prediction_maximum_errors"].values()) <= 1e-10


def _classification_cells(
    *,
    common: int,
    oracle: int,
    material: int,
    repair: int,
    fidelity: int,
    naive: int,
) -> list[dict]:
    cells = []
    for index in range(5):
        for regime in study.REGIMES:
            cells.append(
                {
                    "replicate": index + 1,
                    "regime": regime,
                    "common_robust_endpoint_pass": index < common,
                    "oracle_endpoint_pass": index < oracle,
                    "materiality_pass": index < material,
                    "repair_pass": index < repair,
                    "oracle_fidelity_pass": index < fidelity,
                    "populations": {
                        "mixture": {
                            "operators": {
                                "corrupted_law_specific": {
                                    "fixed_ceiling_pass": index < naive
                                }
                            }
                        }
                    },
                }
            )
    return cells


def test_population_classifications_are_locked() -> None:
    close = _classification_cells(
        common=5, oracle=5, material=5, repair=5, fidelity=5, naive=0
    )
    assert study.classify(close, True)["classification"] == (
        "single_robust_quadratic_closes_hidden_order_and_corruption"
    )
    weak = _classification_cells(
        common=5, oracle=5, material=0, repair=0, fidelity=5, naive=5
    )
    assert study.classify(weak, True)["classification"] == (
        "hidden_order_corruption_not_material"
    )
    selector = _classification_cells(
        common=3, oracle=5, material=5, repair=3, fidelity=3, naive=0
    )
    assert study.classify(selector, True)["classification"] == (
        "recoverable_hidden_order_corruption_exceeds_common_fixed_estimator"
    )
    unidentifiable = _classification_cells(
        common=3, oracle=3, material=5, repair=3, fidelity=3, naive=0
    )
    assert study.classify(unidentifiable, True)["classification"] == (
        "hidden_order_corruption_not_recoverable_at_required_ceiling"
    )
    assert study.classify(close, False)["classification"] == (
        "invalid_hidden_order_corruption_preflight"
    )


def test_authoritative_result_is_valid_but_registered_inconclusive() -> None:
    result = json.loads(study.PRIMARY_RESULT_PATH.read_text(encoding="utf-8"))
    assert study._sha256(study.PRIMARY_RESULT_PATH) == PRIMARY_RESULT_SHA256
    assert result["status"] == "completed"
    assert result["schema_version"] == study.SCHEMA_VERSION
    assert result["source_hashes"]["runner"] == RUNNER_SHA256
    assert result["aggregates"] == {
        "classification": "inconclusive_hidden_order_corruption_preflight",
        "common_robust_endpoint_seed_pass_count": 5,
        "compact_typed_selector_comparison_licensed": False,
        "corrupted_naive_mixture_ceiling_seed_pass_count": 0,
        "materiality_seed_pass_count": 5,
        "oracle_endpoint_seed_pass_count": 5,
        "oracle_fidelity_seed_pass_count": 5,
        "repair_seed_pass_count": 0,
        "required_seed_passes": 4,
        "tinyllm_training_licensed": False,
        "valid": True,
    }
    assert result["accounting"] == {
        "adaptive_selector_decisions": 0,
        "checkpoints_loaded": 0,
        "closed_form_observation_only_fits": 737_280,
        "law_label_reads_by_primary_estimator": 0,
        "matched_base_examples_replayed": 81_920,
        "models_instantiated": 0,
        "new_corrupted_evaluations": 81_920,
        "optimizer_steps": 0,
        "parameters_changed": 0,
        "target_using_fits": 0,
    }
    assert len(result["cells"]) == 10
    for cell in result["cells"]:
        assert cell["valid"] is True
        assert cell["common_robust_endpoint_pass"] is True
        assert cell["oracle_endpoint_pass"] is True
        assert cell["materiality_pass"] is True
        assert cell["repair_pass"] is False
        assert cell["oracle_fidelity_pass"] is True
        speed_population = cell["populations"]["constant_speed"]
        assert speed_population["robust_repair"]["pass"] is False
        assert speed_population["robust_repair"]["accuracy_delta"] < 0.20
        assert speed_population["robust_repair"]["rmse_ratio"] < 0.02
        for population_name, population in cell["populations"].items():
            assert population["valid"] is True
            assert population["corruption_material"] is True
            assert population["operators"]["clean_law_specific"][
                "fixed_ceiling_pass"
            ] is True
            assert population["operators"]["corrupted_law_specific"][
                "fixed_ceiling_pass"
            ] is False
            for arm in (
                "oracle_drop_one_quadratic",
                "robust_drop_one_quadratic",
            ):
                assert population["operators"][arm]["fixed_ceiling_pass"] is True
            assert population["oracle_fidelity"]["pass"] is True
        assert cell["populations"]["constant_acceleration"]["robust_repair"][
            "pass"
        ] is True
        assert cell["populations"]["mixture"]["robust_repair"]["pass"] is True


def test_main_writes_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "status": "completed",
        "aggregates": {
            "classification": "test_classification",
            "common_robust_endpoint_seed_pass_count": 5,
            "materiality_seed_pass_count": 5,
            "repair_seed_pass_count": 5,
            "oracle_fidelity_seed_pass_count": 5,
        },
    }
    monkeypatch.setattr(study, "build_result", lambda: expected)
    output = tmp_path / "result.json"
    assert study.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
