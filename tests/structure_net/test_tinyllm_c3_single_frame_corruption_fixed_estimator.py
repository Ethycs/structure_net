from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from experiments.structure_net import (
    tinyllm_c3_single_frame_corruption_fixed_estimator as study,
)


PRIMARY_RESULT_SHA256 = (
    "59681f2764b988f05b0916965898b87d5b233b2165151fe19e0c97391fe467b9"
)
RUNNER_SHA256 = "8600081fc813ae6da5a49c39222bf3a94286127d7238899d61ec9acd1c6d31f8"


def test_sources_and_preregistration_are_frozen() -> None:
    hashes, cohort_hashes = study.validate_sources()
    assert hashes == {
        "preregistration": study.PREREGISTRATION_SHA256,
        "constant_acceleration_runner": study.ACCELERATION_RUNNER_SHA256,
        "constant_acceleration_result": study.ACCELERATION_RESULT_SHA256,
        "generator": study.GENERATOR_SHA256,
        "interval_likelihood": study.INTERVAL_SHA256,
    }
    assert set(cohort_hashes) == {
        (seed, regime) for seed in study.SEEDS for regime in study.REGIMES
    }


def test_corruption_plan_is_deterministic_deranged_and_replaces_one_frame() -> None:
    dataset = study.acceleration.generate_dataset(
        "composition", 991, sample_count=64, allow_pilot=True
    )
    first_donor, first_frame = study.corruption_plan("composition", 991, 64)
    second_donor, second_frame = study.corruption_plan("composition", 991, 64)
    assert torch.equal(first_donor, second_donor)
    assert torch.equal(first_frame, second_frame)
    assert int((first_donor == torch.arange(64)).sum()) == 0
    corrupted = study.corrupt_frames(dataset.tokens, first_donor, first_frame)
    row = torch.arange(64)
    assert torch.equal(
        corrupted[row, first_frame], dataset.tokens[first_donor, first_frame]
    )
    keep = torch.ones_like(dataset.tokens, dtype=torch.bool)
    keep[row, first_frame] = False
    assert torch.equal(corrupted[keep], dataset.tokens[keep])


def test_robust_deletion_is_exact_for_continuous_quadratic_phase() -> None:
    count = 128
    generator = torch.Generator(device="cpu").manual_seed(7129)
    phase = 2.0 * math.pi * torch.rand(
        count, dtype=torch.float64, generator=generator
    )
    velocity = torch.empty(count, dtype=torch.float64).uniform_(
        -0.15, 0.15, generator=generator
    )
    acceleration = torch.empty(count, dtype=torch.float64).uniform_(
        -0.035, 0.035, generator=generator
    )
    time = torch.arange(9, dtype=torch.float64)[None]
    angle = 3.0 * (
        phase[:, None]
        + velocity[:, None] * time
        + 0.5 * acceleration[:, None] * time * (time - 1.0)
    )
    carrier = torch.polar(torch.ones_like(angle), angle).to(torch.complex128)
    donor = study.fixed.sattolo_derangement(count, 9917)
    frame = torch.randint(
        0,
        study.source.TIME_STEPS,
        (count,),
        generator=torch.Generator(device="cpu").manual_seed(9919),
    )
    corrupted = study.corrupt_frames(carrier[:, :8], donor, frame)
    oracle, robust, selected, _ = study.deletion_predictions(corrupted, frame)
    assert torch.equal(selected, frame)
    assert float((oracle - carrier[:, 8]).abs().max()) <= 1e-10
    assert float((robust - carrier[:, 8]).abs().max()) <= 1e-10


def test_disjoint_pilot_exercises_the_complete_analysis_path() -> None:
    cell = study.analyze_cell(
        "composition", 991, sample_count=64, allow_pilot=True
    )
    assert cell["valid"] is True
    assert cell["corruption_deterministic"] is True
    assert cell["donor_fixed_points"] == 0
    assert sum(cell["corruption_equivariance_token_errors"].values()) == 0
    assert cell["continuous_corruption_index_recovery_count"] == 64
    assert max(cell["continuous_prediction_maximum_errors"].values()) <= 1e-10
    assert cell["minimum_clean_chart_margin"] >= study.CHART_MARGIN_MINIMUM
    assert all(
        operator["action_pass"] for operator in cell["operators"].values()
    )


def test_fixed_ceiling_gate_is_joint_and_inclusive() -> None:
    composition = {
        "posterior_mean_correlation": 0.90,
        "exact_bin_accuracy": 0.90,
        "target_cross_entropy": 1.80,
        "predicted_bin_coverage": 14,
    }
    scalar = {"rmse": 0.020}
    assert study.acceleration._fixed_ceiling_gate(
        scalar, composition, "composition"
    ) is True
    composition["exact_bin_accuracy"] = 0.89999
    assert study.acceleration._fixed_ceiling_gate(
        scalar, composition, "composition"
    ) is False


def _classification_cells(
    *,
    naive: int,
    oracle: int,
    robust: int,
    material: int,
    repair: int,
    fidelity: int,
) -> list[dict]:
    cells = []
    for index, seed in enumerate(study.SEEDS):
        for regime in study.REGIMES:
            cells.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "operators": {
                        "clean_all_frame_degree2": {
                            "fixed_ceiling_pass": True
                        },
                        "corrupted_all_frame_degree2": {
                            "fixed_ceiling_pass": index < naive
                        },
                        "oracle_drop_one_quadratic": {
                            "fixed_ceiling_pass": index < oracle
                        },
                        "robust_drop_one_quadratic": {
                            "fixed_ceiling_pass": index < robust
                        },
                    },
                    "corruption_material": index < material,
                    "robust_repair": {"pass": index < repair},
                    "oracle_fidelity": {"pass": index < fidelity},
                }
            )
    return cells


def test_population_classifications_are_locked() -> None:
    close = _classification_cells(
        naive=0, oracle=5, robust=5, material=5, repair=5, fidelity=5
    )
    assert study.classify(close, True)["classification"] == (
        "fixed_robust_estimator_closes_single_frame_corruption"
    )
    weak = _classification_cells(
        naive=5, oracle=5, robust=5, material=0, repair=0, fidelity=5
    )
    assert study.classify(weak, True)["classification"] == (
        "single_frame_corruption_not_material"
    )
    recoverable = _classification_cells(
        naive=0, oracle=5, robust=3, material=5, repair=3, fidelity=3
    )
    assert study.classify(recoverable, True)["classification"] == (
        "recoverable_corruption_exceeds_registered_fixed_estimator"
    )
    unrecoverable = _classification_cells(
        naive=0, oracle=3, robust=3, material=5, repair=3, fidelity=3
    )
    assert study.classify(unrecoverable, True)["classification"] == (
        "declared_corruption_not_recoverable_at_required_ceiling"
    )
    assert study.classify(close, False)["classification"] == (
        "invalid_single_frame_corruption_preflight"
    )


def test_authoritative_result_closes_single_frame_corruption_training() -> None:
    result = json.loads(study.PRIMARY_RESULT_PATH.read_text(encoding="utf-8"))
    assert study._sha256(study.PRIMARY_RESULT_PATH) == PRIMARY_RESULT_SHA256
    assert result["status"] == "completed"
    assert result["schema_version"] == study.SCHEMA_VERSION
    assert result["source_hashes"]["runner"] == RUNNER_SHA256
    assert result["aggregates"] == {
        "classification": "fixed_robust_estimator_closes_single_frame_corruption",
        "clean_fixed_ceiling_seed_pass_count": 5,
        "compact_robust_sensor_comparison_licensed": False,
        "corrupted_naive_fixed_ceiling_seed_pass_count": 0,
        "corruption_material_seed_pass_count": 5,
        "oracle_fidelity_seed_pass_count": 5,
        "oracle_fixed_ceiling_seed_pass_count": 5,
        "required_seed_passes": 4,
        "robust_fixed_ceiling_seed_pass_count": 5,
        "robust_repair_seed_pass_count": 5,
        "tinyllm_training_licensed": False,
        "valid": True,
    }
    assert result["accounting"] == {
        "checkpoints_loaded": 0,
        "closed_form_observation_only_fits": 368_640,
        "matched_base_examples_replayed": 40_960,
        "models_instantiated": 0,
        "new_base_evaluation_examples": 0,
        "new_corrupted_evaluations": 40_960,
        "optimizer_steps": 0,
        "parameters_changed": 0,
        "target_using_fits": 0,
    }
    assert len(result["cells"]) == 10
    for cell in result["cells"]:
        assert cell["valid"] is True
        assert cell["base_dataset_hash_replayed"] is True
        assert cell["corruption_deterministic"] is True
        assert cell["donor_fixed_points"] == 0
        assert cell["minimum_frame_count"] >= study.MINIMUM_FRAME_COUNT
        assert cell["minimum_clean_chart_margin"] >= study.CHART_MARGIN_MINIMUM
        assert cell["continuous_corruption_index_recovery_count"] == 4_096
        assert max(cell["continuous_prediction_maximum_errors"].values()) <= 1e-10
        assert sum(cell["corruption_equivariance_token_errors"].values()) == 0
        assert cell["corruption_material"] is True
        assert cell["operators"]["clean_all_frame_degree2"][
            "fixed_ceiling_pass"
        ] is True
        assert cell["operators"]["corrupted_all_frame_degree2"][
            "fixed_ceiling_pass"
        ] is False
        for arm in ("oracle_drop_one_quadratic", "robust_drop_one_quadratic"):
            assert cell["operators"][arm]["fixed_ceiling_pass"] is True
        assert cell["robust_repair"]["pass"] is True
        assert cell["oracle_fidelity"]["pass"] is True
        assert cell["quantized_corruption_index_recovery_rate"] >= 0.99


def test_main_writes_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "status": "completed",
        "aggregates": {
            "classification": "test_classification",
            "corrupted_naive_fixed_ceiling_seed_pass_count": 0,
            "oracle_fixed_ceiling_seed_pass_count": 5,
            "robust_fixed_ceiling_seed_pass_count": 5,
            "robust_repair_seed_pass_count": 5,
        },
    }
    monkeypatch.setattr(study, "build_result", lambda: expected)
    output = tmp_path / "result.json"
    assert study.main(["--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8").endswith("\n")
