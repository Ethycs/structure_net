from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net import tinyllm_c3_temporal_sensor_only as study
from experiments.structure_net import tinyllm_c3_temporal_quotient_training as stage0


def _small_config() -> study.SensorOnlyConfig:
    return study.SensorOnlyConfig(
        seeds=(7,),
        training_steps=2,
        train_samples=64,
        batch_size=8,
        reference_samples=32,
        evaluation_samples=32,
        required_seed_passes=1,
        device_ids=(-1,),
        gpu_slots_per_device=1,
        max_gpu_slots_per_device=1,
        max_parallel_experiments=1,
        allow_underpowered=True,
    )


def test_primary_configuration_is_frozen() -> None:
    primary = study.SensorOnlyConfig()
    assert primary.seeds == (7, 17, 29, 41, 53)
    assert primary.training_steps == 600
    assert primary.split_step == 300
    with pytest.raises(ValueError, match="primary sensor-only protocol is frozen"):
        study.SensorOnlyConfig(training_steps=598)


def test_target_permutation_preserves_pairs_and_has_no_fixed_pair() -> None:
    permutation = study.pair_preserving_target_permutation(64, 7)
    assert torch.all(permutation[0::2] % 2 == 0)
    assert torch.all(permutation[1::2] == permutation[0::2] + 1)
    assert not torch.any(permutation[0::2] == torch.arange(0, 64, 2))
    assert torch.equal(torch.sort(permutation).values, torch.arange(64))


def test_protocol_and_held_out_hashes_are_deterministic() -> None:
    config = _small_config()
    task = stage0.C3TaskConfig()
    first = study.protocol_material(task, config, 7)
    second = study.protocol_material(task, config, 7)
    assert first[3] == second[3]
    assert study.protocol_contract(first[0], first[2])["pass"] is True
    first_held = study.held_out_hashes(study.build_held_out_datasets(task, config))
    second_held = study.held_out_hashes(study.build_held_out_datasets(task, config))
    assert first_held == second_held
    assert set(first_held) == {"reference", "composition", "extrapolation"}


def test_orthogonal_gauge_recovers_one_global_rotation() -> None:
    angle = torch.tensor(0.7)
    analytic = torch.polar(
        torch.ones(128), torch.linspace(-3.0, 3.0, 128)
    ).to(torch.complex64)
    learned = analytic * torch.polar(torch.ones(()), angle)
    fitted = study.fit_orthogonal_gauge(learned, analytic)
    assert fitted["reference_mean_dot"] >= 0.999999
    assert fitted["reference_coordinate_rmse"] <= 1e-6
    assert fitted["determinant"] == pytest.approx(1.0, abs=1e-6)


def test_task_gate_is_joint_and_threshold_inclusive() -> None:
    composition = {
        "posterior_mean_correlation": 0.90,
        "exact_bin_accuracy": 0.50,
        "target_cross_entropy": 1.80,
        "predicted_bin_coverage": 14,
    }
    extrapolation = {
        "posterior_mean_correlation": 0.90,
        "exact_bin_accuracy": 0.35,
        "target_cross_entropy": 2.20,
        "predicted_bin_coverage": 12,
    }
    assert study._task_gate(composition, "composition") is True
    assert study._task_gate(extrapolation, "extrapolation") is True
    extrapolation["target_cross_entropy"] = 2.200001
    assert study._task_gate(extrapolation, "extrapolation") is False


def test_aggregate_classifications_are_locked() -> None:
    primary = study.SensorOnlyConfig()

    def cells(true_count: int, shuffled_count: int) -> list[dict]:
        return [
            {
                "seed": seed,
                "gates": {
                    "validity": True,
                    "analytic_positive_control": True,
                    "learned_true_joint": index < true_count,
                    "learned_target_shuffled_joint": index < shuffled_count,
                },
            }
            for index, seed in enumerate(primary.seeds)
        ]

    assert study.aggregate_details(cells(4, 0), primary)["classification"] == (
        "task_only_sensor_acquisition_supported"
    )
    assert study.aggregate_details(cells(3, 0), primary)["classification"] == (
        "function_class_present_but_task_only_sensor_optimization_unreliable"
    )
    assert study.aggregate_details(cells(5, 2), primary)["classification"] == (
        "sensor_acquisition_specificity_failed"
    )


def test_two_arm_cpu_lifecycle_reload_and_exact_resume(tmp_path: Path) -> None:
    config = _small_config()
    sources, _license = study._validate_sources()
    implementation = study._implementation_digest(sources)
    task = stage0.C3TaskConfig()
    held_out = study.build_held_out_datasets(task, config)
    evaluation_hashes = study.held_out_hashes(held_out)
    experiment = study._experiment(
        config, 7, tmp_path, implementation, evaluation_hashes
    )
    result = study.sensor_only_worker(experiment, -1)
    assert result.error is None
    detail_path = tmp_path / "runs" / "seed_7" / "result.json"
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    assert detail["status"] == "completed"
    assert detail["gates"]["validity"] is True
    assert detail["gates"]["matched_initialization"] is True
    assert detail["accounting"]["tinyllm_models_instantiated"] == 0
    for arm in study.ARMS:
        assert detail["arms"][arm]["state_changed"] is True
        assert detail["arms"][arm]["reload"]["pass"] is True
        assert detail["arms"][arm]["exact_resume"]["pass"] is True
        for point in ("midpoint", "final"):
            artifact = detail["artifacts"]["checkpoints"][arm][point]
            assert Path(artifact["path"]).is_file()
            assert study._sha256(Path(artifact["path"])) == artifact["sha256"]
