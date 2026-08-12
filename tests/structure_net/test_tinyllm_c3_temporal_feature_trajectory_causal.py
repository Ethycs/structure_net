from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from experiments.structure_net import (
    tinyllm_c3_temporal_feature_trajectory_causal as study,
)


RESULT_SHA256 = (
    "a0ac3315b03aa65df273539a24d8c08f51f12e8ceb702859cc16a282886ddf27"
)
RUNNER_SHA256 = (
    "bcb10634a4928848d55608f3ef3fbe054fb3ed84c145f8311af465fb8a1fd17f"
)


def test_sources_and_preregistration_are_frozen() -> None:
    record, details, hashes = study.validate_sources()
    assert record["aggregates"]["classification"] == (
        "c3_positive_control_task_failure"
    )
    assert set(details) == set(study.SEEDS)
    assert hashes["preregistration"] == study.PREREGISTRATION_SHA256
    assert hashes["fixed_operator_result"] == study.FIXED_OPERATOR_RESULT_SHA256


def test_constant_trajectory_projection_preserves_declared_anchors() -> None:
    phase = torch.linspace(-math.pi, math.pi, 128, dtype=torch.float64)
    speed = torch.linspace(-0.2, 0.2, 128, dtype=torch.float64)
    time = torch.arange(8, dtype=torch.float64)[None]
    angle = 3.0 * (phase[:, None] + speed[:, None] * time)
    carrier = torch.polar(torch.ones_like(angle), angle).to(torch.complex128)
    last = carrier[:, -1] * carrier[:, -2].conj()
    averaged = study.mean_step(carrier)
    last_projection = study.project_trajectory(carrier, last)
    mean_projection = study.project_trajectory(carrier, averaged)
    assert float((last_projection[:, 6:] - carrier[:, 6:]).abs().max()) <= 1e-12
    assert float((mean_projection - carrier).abs().max()) <= 1e-12


def test_feature_interventions_satisfy_algebra_on_small_source_cohort() -> None:
    task = study.stage0.C3TaskConfig()
    dataset = study.stage0.generate_evaluation_dataset(
        task, "composition", sample_count=32
    )
    arms, contracts = study.feature_arms(dataset, "composition")
    assert set(arms) == set(study.ARMS)
    assert contracts["pass"] is True
    assert contracts["early_derangement_fixed_points"] == 0
    assert arms["source"].shape == (32, 8, 2)


def test_feature_forward_matches_native_analytic_system() -> None:
    task = study.stage0.C3TaskConfig()
    config = study.stage0.LifecycleConfig(
        preset="tiny",
        steps=2,
        split_step=1,
        train_samples=8,
        batch_size=4,
        evaluation_samples=8,
    )
    device = torch.device("cpu")
    system = study.stage0.build_system(task, config, "analytic", device)
    dataset = study.stage0.generate_evaluation_dataset(
        task, "composition", sample_count=8
    )
    feature = system.feature(dataset.tokens, dataset.calibration)
    custom = study.posterior_from_feature(system, feature, task, device, 4)
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long)
    native = torch.softmax(
        system.task_logits(dataset.tokens, dataset.calibration, answer_ids), -1
    )
    assert float((custom - native).abs().max()) <= 2e-6


def test_posterior_effect_is_zero_only_for_identity() -> None:
    left = torch.softmax(torch.randn(64, 16), -1)
    identity = study.posterior_effect(left, left)
    assert identity == {
        "maximum_absolute_posterior_change": 0.0,
        "mean_absolute_posterior_change": 0.0,
        "mean_jensen_shannon_divergence": 0.0,
        "posterior_mean_rms_change": 0.0,
        "argmax_change_fraction": 0.0,
    }
    changed = study.posterior_effect(left, torch.roll(left, 1, 0))
    assert changed["mean_jensen_shannon_divergence"] > 0.0
    assert changed["posterior_mean_rms_change"] > 0.0


def test_population_classifications_are_locked() -> None:
    def cells(mean: int, last: int, shuffled: int, bypass: int = 5) -> list[dict]:
        values = []
        for index, seed in enumerate(study.SEEDS):
            values.append(
                {
                    "seed": seed,
                    "arm_seed_gates": {
                        "source": {
                            "true_both_shifts": index < 2,
                            "shuffled_both_shifts": False,
                        },
                        "last_consistent": {
                            "true_both_shifts": index < last,
                            "shuffled_both_shifts": False,
                        },
                        "mean_consistent": {
                            "true_both_shifts": index < mean,
                            "shuffled_both_shifts": index < shuffled,
                        },
                        "early_deranged": {
                            "true_both_shifts": False,
                            "shuffled_both_shifts": False,
                        },
                    },
                    "fixed_mean_bypass_both_shifts": index < bypass,
                }
            )
        return values

    assert study.classify(cells(4, 3, 0), True)["classification"] == (
        "all_frame_projection_repairs_frozen_continuation"
    )
    assert study.classify(cells(4, 4, 0), True)["classification"] == (
        "constant_trajectory_projection_repairs_without_all_frame_specificity"
    )
    assert study.classify(cells(3, 2, 0), True)["classification"] == (
        "fixed_operator_available_but_frozen_continuation_cannot_use_"
        "projected_trajectory"
    )
    assert study.classify(cells(5, 3, 2), True)["classification"] == (
        "feature_trajectory_specificity_failed"
    )
    assert study.classify(cells(5, 3, 0), False)["classification"] == (
        "invalid_feature_trajectory_causal_contract"
    )


def test_authoritative_result_rejects_frozen_continuation_repair() -> None:
    result = json.loads(study.PRIMARY_RESULT_PATH.read_text(encoding="utf-8"))
    assert study._sha256(study.PRIMARY_RESULT_PATH) == RESULT_SHA256
    assert result["status"] == "completed"
    assert result["schema_version"] == study.SCHEMA_VERSION
    assert result["hypothesis_id"] == study.HYPOTHESIS_ID
    assert result["source_hashes"]["runner"] == RUNNER_SHA256
    assert result["source_hashes"]["preregistration"] == (
        study.PREREGISTRATION_SHA256
    )
    assert result["aggregates"] == {
        "classification": (
            "fixed_operator_available_but_frozen_continuation_cannot_use_"
            "projected_trajectory"
        ),
        "fixed_mean_bypass_pass_count": 5,
        "last_consistent_task_pass_count": 2,
        "mean_consistent_shuffled_pass_count": 0,
        "mean_consistent_task_pass_count": 2,
        "required_seed_passes": 4,
        "shuffled_seed_pass_maximum": 1,
        "source_natural_task_pass_count": 2,
        "valid": True,
    }
    assert result["accounting"] == {
        "checkpoints_loaded": 5,
        "optimizer_steps": 0,
        "parameters_changed": 0,
        "target_using_fits": 0,
        "tinyllm_models_instantiated": 5,
    }
    assert len(result["cells"]) == 5
    for cell in result["cells"]:
        assert cell["valid"] is True
        assert cell["state_unchanged"] is True
        assert cell["fixed_mean_bypass_both_shifts"] is True
        assert cell["maximum_source_posterior_error"] <= study.REPLAY_TOLERANCE
        assert cell["maximum_source_metric_error"] <= study.REPLAY_TOLERANCE
        assert cell["arm_seed_gates"]["early_deranged"] == {
            "shuffled_both_shifts": False,
            "true_both_shifts": False,
        }
        assert cell["arm_seed_gates"]["mean_consistent"][
            "shuffled_both_shifts"
        ] is False
        for regime in study.REGIMES:
            regime_record = cell["regimes"][regime]
            assert regime_record["feature_contracts"]["pass"] is True
            assert regime_record["target_derangement_fixed_points"] == 0
            assert regime_record["fixed_mean_bypass"]["task_gate"] is True


def test_main_writes_strict_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {
        "status": "completed",
        "aggregates": {
            "classification": "test_classification",
            "source_natural_task_pass_count": 2,
            "last_consistent_task_pass_count": 2,
            "mean_consistent_task_pass_count": 2,
        },
    }
    monkeypatch.setattr(study, "build_result", lambda _device: expected)
    output = tmp_path / "result.json"
    assert study.main(["--device", "cpu", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
