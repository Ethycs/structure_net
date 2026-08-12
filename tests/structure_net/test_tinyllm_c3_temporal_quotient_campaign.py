from __future__ import annotations

from pathlib import Path

import pytest

from experiments.structure_net.tinyllm_c3_temporal_quotient_campaign import (
    ANALYSIS_SHA256,
    ARMS,
    PREREGISTRATION_SHA256,
    SEEDS,
    STAGE0_RUNNER_SHA256,
    C3CampaignConfig,
    _dataset_contract,
    _experiments,
    _implementation_digest,
    _source_hashes,
    aggregate_details,
    causal_pass,
    natural_task_pass,
    relative_utility,
    representation_pass,
)
from experiments.structure_net.tinyllm_c3_temporal_quotient_training import (
    C3TaskConfig,
)


def _metrics(accuracy: float = 0.8) -> dict:
    return {
        regime: {
            "posterior_mean_correlation": 0.95,
            "exact_bin_accuracy": accuracy,
            "target_cross_entropy": 1.5,
            "predicted_bin_coverage": 16,
        }
        for regime in ("composition", "extrapolation")
    }


def _detail(arm: str, seed: int, *, joint: bool = True, deranged: bool = False) -> dict:
    return {
        "arm": arm,
        "seed": seed,
        "status": "completed",
        "task_metrics": _metrics(),
        "gates": {
            "validity": True,
            "natural_task": joint,
            "representation": joint if arm != "raw" else False,
            "causal_all_cuts": joint if arm != "raw" else False,
            "exact_action": True,
            "identity_replay": True,
            "target_derangement_pass_by_regime": {
                "composition": deranged,
                "extrapolation": deranged,
            },
            "joint_without_relative_utility": joint if arm != "raw" else True,
        },
    }


def test_primary_configuration_and_sources_are_locked() -> None:
    C3CampaignConfig()
    with pytest.raises(ValueError, match="primary C3 d6"):
        C3CampaignConfig(training_steps=599)
    sources = _source_hashes()
    assert sources["preregistration"] == PREREGISTRATION_SHA256
    assert sources["stage0_runner"] == STAGE0_RUNNER_SHA256
    assert sources["analysis"] == ANALYSIS_SHA256
    assert len(_implementation_digest(sources)) == 64


def test_fresh_dataset_contract_has_disjoint_fit_and_final_splits() -> None:
    config = C3CampaignConfig(
        arms=("analytic",),
        seeds=(7,),
        training_steps=2,
        train_samples=32,
        batch_size=8,
        probe_train_latents=32,
        probe_validation_latents=16,
        probe_test_latents=24,
        probe_steps=4,
        required_seed_passes=1,
        allow_underpowered=True,
    )
    contract = _dataset_contract(C3TaskConfig(), config)
    assert contract["split_contract"]["pass"] is True
    assert contract["split_contract"]["fit_final_disjoint"] is True
    assert set(contract["probe_hashes"]) == {
        "train",
        "validation",
        "composition",
        "extrapolation",
    }


def test_numeric_task_representation_and_causal_gates_are_joint() -> None:
    config = C3CampaignConfig()
    metrics = _metrics()
    assert natural_task_pass(metrics, config)
    metrics["extrapolation"]["posterior_mean_correlation"] = 0.899
    assert not natural_task_pass(metrics, config)

    representation = {"cuts": {}}
    for cut in ("frontend", "full"):
        representation["cuts"][cut] = {
            "semantic": {
                "evaluations": {
                    regime: {"target_correlation": 0.91}
                    for regime in ("composition", "extrapolation")
                }
            },
            "conditional_deck": {
                "evaluations": {
                    regime: {
                        "balanced_accuracy": 0.38,
                        "conditional_log_loss_gain": 0.019,
                    }
                    for regime in ("composition", "extrapolation")
                }
            },
        }
    assert representation_pass(representation, config)
    representation["cuts"]["full"]["conditional_deck"]["evaluations"][
        "extrapolation"
    ]["conditional_log_loss_gain"] = 0.021
    assert not representation_pass(representation, config)

    causal = {
        "cuts": {
            cut: {
                "orbit_barycenter_preservation": {"pass": True},
                "maximum_identity_replay_logit_error": 1e-6,
            }
            for cut in ("frontend", "post_attention", "post_mlp", "full")
        }
    }
    assert causal_pass(causal, config)
    causal["cuts"]["full"]["maximum_identity_replay_logit_error"] = 3e-6
    assert not causal_pass(causal, config)


def test_aggregate_requires_both_structured_populations_and_controls() -> None:
    config = C3CampaignConfig()
    details = [_detail(arm, seed) for arm in ARMS for seed in SEEDS]
    aggregate = aggregate_details(details, config)
    assert aggregate["classification"] == "c3_d6_structured_quotient_supported"
    assert aggregate["primary_hypothesis_pass"] is True
    assert aggregate["arms"]["analytic"]["joint_pass_count"] == 5
    assert aggregate["arms"]["learned_c3"]["joint_pass_count"] == 5

    for item in details:
        if item["arm"] == "learned_c3" and item["seed"] in (7, 17):
            item["gates"]["joint_without_relative_utility"] = False
            item["gates"]["representation"] = False
    aggregate = aggregate_details(details, config)
    assert aggregate["classification"] == "c3_architectural_invariance_not_learned_useful"
    assert aggregate["primary_hypothesis_pass"] is False

    details = [_detail(arm, seed) for arm in ARMS for seed in SEEDS]
    for item in details:
        if item["arm"] == "analytic" and item["seed"] in (7, 17):
            item["gates"]["joint_without_relative_utility"] = False
            item["gates"]["natural_task"] = False
    aggregate = aggregate_details(details, config)
    assert aggregate["classification"] == "c3_positive_control_task_failure"


def test_target_changing_population_control_cannot_be_outvoted() -> None:
    config = C3CampaignConfig()
    details = [_detail(arm, seed) for arm in ARMS for seed in SEEDS]
    for item in details:
        if item["arm"] == "learned_c3" and item["seed"] in (7, 17):
            item["gates"]["target_derangement_pass_by_regime"]["composition"] = True
    aggregate = aggregate_details(details, config)
    assert aggregate["valid"] is True
    assert aggregate["controls_pass"] is False
    assert aggregate["classification"] == "invalid"


def test_relative_utility_applies_only_when_raw_is_adequate() -> None:
    config = C3CampaignConfig()
    structured = _detail("analytic", 7)
    raw = _detail("raw", 7)
    raw["task_metrics"] = _metrics(0.9)
    structured["task_metrics"] = _metrics(0.85)
    comparison = relative_utility(structured, raw, config)
    assert comparison["applicable"] is True
    assert comparison["pass"] is False
    raw["gates"]["natural_task"] = False
    comparison = relative_utility(structured, raw, config)
    assert comparison["applicable"] is False
    assert comparison["pass"] is True


def test_primary_grid_has_fifteen_fingerprinted_cells(tmp_path: Path) -> None:
    config = C3CampaignConfig()
    task = C3TaskConfig()
    small = C3CampaignConfig(
        arms=("analytic",),
        seeds=(7,),
        training_steps=2,
        train_samples=32,
        batch_size=8,
        probe_train_latents=32,
        probe_validation_latents=16,
        probe_test_latents=24,
        probe_steps=4,
        required_seed_passes=1,
        allow_underpowered=True,
    )
    dataset_contract = _dataset_contract(task, small)
    experiments = _experiments(
        config,
        task,
        tmp_path,
        "implementation",
        dataset_contract,
        ARMS,
    )
    assert len(experiments) == 15
    assert len({item.id for item in experiments}) == 15
    assert {item.parameters["arm"] for item in experiments} == set(ARMS)
