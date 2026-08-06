import copy

import pytest
import torch

from experiments.structure_net.tinyllm_block1_quotient_control import (
    CONDITIONS,
    CUTS,
    PRIMARY_CUTS,
    PRIMARY_REGIMES,
    REGIMES,
    Block1QuotientControlConfig,
    _experiments,
    _analysis_config,
    _protocol_material,
    aggregate_details,
    endpoint_pass,
    gradient_reverse,
    train_controlled,
)
from experiments.structure_net.tinyllm_nuisance_support_scaling import train_cell
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


def _task() -> CircleTaskConfig:
    return CircleTaskConfig(
        phase_bins=8,
        sensor_steps=4,
        value_bins=8,
        vocab_size=512,
        answer_token_start=400,
        train_samples=32,
        evaluation_samples=16,
        phase_grid_points=12,
        nuisance_phase_count=2,
        nuisance_replicates=4,
    )


def _config(**changes) -> Block1QuotientControlConfig:
    values = {
        "preset": "tiny",
        "seeds": (7,),
        "training_steps": 2,
        "train_samples": 32,
        "batch_size": 8,
        "probe_train_samples": 64,
        "probe_validation_samples": 32,
        "probe_test_samples": 32,
        "probe_steps": 20,
        "probe_width": 16,
        "probe_batch_size": 16,
        "local_replicates": 2,
        "fiber_replicates": 3,
        "device_ids": (-1,),
        "allow_underpowered": True,
    }
    values.update(changes)
    return Block1QuotientControlConfig(**values)


def test_gradient_reversal_changes_only_the_backward_sign():
    values = torch.tensor([1.0, -2.0], requires_grad=True)
    reversed_values = gradient_reverse(values)

    assert torch.equal(reversed_values, values)
    (3.0 * reversed_values).sum().backward()
    assert torch.equal(values.grad, torch.full_like(values, -3.0))


def test_protocol_material_is_exactly_reproducible():
    first = _protocol_material(_task(), _config(), 7)
    second = _protocol_material(_task(), _config(), 7)

    assert first[2:] == second[2:]
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[0].circle.input_ids, second[0].circle.input_ids)


def test_zero_weight_intervention_reproduces_ordinary_training(tmp_path):
    config = _config(base_loss_weight=0.0, invariance_loss_weight=0.0)
    task = _task()
    ordinary, ordinary_record = train_cell(
        task,
        _analysis_config(config),
        "N3",
        "standard_final",
        7,
        torch.device("cpu"),
        tmp_path / "ordinary",
    )
    controlled, controlled_record = train_controlled(
        task,
        config,
        7,
        torch.device("cpu"),
        tmp_path / "controlled",
    )

    assert controlled_record["final_state_sha256"] == ordinary_record[
        "final_state_sha256"
    ]
    assert all(
        torch.equal(ordinary.state_dict()[name], value)
        for name, value in controlled.state_dict().items()
    )


def test_campaign_has_two_conditions_and_one_checkpoint_cell_per_seed(tmp_path):
    config = Block1QuotientControlConfig(device_ids=(-1,))
    experiments = _experiments(config, CircleTaskConfig(), tmp_path)

    assert len(experiments) == 10
    assert {
        experiment.parameters["condition"] for experiment in experiments
    } == set(CONDITIONS)


def test_endpoint_includes_log_loss_gain():
    passing = {
        "cosine_pearson": 0.91,
        "balanced_accuracy": 0.54,
        "conditional_log_loss_gain_over_cosine_only": 0.019,
    }
    assert endpoint_pass(passing)
    failing = {**passing, "conditional_log_loss_gain_over_cosine_only": 0.021}
    assert not endpoint_pass(failing)


def _run(condition: str, seed: int) -> dict:
    passing = {
        "cosine_pearson": 0.95,
        "balanced_accuracy": 0.50,
        "conditional_log_loss_gain_over_cosine_only": 0.0,
    }
    analysis = {
        "cuts": {
            cut: {
                "probe": {
                    "evaluations": {
                        "interpolation": copy.deepcopy(passing),
                        "composition": copy.deepcopy(passing),
                        "extrapolation": copy.deepcopy(passing),
                    }
                }
            }
            for cut in CUTS
        }
    }
    task_metrics = {
        regime: {
            "post_mlp": {"exact_bin_accuracy": 0.75},
            "full": {"exact_bin_accuracy": 0.75},
        }
        for regime in REGIMES
    }
    local = {
        regime: {
            cut: {"median_q_local": 1.0} for cut in CUTS
        }
        for regime in REGIMES
    }
    return {
        "condition": condition,
        "seed": seed,
        "analysis": analysis,
        "task_metrics": task_metrics,
        "local_derivative": local,
    }


def test_primary_gate_requires_the_same_four_seeds_to_pass_every_cell():
    seeds = (7, 17, 29, 41, 53)
    config = Block1QuotientControlConfig(seeds=seeds, device_ids=(-1,))
    runs = [_run(condition, seed) for condition in CONDITIONS for seed in seeds]
    controlled = {
        run["seed"]: run for run in runs if run["condition"] == CONDITIONS[1]
    }
    # Each cell has four passing seeds, but each cell fails a different seed.
    for failed_seed, (cut, regime) in zip(
        seeds[:4],
        [
            (PRIMARY_CUTS[0], PRIMARY_REGIMES[0]),
            (PRIMARY_CUTS[0], PRIMARY_REGIMES[1]),
            (PRIMARY_CUTS[1], PRIMARY_REGIMES[0]),
            (PRIMARY_CUTS[1], PRIMARY_REGIMES[1]),
        ],
    ):
        source_regime = regime
        controlled[failed_seed]["analysis"]["cuts"][cut]["probe"][
            "evaluations"
        ][source_regime]["cosine_pearson"] = 0.5

    aggregate = aggregate_details(runs, config)

    assert all(
        aggregate["cells"][CONDITIONS[1]]["cuts"][cut][regime]["pass_count"]
        == 4
        for cut in PRIMARY_CUTS
        for regime in PRIMARY_REGIMES
    )
    assert aggregate["preregistered_gate"]["joint_pass_count"] == 1
    assert not aggregate["preregistered_gate"]["overall_success"]


def test_preregistered_configuration_requires_five_seeds():
    with pytest.raises(ValueError, match="five seeds"):
        Block1QuotientControlConfig(seeds=(7,))
