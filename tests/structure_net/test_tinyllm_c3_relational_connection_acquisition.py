from __future__ import annotations

import copy

import pytest
import torch

from experiments.structure_net import (
    tinyllm_c3_relational_connection_acquisition as study,
)


def _small_config() -> study.AcquisitionConfig:
    return study.AcquisitionConfig(
        seeds=(1409,),
        training_steps=2,
        train_samples=128,
        evaluation_samples=128,
        batch_size=8,
        required_seed_passes=1,
        device_ids=(-1,),
        gpu_slots_per_device=1,
        max_gpu_slots_per_device=1,
        max_parallel_experiments=1,
        allow_underpowered=True,
    )


def test_primary_configuration_and_sources_are_frozen() -> None:
    config = study.AcquisitionConfig()
    assert config.seeds == study.SEEDS
    assert config.training_steps == 2_400
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 0.0
    assert config.split_step == 1_200
    with pytest.raises(ValueError, match="protocol is frozen"):
        study.AcquisitionConfig(training_steps=2_402)
    sources, predecessor = study._validate_sources()
    assert sources["preregistration"] == study.PREREGISTRATION_SHA256
    assert predecessor["aggregates"]["classification"] == study.fc.CLASSIFICATION_PASS
    assert predecessor["aggregates"]["unrestricted_tinyllm_training_licensed"] is False


def test_protocol_material_is_deterministic_and_controls_change_information() -> None:
    config = _small_config()
    first = study.protocol_material(config, 1409)
    second = study.protocol_material(config, 1409)
    assert first[-1] == second[-1]
    evaluations, permutations, hashes = study.build_evaluation_material(
        config, 1409
    )
    again_evaluations, again_permutations, again_hashes = (
        study.build_evaluation_material(config, 1409)
    )
    assert hashes == again_hashes
    assert all(
        torch.equal(permutations[regime], again_permutations[regime])
        for regime in study.REGIMES
    )
    assert all(
        study.dataset_digest(evaluations[regime])
        == study.dataset_digest(again_evaluations[regime])
        for regime in study.REGIMES
    )
    contract = study.protocol_contract(
        first[0], first[2], first[3], evaluations, permutations
    )
    assert contract["pass"] is True
    assert contract["connection_permutation_fixed_points"] == 0
    assert contract["target_permutation_fixed_points"] == 0
    assert contract["training_changed_connection_fraction"] >= 0.90
    assert contract["training_mean_absolute_target_change"] >= 0.25


def test_arms_share_initialization_and_apply_only_registered_changes() -> None:
    config = _small_config()
    dataset, _batches, connection_permutation, target_permutation, _hashes = (
        study.protocol_material(config, 1409)
    )
    initial = [
        study.fc._state_digest(study._initial_module(1409, torch.device("cpu")))
        for _arm in study.ARMS
    ]
    assert len(set(initial)) == 1
    true_connection, true_target = study._arm_training_material(
        "learned_true", dataset, connection_permutation, target_permutation
    )
    no_connection, no_target = study._arm_training_material(
        "learned_no_connection",
        dataset,
        connection_permutation,
        target_permutation,
    )
    shuffled_connection, shuffled_connection_target = (
        study._arm_training_material(
            "learned_connection_shuffled",
            dataset,
            connection_permutation,
            target_permutation,
        )
    )
    target_connection, shuffled_target = study._arm_training_material(
        "learned_target_shuffled",
        dataset,
        connection_permutation,
        target_permutation,
    )
    assert torch.equal(true_connection, dataset.connection)
    assert torch.equal(true_target, dataset.target)
    assert torch.count_nonzero(no_connection) == 0
    assert torch.equal(no_target, dataset.target)
    assert torch.equal(
        shuffled_connection, dataset.connection[connection_permutation]
    )
    assert torch.equal(shuffled_connection_target, dataset.target)
    assert torch.equal(target_connection, dataset.connection)
    assert torch.equal(shuffled_target, dataset.target[target_permutation])


def test_analytic_control_and_random_state_obey_exact_action_class() -> None:
    config = _small_config()
    analytic = study.evaluate_analytic_seed(config, 1409)
    assert analytic["joint_pass"] is True
    module = study._initial_module(1409, torch.device("cpu"))
    datasets, _permutations, _hashes = study.build_evaluation_material(
        config, 1409
    )
    action = torch.randint(
        0,
        study.rel.CHANNELS,
        (config.evaluation_samples, study.rel.TIME_STEPS),
        generator=torch.Generator().manual_seed(123),
    )
    error = study.action_error(
        module,
        datasets["composition"],
        datasets["composition"].connection,
        action,
    )
    winding = study.winding_diagnostic(module, points=257)
    assert error <= study.ACTION_ERROR_MAXIMUM
    assert winding["winding_number"] in (-2, 1, 4)
    assert winding["minimum_raw_magnitude"] >= 0.0


def _detail(seed: int, *, valid: bool, passes: dict[str, bool]) -> dict:
    return {
        "seed": seed,
        "gates": {
            "validity": valid,
            **{f"{arm}_joint": passes.get(arm, False) for arm in study.ARMS},
        },
    }


def test_aggregate_classifications_are_locked() -> None:
    config = study.AcquisitionConfig()
    analytic = [{"joint_pass": True} for _seed in config.seeds]
    positive = [
        _detail(
            seed,
            valid=True,
            passes={"learned_true": index < 4},
        )
        for index, seed in enumerate(config.seeds)
    ]
    supported = study.aggregate_details(positive, analytic, config)
    assert supported["primary_hypothesis_pass"] is True
    assert (
        supported["classification"]
        == "connection_invariant_relation_acquired_by_gradient_training"
    )
    assert supported["unrestricted_tinyllm_training_licensed"] is False

    unreliable = copy.deepcopy(positive)
    unreliable[3]["gates"]["learned_true_joint"] = False
    result = study.aggregate_details(unreliable, analytic, config)
    assert result["primary_hypothesis_pass"] is False
    assert (
        result["classification"]
        == "exact_function_class_but_population_acquisition_unreliable"
    )

    nonspecific = copy.deepcopy(positive)
    nonspecific[0]["gates"]["learned_no_connection_joint"] = True
    nonspecific[1]["gates"]["learned_no_connection_joint"] = True
    result = study.aggregate_details(nonspecific, analytic, config)
    assert result["classification"] == "connection_acquisition_specificity_failed"

    stopped = study.aggregate_details([], [{"joint_pass": False}] * 5, config)
    assert (
        stopped["classification"]
        == "analytic_connection_ceiling_failed_on_fresh_streams"
    )
