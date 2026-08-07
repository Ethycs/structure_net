from pathlib import Path

import numpy as np
import torch

from experiments.structure_net.tinyllm_calibrated_frontend_causal import (
    generate_calibrated_dataset,
)
from experiments.structure_net.tinyllm_internal_quotient_probe import FiberDataset
from experiments.structure_net.tinyllm_io_correspondence import (
    IOCorrespondenceConfig,
    _protocol_material,
    condition_spec,
    generate_relation_fiber_dataset,
    io_relation_contract,
    mapper_summary,
    paired_map_distortion,
    verify_calibrated_source,
    with_relation_target,
)
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SOURCE = Path(
    "data/experiments/tinyllm_calibrated_frontend_causal/"
    "20260806_d8_preregistered"
)


def _small_config(**updates):
    values = {
        "seeds": (7,),
        "training_steps": 2,
        "train_samples": 32,
        "batch_size": 8,
        "probe_train_samples": 64,
        "probe_validation_samples": 32,
        "probe_test_samples": 32,
        "probe_steps": 20,
        "reuse_calibrated_controls": False,
        "allow_underpowered": True,
    }
    values.update(updates)
    return IOCorrespondenceConfig(**values)


def _fiber(targets: np.ndarray) -> FiberDataset:
    count = len(targets)
    return FiberDataset(
        input_ids=torch.zeros((count, 1), dtype=torch.int64),
        cosine=torch.from_numpy(targets.astype(np.float32)),
        branch=torch.from_numpy(np.tile([0.0, 1.0], count // 2).astype(np.float32)),
        phase=torch.zeros(count),
        fiber_id=torch.arange(count // 2).repeat_interleave(2),
    )


def test_three_relation_contract_has_one_deliberate_failure_and_two_repairs():
    contract = io_relation_contract()
    relations = contract["relations"]
    assert contract["passed"] is True
    assert relations["uncalibrated_absolute"]["expected_to_descend"] is False
    assert relations["uncalibrated_absolute"]["counterexample_count"] > 0
    assert relations["calibrated_absolute"]["violations"] == 0
    assert relations["uncalibrated_relative"]["violations"] == 0


def test_condition_table_separates_observation_target_and_frontend():
    assert condition_spec("uncalibrated_absolute_equivariant") == {
        "observation": "uncalibrated",
        "target": "absolute",
        "frontend": "equivariant",
    }
    assert condition_spec("calibrated_absolute_analytic")["frontend"] == "analytic"
    assert condition_spec("uncalibrated_relative_raw")["target"] == "relative"


def test_absolute_relation_generator_matches_existing_observations_and_calibration():
    task = CircleTaskConfig(train_samples=32)
    generated = generate_relation_fiber_dataset(
        task, target_kind="absolute", sample_count=32, seed=1_084
    )
    existing = generate_calibrated_dataset(task, sample_count=32, seed=1_084)
    assert torch.equal(
        generated.paired.circle.input_ids, existing.paired.circle.input_ids
    )
    assert torch.equal(generated.calibration, existing.calibration)
    assert torch.equal(generated.paired.fiber.cosine, existing.paired.fiber.cosine)
    assert torch.equal(generated.paired.fiber.branch, existing.paired.fiber.branch)


def test_relative_relation_has_exact_output_matched_opposite_branches():
    task = CircleTaskConfig(train_samples=64)
    dataset = generate_relation_fiber_dataset(
        task, target_kind="relative", sample_count=64, seed=2_019
    )
    fibers = dataset.paired.fiber
    for fiber_id in torch.unique(fibers.fiber_id):
        indices = torch.nonzero(fibers.fiber_id == fiber_id, as_tuple=False).flatten()
        assert len(indices) == 2
        assert torch.allclose(fibers.cosine[indices[0]], fibers.cosine[indices[1]])
        assert set(fibers.branch[indices].tolist()) == {0.0, 1.0}


def test_target_side_repair_changes_only_targets_on_shared_training_cohort():
    task = CircleTaskConfig(train_samples=32)
    base = generate_calibrated_dataset(
        task, sample_count=32, seed=1_008, shuffle=False
    )
    relative = with_relation_target(base, task, "relative")
    assert torch.equal(base.paired.circle.input_ids, relative.paired.circle.input_ids)
    assert torch.equal(base.calibration, relative.calibration)
    assert not torch.equal(
        base.paired.circle.target_posteriors,
        relative.paired.circle.target_posteriors,
    )


def test_matched_protocol_keeps_base_and_minibatches_fixed_across_targets():
    task = CircleTaskConfig(train_samples=32)
    config = _small_config()
    _, absolute_batches, absolute_base, absolute_target, absolute_schedule = (
        _protocol_material(task, config, 7, "absolute")
    )
    _, relative_batches, relative_base, relative_target, relative_schedule = (
        _protocol_material(task, config, 7, "relative")
    )
    assert absolute_base == relative_base
    assert absolute_schedule == relative_schedule
    assert torch.equal(absolute_batches, relative_batches)
    assert absolute_target != relative_target


def test_frozen_calibrated_source_passes_exact_reuse_audit():
    task = CircleTaskConfig(train_samples=4_096)
    config = IOCorrespondenceConfig()
    detail = verify_calibrated_source(
        task, config, "calibrated_absolute_equivariant", 7, SOURCE
    )
    assert detail["training"]["initial_model_state_sha256"]
    assert detail["training"]["training_data_sha256"]
    assert detail["training"]["minibatch_schedule_sha256"]


def test_mapper_distinguishes_one_sheet_from_two_branch_sheets():
    pair_count = 128
    target_pairs = np.repeat(np.linspace(-0.94, 0.94, pair_count), 2)
    fiber = _fiber(target_pairs)
    generator = np.random.default_rng(9)
    one_sheet = np.column_stack(
        (target_pairs, generator.normal(0.0, 0.01, len(target_pairs)))
    )
    branch = fiber.branch.numpy()
    two_sheets = np.column_stack((target_pairs, 8.0 * branch))
    config = _small_config(mapper_neighbors=8)
    one = mapper_summary(one_sheet, fiber, config)
    two = mapper_summary(two_sheets, fiber, config)
    assert one["interior_single_sheet_fraction"] > two["interior_single_sheet_fraction"]
    assert two["interior_two_branch_sheet_fraction"] > 0.5


def test_mapper_handles_shakedown_sized_sparse_covers():
    pair_count = 16
    targets = np.repeat(np.linspace(-0.94, 0.94, pair_count), 2)
    fiber = _fiber(targets)
    projected = np.column_stack((targets, np.zeros_like(targets)))
    result = mapper_summary(projected, fiber, _small_config(mapper_neighbors=12))
    assert result["cover_intervals"] == 9
    assert 0.0 <= result["sample_coverage"] <= 1.0


def test_paired_distortion_rewards_ordered_fiber_centroids():
    pair_count = 96
    targets = np.repeat(np.linspace(-0.95, 0.95, pair_count), 2)
    fiber = _fiber(targets)
    generator = np.random.default_rng(11)
    ordered = np.column_stack(
        (targets, generator.normal(0.0, 0.01, len(targets)))
    )
    shuffled_centers = generator.permutation(np.linspace(-0.95, 0.95, pair_count))
    shuffled = np.column_stack(
        (
            np.repeat(shuffled_centers, 2),
            generator.normal(0.0, 0.01, len(targets)),
        )
    )
    ordered_result = paired_map_distortion(ordered, fiber)
    shuffled_result = paired_map_distortion(shuffled, fiber)
    assert ordered_result["fiber_averaged_whitened_order_spearman"] > 0.95
    assert ordered_result["fiber_averaged_whitened_distortion"] < shuffled_result[
        "fiber_averaged_whitened_distortion"
    ]
