from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import tinyllm_calibrated_frontend_causal as calibrated
from experiments.structure_net import tinyllm_cross_seed_symmetry_feature_swap as swap
from experiments.structure_net import tinyllm_invariant_frontend_causal as invariant
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from structure_net.components.models import TinyLLMConfig, TinyLLMModel


def _packet(count: int, generator: torch.Generator) -> torch.Tensor:
    angle = torch.randn(count, generator=generator)
    packet = torch.zeros((count, calibrated.CALIBRATION_WIDTH))
    packet[:, 0] = torch.cos(angle)
    packet[:, 1] = torch.sin(angle)
    packet[:, 2] = torch.where(torch.arange(count) % 2 == 0, 0.35, -0.35)
    packet[:, 3] = 0.5 + torch.rand(count, generator=generator)
    packet[:, 4:8] = torch.randn((count, 4), generator=generator) * 0.2
    return packet


def _small_system() -> tuple[
    calibrated.CalibratedTinyLLM, CircleTaskConfig, calibrated.CalibratedFrontendConfig
]:
    task = CircleTaskConfig(
        phase_bins=8,
        sensor_steps=3,
        value_bins=4,
        vocab_size=128,
        answer_token_start=64,
        quantization_limit=2.0,
    )
    config = calibrated.CalibratedFrontendConfig(
        preset="d6",
        seeds=(1,),
        vector_channels=4,
        allow_underpowered=True,
    )
    model = TinyLLMModel(
        TinyLLMConfig(
            block_size=task.sequence_length + 1,
            vocab_size=task.vocab_size,
            n_layer=1,
            n_head=2,
            n_embd=16,
            initialization_seed=5,
        )
    )
    return calibrated.CalibratedTinyLLM(
        model, swap.CONDITION, task, config
    ).eval(), task, config


def test_primary_configuration_is_fixed_and_shakedown_is_labeled() -> None:
    config = swap.SymmetryFeatureSwapConfig(device="cpu")
    assert config.seeds == (7, 17, 29, 41, 53)
    assert config.required_pair_passes == 16
    with pytest.raises(ValueError, match="five source seeds"):
        swap.SymmetryFeatureSwapConfig(seeds=(7, 17), device="cpu")
    pilot = swap.SymmetryFeatureSwapConfig(
        seeds=(7, 17),
        sample_count=16,
        required_pair_passes=2,
        required_target_passes=2,
        required_sources_per_target=1,
        device="cpu",
        allow_underpowered=True,
    )
    assert swap._evidence_role(pilot) == "systems_lifecycle_only_not_quality_evidence"


def test_canonical_feature_is_declared_dot_cross_speed_interface() -> None:
    system, task, _ = _small_system()
    generator = torch.Generator().manual_seed(11)
    sensor = torch.randn((9, task.sensor_steps, 3), generator=generator)
    packet = _packet(9, generator)
    feature = swap.canonical_feature(system, sensor, packet)
    vector = system.encoder.equivariant_vector(sensor, packet)  # type: ignore[union-attr]
    expected = torch.stack(
        (
            (vector * packet[:, :2]).sum(-1),
            packet[:, 0] * vector[:, 1] - packet[:, 1] * vector[:, 0],
            packet[:, 2],
        ),
        -1,
    )
    assert torch.equal(feature, expected)
    assert torch.allclose(feature[:, :2].square().sum(-1), torch.ones(9), atol=1e-6)


def test_positive_similarity_action_preserves_canonical_feature() -> None:
    system, task, _ = _small_system()
    generator = torch.Generator().manual_seed(17)
    sensor = torch.randn((13, task.sensor_steps, 3), generator=generator)
    packet = _packet(13, generator)
    baseline = swap.canonical_feature(system, sensor, packet)
    transformed_sensor, transformed_packet = swap.apply_acquisition_action(
        sensor,
        packet,
        system.encoder.normalized_history,  # type: ignore[union-attr]
        swap.GROUP_ACTIONS[0],
    )
    transformed = swap.canonical_feature(system, transformed_sensor, transformed_packet)
    assert torch.allclose(transformed, baseline, atol=2e-6, rtol=2e-6)


def test_custom_scalar_continuation_exactly_replays_structured_system() -> None:
    system, task, _ = _small_system()
    generator = torch.Generator().manual_seed(29)
    count = 8
    input_ids = torch.randint(
        0, task.vocab_size, (count, task.sequence_length), generator=generator
    )
    sensor = torch.randn((count, task.sensor_steps, 3), generator=generator)
    packet = _packet(count, generator)
    feature = swap.canonical_feature(system, sensor, packet)
    scalar = system.encoder.scalar_map(feature)  # type: ignore[union-attr]
    replay = swap._continue_from_scalar(
        system, input_ids, scalar, task, batch_size=4, device=torch.device("cpu")
    )
    residual = system.forward_cuts(input_ids, sensor, packet)["full"]
    answer_ids = torch.tensor(task.answer_token_ids)
    ordinary = torch.softmax(invariant._task_logits(system.model, residual, answer_ids), -1)
    assert float((replay - ordinary).abs().max()) <= 1e-7


def test_posterior_metrics_and_primary_gate_use_all_declared_conditions() -> None:
    posterior = torch.tensor(
        [[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]], dtype=torch.float64
    )
    target = posterior.clone()
    centers = torch.tensor([-1.0, 1.0], dtype=posterior.dtype)
    true = posterior @ centers
    direct = swap._posterior_metrics(posterior, target, true, posterior)
    assert direct["exact_bin_accuracy"] == 1.0
    assert direct["mean_fisher_rao_from_direct"] == pytest.approx(0.0, abs=1e-7)
    config = swap.SymmetryFeatureSwapConfig(device="cpu")
    good = dict(direct)
    shuffled = dict(direct)
    half = dict(direct)
    shuffled["target_cross_entropy"] += 0.2
    half["target_cross_entropy"] += 0.3
    gate = swap.cell_gate(good, direct, shuffled, half, 0.0, config)
    assert gate["pass"]
    shuffled["target_cross_entropy"] = good["target_cross_entropy"]
    gate = swap.cell_gate(good, direct, shuffled, half, 0.0, config)
    assert not gate["pass"]
    assert not gate["checks"]["shuffled_specificity"]


def test_feature_geometry_detects_a_constant_gauge_rotation() -> None:
    angle = torch.linspace(-2.5, 2.5, 128)
    offset = 0.43
    source = torch.stack((torch.cos(angle + offset), torch.sin(angle + offset), torch.ones_like(angle)), -1)
    target = torch.stack((torch.cos(angle), torch.sin(angle), torch.ones_like(angle)), -1)
    source_scalar = torch.cos(angle + offset)[:, None]
    target_scalar = torch.cos(angle)[:, None]
    geometry = swap.feature_geometry(source, target, source_scalar, target_scalar)
    assert geometry["mean_absolute_angular_displacement_radians"] == pytest.approx(offset, abs=1e-6)
    assert geometry["circular_concentration"] == pytest.approx(1.0, abs=1e-6)
    assert geometry["circular_mean_offset_radians"] == pytest.approx(offset, abs=1e-6)


def test_campaign_gate_counts_targets_not_directed_pairs_as_replicates() -> None:
    config = swap.SymmetryFeatureSwapConfig(device="cpu")
    results = []
    for source in config.seeds:
        for target in config.seeds:
            if source == target:
                continue
            results.append(
                {
                    "source_seed": source,
                    "target_seed": target,
                    "gates": {"pair_pass": True, "scalar_pair_pass": True},
                    "cells": {
                        "a": {
                            "feature_swap_gate": {
                                "checks": {"direct_replay": True}
                            }
                        }
                    },
                }
            )
    contracts = {seed: {"pass": True} for seed in config.seeds}
    aggregate = swap.aggregate_pair_results(results, contracts, config)
    assert aggregate["confirmed"]
    assert aggregate["pair_pass_count"] == 20
    for result in results:
        if result["target_seed"] == 53:
            result["gates"]["pair_pass"] = False
    aggregate = swap.aggregate_pair_results(results, contracts, config)
    assert aggregate["confirmed"]
    assert aggregate["target_pass_count"] == 4
    for result in results:
        if result["target_seed"] == 41:
            result["gates"]["pair_pass"] = False
    aggregate = swap.aggregate_pair_results(results, contracts, config)
    assert not aggregate["confirmed"]
    assert aggregate["target_pass_count"] == 3
