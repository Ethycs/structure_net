from pathlib import Path

import numpy as np
import pytest

from experiments.structure_net.tinyllm_noisy_bias_pilot_acquisition import (
    CONDITIONS,
    COUNTS,
    DRAW_COUNT,
    NoisyBiasPilotConfig,
    PREREGISTRATION_SHA256,
    SEEDS,
    SOURCE_ACQUISITION_ARRAY_SHA256,
    SOURCE_ACQUISITION_CAMPAIGN_SHA256,
    SOURCE_REPAIR_CAMPAIGN_SHA256,
    _load_acquisition_source,
    _load_repair_source,
    _sha256,
    _source_digests,
    _worker_cells,
    aggregate_results,
    build_pilot_arrays,
    classify_campaign,
    pilot_contract,
    prepare_campaign,
)


def _underpowered() -> NoisyBiasPilotConfig:
    return NoisyBiasPilotConfig(
        conditions=("analytic_calibrated",),
        seeds=(7,),
        counts=(1, 256),
        draw_count=2,
        required_seed_passes=1,
        required_draw_passes=2,
        maximum_control_seed_passes=0,
        sample_limit=32,
        devices=("cpu",),
        allow_underpowered=True,
    )


def test_primary_configuration_is_locked() -> None:
    config = NoisyBiasPilotConfig()
    assert config.conditions == CONDITIONS
    assert config.seeds == SEEDS
    assert config.counts == COUNTS
    assert config.draw_count == DRAW_COUNT
    assert config.devices == ("cuda:0", "cuda:1", "cuda:2")
    with pytest.raises(ValueError, match="primary noisy-pilot"):
        NoisyBiasPilotConfig(draw_count=8, required_draw_passes=8)


def test_source_digests_pin_all_runners_and_preregistration() -> None:
    digests = _source_digests()
    assert digests["preregistration"] == PREREGISTRATION_SHA256
    assert len(digests["runner"]) == 64
    assert len(digests["source_repair_runner"]) == 64
    assert len(digests["source_acquisition_runner"]) == 64


def test_source_campaigns_and_arrays_are_exact() -> None:
    repair_campaign, repair_path, details, diagnostics = _load_repair_source(
        _underpowered()
    )
    assert _sha256(repair_path) == SOURCE_REPAIR_CAMPAIGN_SHA256
    assert repair_campaign["aggregates"]["primary_hypothesis_pass"] is True
    assert len(details) == len(diagnostics) == 10
    acquisition_campaign, array_path, errors = _load_acquisition_source(
        _underpowered()
    )
    assert _sha256(array_path) == SOURCE_ACQUISITION_ARRAY_SHA256
    assert _sha256(Path(acquisition_campaign["artifacts"]["campaign"])) == (
        SOURCE_ACQUISITION_CAMPAIGN_SHA256
    )
    assert errors.shape == (16, 256, 512)


def test_pilot_arrays_are_nested_target_free_and_valid() -> None:
    config = NoisyBiasPilotConfig()
    _, _, errors = _load_acquisition_source(config)
    arrays = build_pilot_arrays(errors, config)
    contract = pilot_contract(arrays, config)
    assert arrays["standard_normal_streams"].shape == (16, 256, 2)
    assert arrays["source_audit_standard_normal_streams"].shape == (16, 256, 2)
    assert arrays["pilot_estimates"].shape == (16, 5, 2)
    assert contract["pass"] is True
    assert contract["no_new_random_draws"] is True
    assert abs(contract["cross_channel_correlation"]) < 0.05
    assert contract["prefix_mean_rmse"]["256"] < contract["prefix_mean_rmse"]["1"]
    expected = np.asarray([0.03125, 0.0]) + (
        config.pilot_noise_sigma
        * arrays["standard_normal_streams"][0, :4].mean(axis=0)
    )
    assert np.array_equal(arrays["pilot_estimates"][0, 1], expected)


def test_prepare_round_trip_reuses_frozen_arrays(tmp_path: Path) -> None:
    config = _underpowered()
    first = prepare_campaign(config, tmp_path)
    second = prepare_campaign(config, tmp_path)
    assert first == second
    assert first["pilot_contract"]["pass"] is True
    assert Path(first["pilot_arrays"]).is_file()


def test_worker_mapping_is_deterministic_and_complete() -> None:
    config = NoisyBiasPilotConfig()
    workers = [_worker_cells(config, index) for index in range(3)]
    assert sum(len(cells) for cells in workers) == 10
    flattened = {cell for cells in workers for cell in cells}
    assert flattened == {
        (condition, seed) for condition in CONDITIONS for seed in SEEDS
    }
    assert not set(workers[0]).intersection(workers[1])


def test_classification_requires_both_arms_and_controls() -> None:
    common = {
        "integrity_valid": True,
        "pilot_contract_pass": True,
        "source_controls_pass": True,
        "analytic_m256_draw_passes": 15,
        "learned_m256_draw_passes": 15,
        "required_draw_passes": 15,
    }
    assert classify_campaign(**common) == (
        "finite_noisy_pilot_repair_reliable",
        True,
    )
    assert classify_campaign(
        **{**common, "learned_m256_draw_passes": 14}
    ) == ("finite_noisy_pilot_arm_asymmetry", False)
    assert classify_campaign(
        **{
            **common,
            "analytic_m256_draw_passes": 14,
            "learned_m256_draw_passes": 14,
        }
    ) == ("finite_noisy_pilot_insufficient", False)
    assert classify_campaign(**{**common, "source_controls_pass": False}) == (
        "invalid",
        False,
    )


def test_aggregate_finds_smallest_reliable_count() -> None:
    config = NoisyBiasPilotConfig()
    results = []
    for condition in CONDITIONS:
        for seed_index, seed in enumerate(SEEDS):
            gates = {}
            for draw_index in range(DRAW_COUNT):
                gates[f"draw_{draw_index:02d}"] = {
                    str(count): (
                        seed_index < (3 if count < 16 else 4)
                    )
                    for count in COUNTS
                }
            results.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "draw_count_seed_gates": gates,
                    "source_exact_pilot_seed_gate": True,
                    "source_full_plus_seed_gate": (
                        seed_index < (1 if condition == "analytic_calibrated" else 3)
                    ),
                    "wrong_sign_draw0_m256_seed_gate": (
                        seed_index == 0 and condition == "analytic_calibrated"
                    ),
                    "gates": {"validity": True},
                }
            )
    population = aggregate_results(results, config)
    assert population["smallest_reliable_count"] == 16
    assert population["counts"]["4"]["complete_draw_passes"] == 0
    assert population["counts"]["16"]["complete_draw_passes"] == 16
    assert population["source_controls_pass"] is True
    assert population["integrity_valid"] is True


def test_preregistration_is_frozen() -> None:
    path = Path(
        "docs/07 - Status Reports/"
        "2026-08-10_tinyllm-noisy-bias-pilot-acquisition-preregistration.md"
    )
    assert _sha256(path) == PREREGISTRATION_SHA256
