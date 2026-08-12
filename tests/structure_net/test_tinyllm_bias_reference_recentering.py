import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net.tinyllm_bias_reference_recentering import (
    BiasReferenceRecenteringConfig,
    CONDITIONS,
    PREREGISTRATION_SHA256,
    SEEDS,
    SOURCE_CAMPAIGN_SHA256,
    SOURCE_COMPONENT_CONTRACT_SHA256,
    SOURCE_RUNNER_SHA256,
    _base_config,
    _load_source_campaign,
    _sha256,
    _source_digests,
    aggregate_results,
    classify_campaign,
    construct_interventions,
    intervention_contract,
)
import experiments.structure_net.tinyllm_bias_component_causal_decomposition as bias


def _underpowered() -> BiasReferenceRecenteringConfig:
    return BiasReferenceRecenteringConfig(
        conditions=("analytic_calibrated",),
        seeds=(7,),
        sample_limit=32,
        required_seed_passes=1,
        maximum_control_seed_passes=0,
        device="cpu",
        allow_underpowered=True,
    )


def test_primary_configuration_is_locked() -> None:
    config = BiasReferenceRecenteringConfig()
    assert config.conditions == CONDITIONS
    assert config.seeds == SEEDS
    assert config.required_seed_passes == 4
    assert config.maximum_control_seed_passes == 1
    with pytest.raises(ValueError, match="primary bias-reference"):
        BiasReferenceRecenteringConfig(batch_size=128)


def test_source_digests_pin_preregistration_and_runner() -> None:
    digests = _source_digests()
    assert digests["preregistration"] == PREREGISTRATION_SHA256
    assert digests["source_component_runner"] == SOURCE_RUNNER_SHA256


def test_source_campaign_is_exact_and_confirmed() -> None:
    campaign, path, details, diagnostics = _load_source_campaign(_underpowered())
    assert _sha256(path) == SOURCE_CAMPAIGN_SHA256
    assert campaign["component_contract_sha256"] == SOURCE_COMPONENT_CONTRACT_SHA256
    assert campaign["aggregates"]["classification"] == (
        "deterministic_mean_sufficient"
    )
    assert len(details) == 10
    assert len(diagnostics) == 10


def test_construct_interventions_changes_only_declared_offset() -> None:
    sensor = torch.zeros(3, 4, 3)
    calibration = torch.zeros(3, 8)
    calibration[:, 0] = 1.0
    calibration[:, 2] = 1.0
    calibration[:, 3] = 1.0
    centered = torch.full((3, 4, 2), 0.01)
    mean = torch.zeros_like(centered)
    mean[..., 0] = 0.03125
    components = {
        "centered": centered,
        "mean_plus": mean,
        "full_plus": centered + mean,
    }
    values = construct_interventions(
        sensor, calibration, components, SimpleNamespace(sensor_steps=4)
    )
    repaired_sensor, repaired_calibration = values["recenter_correct"]
    wrong_sensor, wrong_calibration = values["recenter_wrong_sign"]
    assert torch.equal(repaired_sensor, wrong_sensor)
    assert torch.allclose(repaired_calibration[:, 4], torch.full((3,), 0.03125))
    assert torch.allclose(wrong_calibration[:, 4], torch.full((3,), -0.03125))
    assert torch.equal(repaired_calibration[:, :4], calibration[:, :4])
    assert torch.equal(repaired_calibration[:, 6:], calibration[:, 6:])


def test_intervention_contract_passes_on_frozen_arrays() -> None:
    config = _underpowered()
    base_config = _base_config(config)
    arrays = bias._load_base_noise(base_config)
    components = bias.construct_components(arrays, base_config)
    runtime = bias.dose._load_runtime_sources(bias._dose_config(base_config))
    contract = intervention_contract(runtime, components, config)
    assert contract["pass"] is True
    assert contract["no_new_random_draws"] is True
    for record in contract["regimes"].values():
        assert record["repaired_vs_centered_corrected_maximum_absolute_error"] < 2e-7
        assert record[
            "wrong_sign_vs_centered_plus_two_mean_maximum_absolute_error"
        ] < 2e-7
        assert record["target_changing_analytic_feature_rms"] >= 0.50


def test_classification_requires_repair_and_both_controls() -> None:
    common = {
        "integrity_valid": True,
        "contract_pass": True,
        "source_fails": True,
        "repaired_passes": True,
        "wrong_sign_specific": True,
        "target_specific": True,
    }
    assert classify_campaign(**common) == (
        "observed_bias_reference_repair_specific",
        True,
    )
    assert classify_campaign(**{**common, "wrong_sign_specific": False}) == (
        "algebraic_repair_without_specificity",
        False,
    )
    assert classify_campaign(**{**common, "repaired_passes": False}) == (
        "observed_bias_reference_insufficient",
        False,
    )
    assert classify_campaign(**{**common, "integrity_valid": False}) == (
        "invalid",
        False,
    )


def test_aggregate_uses_registered_population_and_control_thresholds() -> None:
    config = BiasReferenceRecenteringConfig()
    results = []
    for condition in CONDITIONS:
        for index, seed in enumerate(SEEDS):
            results.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "variant_seed_gates": {
                        "source_full_plus": index == 0,
                        "recenter_correct": index < 4,
                        "recenter_wrong_sign": index == 0,
                        "recenter_target_changing": False,
                    },
                    "gates": {"validity": True},
                }
            )
    population = aggregate_results(results, config)
    assert population["source_full_plus_fails_both_arms"] is True
    assert population["recenter_correct_passes_both_arms"] is True
    assert population["wrong_sign_specific_both_arms"] is True
    assert population["target_changing_specific_both_arms"] is True
    assert population["integrity_valid"] is True


def test_preregistration_exists_at_declared_path() -> None:
    path = Path(
        "docs/07 - Status Reports/"
        "2026-08-10_tinyllm-bias-reference-recentering-v2-preregistration.md"
    )
    assert path.is_file()
    assert _sha256(path) == PREREGISTRATION_SHA256


def test_v1_preflight_failure_is_preserved() -> None:
    path = Path(
        "data/experiments/tinyllm_bias_reference_recentering/"
        "20260810_shakedown_analytic_cuda/campaign_results.json"
    )
    campaign = json.loads(path.read_text())
    assert campaign["schema_version"] == (
        "nal.tinyllm-bias-reference-recentering.v1"
    )
    assert campaign["aggregates"]["classification"] == "invalid"
    assert campaign["aggregates"]["integrity_valid"] is False


def test_completed_v2_campaign_pins_primary_result() -> None:
    path = Path(
        "data/experiments/tinyllm_bias_reference_recentering/"
        "20260810_d10_preregistered_v2/campaign_results.json"
    )
    assert _sha256(path) == (
        "1996ac4c2534b62a25a2f52ceadfd21055a91bdadc81f38ccf01c6855da2b7d0"
    )
    campaign = json.loads(path.read_text())
    assert campaign["aggregates"] == {
        "classification": "observed_bias_reference_repair_specific",
        "integrity_valid": True,
        "intervention_contract_pass": True,
        "maximum_control_seed_passes": 1,
        "primary_evaluable": True,
        "primary_hypothesis_pass": True,
        "required_seed_passes": 4,
        "valid": True,
    }
    assert campaign["result_manifest_sha256"] == (
        "7dbbe3a49f4e3ebac36e891ec63d5336ff3be2e176e26f1a610cbfceecaabb4e"
    )
    assert campaign["intervention_contract_sha256"] == (
        "6bed75b6cd9a15be35f21e53463efa28bcc2f775f1490f31414e005398894004"
    )
