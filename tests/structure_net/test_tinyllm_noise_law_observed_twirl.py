from pathlib import Path
import json

import pytest
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_calibrated_frontend_causal_closure as closure
import experiments.structure_net.tinyllm_noise_law_observed_twirl as experiment
import experiments.structure_net.tinyllm_observed_deck_twirl as observed
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


PRIMARY = Path(
    "data/experiments/tinyllm_noise_law_observed_twirl/"
    "20260810_d10_preregistered/campaign_results.json"
)


def _datasets():
    return closure._datasets(CircleTaskConfig())


def _noise(datasets, config):
    return {
        regime: experiment.generate_noise_laws(
            sample_count=len(dataset.calibration),
            sensor_steps=8,
            sigma=config.noise_sigma,
            seed=experiment.NOISE_SEEDS[regime],
        )
        for regime, dataset in datasets.items()
    }


def test_primary_configuration_and_preregistration_are_locked() -> None:
    config = experiment.NoiseLawObservedTwirlConfig()
    assert config.conditions == experiment.CONDITIONS
    assert config.seeds == experiment.SEEDS
    assert config.laws == experiment.LAWS
    assert config.noise_sigma == 0.05
    assert config.sample_limit is None
    assert config.required_seed_passes == 4
    assert config.maximum_control_seed_passes == 1
    assert experiment._sha256(experiment.PREREGISTRATION_PATH) == (
        experiment.PREREGISTRATION_SHA256
    )
    with pytest.raises(ValueError, match="primary noise-law"):
        experiment.NoiseLawObservedTwirlConfig(noise_sigma=0.051)
    underpowered = experiment.NoiseLawObservedTwirlConfig(
        conditions=("analytic_calibrated",),
        seeds=(7,),
        laws=("isotropic",),
        required_seed_passes=1,
        sample_limit=64,
        allow_underpowered=True,
    )
    assert underpowered.allow_underpowered is True


def test_frozen_source_lineage_matches_predecessor() -> None:
    digests = experiment._source_digests()
    assert digests["observed_deck"] == experiment.SOURCE_OBSERVED_RUNNER_SHA256
    assert digests["causal_closure"] == experiment.SOURCE_CLOSURE_RUNNER_SHA256
    assert digests["calibrated_frontend"] == (
        experiment.SOURCE_CALIBRATED_RUNNER_SHA256
    )
    campaign, path, selected = experiment._load_predecessor(
        experiment.NoiseLawObservedTwirlConfig()
    )
    assert experiment._sha256(path) == experiment.SOURCE_OBSERVED_CAMPAIGN_SHA256
    assert campaign["result_manifest_sha256"] == (
        experiment.SOURCE_OBSERVED_RESULT_MANIFEST_SHA256
    )
    assert len(selected) == 10


def test_noise_laws_are_deterministic_and_share_base_draws() -> None:
    first = experiment.generate_noise_laws(
        sample_count=32, sensor_steps=8, sigma=0.05, seed=861001
    )
    second = experiment.generate_noise_laws(
        sample_count=32, sensor_steps=8, sigma=0.05, seed=861001
    )
    for law in experiment.LAWS:
        assert torch.equal(first[law], second[law])
    standard = first["isotropic"] / 0.05
    assert torch.allclose(
        first["lab_anisotropic"][..., 0],
        0.05 * standard[..., 0] * (1.8**0.5),
    )
    assert torch.allclose(
        first["lab_anisotropic"][..., 1],
        0.05 * standard[..., 1] * (0.2**0.5),
    )
    assert torch.allclose(
        first["lab_biased"],
        0.05
        * (
            standard / (2.0**0.5)
            + torch.tensor((1.0, 0.0))[None, None]
        ),
    )


def test_noise_law_contract_separates_symmetric_and_asymmetric_laws() -> None:
    config = experiment.NoiseLawObservedTwirlConfig()
    datasets = _datasets()
    contract = experiment.noise_law_contract(
        datasets, _noise(datasets, config), config
    )
    assert contract["pass"] is True
    for regime in experiment.REGIMES:
        records = contract["regimes"][regime]
        assert records["isotropic"][
            "maximum_normalized_covariance_reflection_defect"
        ] < 1e-12
        assert records["lab_anisotropic"][
            "median_normalized_covariance_reflection_defect"
        ] >= 0.10
        assert records["lab_biased"][
            "median_normalized_mean_reflection_defect"
        ] >= 0.05
        assert max(item["energy_relative_error"] for item in records.values()) < 0.01


def test_analytic_feature_is_exactly_invariant_to_noisy_action() -> None:
    config = experiment.NoiseLawObservedTwirlConfig()
    task = CircleTaskConfig()
    dataset = experiment._subset_dataset(_datasets()["composition"], 64)
    sensor = calibrated.decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    laws = experiment.generate_noise_laws(
        sample_count=64,
        sensor_steps=task.sensor_steps,
        sigma=config.noise_sigma,
        seed=experiment.NOISE_SEEDS["composition"],
    )
    analytic = calibrated.AnalyticCalibratedCanonicalizer(task)
    for noise in laws.values():
        noisy = sensor.clone()
        noisy[..., :2] += noise
        transformed, transformed_calibration = observed.observed_deck_action(
            noisy, dataset.calibration, task
        )
        difference = (
            analytic(noisy, dataset.calibration)
            - analytic(transformed, transformed_calibration)
        ).abs()
        assert float(difference.max()) <= config.analytic_feature_tolerance


def test_task_gate_is_joint_and_directional() -> None:
    baseline = {
        "exact_bin_accuracy": 0.70,
        "mean_circular_error_radians": 0.20,
        "mean_target_cross_entropy": 1.00,
    }
    passing = {
        "exact_bin_accuracy": 0.671,
        "mean_circular_error_radians": 0.39,
        "mean_target_cross_entropy": 1.09,
    }
    passed, gates = experiment.task_gate(
        passing,
        baseline,
        accuracy_loss_ceiling=0.03,
        circular_error_increase_ceiling=3.141592653589793 / 16.0,
        cross_entropy_increase_ceiling=0.10,
    )
    assert passed is True
    assert all(value for name, value in gates.items() if name.endswith("_pass"))
    failed, _ = experiment.task_gate(
        {**passing, "exact_bin_accuracy": 0.669},
        baseline,
        accuracy_loss_ceiling=0.03,
        circular_error_increase_ceiling=3.141592653589793 / 16.0,
        cross_entropy_increase_ceiling=0.10,
    )
    assert failed is False


@pytest.mark.parametrize(
    ("arguments", "classification", "primary"),
    [
        (
            {},
            "observed_quotient_closed_under_asymmetric_noise",
            True,
        ),
        (
            {"learned_law_passes": {
                "isotropic": True,
                "lab_anisotropic": False,
                "lab_biased": True,
            }},
            "learned_quotient_support_relative_to_noise_law",
            False,
        ),
        (
            {
                "learned_law_passes": {
                    "isotropic": True,
                    "lab_anisotropic": False,
                    "lab_biased": False,
                },
                "any_natural_failure": True,
            },
            "natural_utility_breaks_before_closure",
            False,
        ),
        (
            {"controls_pass": False},
            "nonspecific_target_changing_control",
            False,
        ),
        (
            {"analytic_positive": False},
            "invalid_analytic_positive_control",
            False,
        ),
    ],
)
def test_classification_preserves_registered_outcomes(
    arguments: dict, classification: str, primary: bool
) -> None:
    defaults = {
        "integrity_valid": True,
        "isotropic_positive": True,
        "analytic_positive": True,
        "controls_pass": True,
        "learned_law_passes": {law: True for law in experiment.LAWS},
        "any_natural_failure": False,
    }
    defaults.update(arguments)
    assert experiment.classify_campaign(**defaults) == (classification, primary)


def test_preregistration_file_is_present() -> None:
    assert Path(experiment.PREREGISTRATION_PATH).is_file()


def test_primary_preserves_invalid_positive_control_verdict() -> None:
    campaign = json.loads(PRIMARY.read_text())
    assert experiment._sha256(PRIMARY) == (
        "868ad0ffee546f157e701790c34a83f20bfb3116e78b2f8c5bc34dd7bfe660d7"
    )
    assert campaign["schema_version"] == experiment.SCHEMA_VERSION
    assert campaign["hypothesis_id"] == experiment.HYPOTHESIS_ID
    assert campaign["status"] == "completed"
    assert campaign["implementation_sha256"] == (
        "d4a7e172b0cb9ed5da9a4508c812211882075fcb75db540a17ac6912a8330d6a"
    )
    assert campaign["summary"] == {
        "requested": 10,
        "scheduled": 10,
        "completed": 10,
        "failed": 0,
        "excluded": 0,
        "retries": 0,
        "reused": 0,
        "trained_models": 0,
        "trained_frontends": 0,
        "trained_task_heads": 0,
        "fitted_probes": 0,
        "fitted_observers": 0,
        "fitted_noise_models": 0,
        "fitted_action_parameters": 0,
    }
    aggregates = campaign["aggregates"]
    assert aggregates["classification"] == "invalid_isotropic_positive_control"
    assert aggregates["primary_hypothesis_pass"] is False
    assert aggregates["valid"] is False
    assert aggregates["integrity_valid"] is True
    assert aggregates["isotropic_positive_control"] is False
    assert aggregates["analytic_positive_control"] is False
    assert aggregates["controls_pass"] is True


def test_primary_keeps_narrow_mechanics_separate_from_natural_failure() -> None:
    campaign = json.loads(PRIMARY.read_text())
    arms = campaign["aggregates"]["arms"]
    expected_joint = {
        "analytic_calibrated": {
            "isotropic": 0,
            "lab_anisotropic": 0,
            "lab_biased": 1,
        },
        "learned_calibrated_equivariant": {
            "isotropic": 4,
            "lab_anisotropic": 3,
            "lab_biased": 0,
        },
    }
    for condition in experiment.CONDITIONS:
        for law in experiment.LAWS:
            detail = arms[condition]["laws"][law]
            assert detail["joint_pass_count"] == expected_joint[condition][law]
            assert detail["natural_utility_pass_count"] == expected_joint[condition][law]
            assert detail["action_pass_counts"] == {"pre_block": 5, "full": 5}
            assert detail["twirl_pass_counts"] == {"pre_block": 5, "full": 5}
            assert detail["control_pass_count"] == 0


def test_primary_artifact_and_state_contracts_are_exact() -> None:
    campaign = json.loads(PRIMARY.read_text())
    assert campaign["result_manifest_sha256"] == (
        "7246968593214d5a91b9283e856472cf351b2e921d6712402f9fc128bb457d4d"
    )
    noise_path = Path(campaign["artifacts"]["noise_law_arrays"])
    assert experiment._sha256(noise_path) == (
        "d3771eac8e29f7940df7feaedebe74a5a78fb273cda2e70928c9be9e37ff3ba6"
    )
    for entry in campaign["results"]:
        result_path = Path(entry["path"])
        diagnostics_path = Path(entry["diagnostics_path"])
        assert experiment._sha256(result_path) == entry["result_sha256"]
        assert experiment._sha256(diagnostics_path) == entry["diagnostics_sha256"]
        result = json.loads(result_path.read_text())
        assert result["gates"] == {
            "source_clean_replay": True,
            "cut_replay": True,
            "analytic_feature_invariance": True,
            "state_unchanged": True,
            "finite": True,
            "validity": True,
        }
        assert all(
            result["regimes"][regime]["source_clean_replay_maximum_absolute_error"]
            == 0.0
            for regime in experiment.REGIMES
        )
        assert all(
            result["regimes"][regime]["laws"][law]["maximum_replay_error"]
            == 0.0
            for regime in experiment.REGIMES
            for law in experiment.LAWS
        )
