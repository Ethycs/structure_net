from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

import experiments.structure_net.tinyllm_noise_law_dose_localization as dose


PRIMARY_ROOT = Path(
    "data/experiments/tinyllm_noise_law_dose_localization/"
    "20260810_d10_preregistered"
)
PRIMARY_CAMPAIGN_SHA256 = (
    "9b05823ebdb88bd828f27699da596dc5e7dcf0c4af5e13f1664fa70e5111f9bd"
)
PRIMARY_RUNNER_SHA256 = (
    "39a72dd535f96f13bae644c74096b298b85fb8587d980211dc489ed463aeb725"
)


def _stage1_population(
    *,
    analytic_counts: dict[float, int],
    learned_counts: dict[float, int],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for condition, counts in (
        ("analytic_calibrated", analytic_counts),
        ("learned_calibrated_equivariant", learned_counts),
    ):
        for index, seed in enumerate(dose.SEEDS):
            results.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "natural_seed_gates": {
                        dose._dose_key(multiplier): index < counts[multiplier]
                        for multiplier in dose.DOSE_MULTIPLIERS
                    },
                    "gates": {
                        "zero_dose_replay": True,
                        "validity": True,
                    },
                }
            )
    return results


def _stage2_population(
    *,
    joint_counts: dict[tuple[str, str], int],
    control_counts: dict[tuple[str, str], int] | None = None,
) -> list[dict[str, object]]:
    controls = control_counts or {
        (condition, law): 0 for condition in dose.CONDITIONS for law in dose.LAWS
    }
    results: list[dict[str, object]] = []
    for condition in dose.CONDITIONS:
        for index, seed in enumerate(dose.SEEDS):
            results.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "law_seed_gates": {
                        law: index < joint_counts[(condition, law)]
                        for law in dose.LAWS
                    },
                    "natural_seed_gates": {
                        law: index < joint_counts[(condition, law)]
                        for law in dose.LAWS
                    },
                    "control_seed_gates": {
                        law: index < controls[(condition, law)]
                        for law in dose.LAWS
                    },
                    "action_seed_gates": {
                        law: {cut: True for cut in dose.CUTS}
                        for law in dose.LAWS
                    },
                    "twirl_seed_gates": {
                        law: {cut: True for cut in dose.CUTS}
                        for law in dose.LAWS
                    },
                    "gates": {"validity": True},
                }
            )
    return results


def test_default_config_is_the_registered_primary() -> None:
    config = dose.NoiseLawDoseLocalizationConfig()
    assert config.dose_multipliers == dose.DOSE_MULTIPLIERS
    assert config.conditions == dose.CONDITIONS
    assert config.seeds == dose.SEEDS
    assert config.sample_limit is None
    assert not config.allow_underpowered


def test_primary_rejects_changed_population_and_dose_ladder() -> None:
    with pytest.raises(ValueError, match="required seed count"):
        dose.NoiseLawDoseLocalizationConfig(seeds=(7,))
    with pytest.raises(ValueError, match="nested dose ladder"):
        dose.NoiseLawDoseLocalizationConfig(
            dose_multipliers=(0.0, 0.5, 1.0), allow_underpowered=True
        )


def test_frozen_source_and_scaled_arrays_match_registered_hashes() -> None:
    config = dose.NoiseLawDoseLocalizationConfig()
    campaign, campaign_path, arrays = dose._load_source_campaign(config)
    assert dose._sha256(campaign_path) == dose.SOURCE_CAMPAIGN_SHA256
    assert campaign["aggregates"]["classification"] == (
        "invalid_isotropic_positive_control"
    )
    half = dose.scaled_noise_arrays(arrays, 0.5)
    zero = dose.scaled_noise_arrays(arrays, 0.0)
    for regime in dose.REGIMES:
        for law in dose.LAWS:
            assert torch.equal(half[regime][law], arrays[regime][law] * 0.5)
            assert torch.count_nonzero(zero[regime][law]) == 0


def test_selection_uses_largest_common_prefix_valid_dose() -> None:
    analytic = {
        multiplier: (5 if multiplier <= 0.5 else 3)
        for multiplier in dose.DOSE_MULTIPLIERS
    }
    learned = {
        multiplier: (5 if multiplier <= 0.75 else 3)
        for multiplier in dose.DOSE_MULTIPLIERS
    }
    selected = dose.select_prefix_valid_dose(
        _stage1_population(analytic_counts=analytic, learned_counts=learned),
        dose.NoiseLawDoseLocalizationConfig(),
    )
    assert selected["selected_multiplier"] == 0.5
    assert selected["selected_noise_sigma"] == 0.025
    assert selected["zero_dose_control_pass"]
    assert not selected["selection_uses_asymmetric_outcomes"]


def test_selection_does_not_skip_a_failed_lower_dose() -> None:
    analytic = {multiplier: 5 for multiplier in dose.DOSE_MULTIPLIERS}
    learned = {multiplier: 5 for multiplier in dose.DOSE_MULTIPLIERS}
    analytic[0.25] = 3
    selected = dose.select_prefix_valid_dose(
        _stage1_population(analytic_counts=analytic, learned_counts=learned),
        dose.NoiseLawDoseLocalizationConfig(),
    )
    assert selected["selected_multiplier"] == 0.125
    assert not selected["doses"][dose._dose_key(0.375)]["prefix_valid"]


@pytest.mark.parametrize(
    ("arguments", "classification", "primary", "evaluable"),
    [
        (
            dict(
                integrity_valid=False,
                zero_dose_control_pass=True,
                selected_multiplier=0.5,
                controls_pass=True,
                isotropic_joint_population_pass=True,
                all_laws_joint_population_pass=True,
            ),
            "invalid_integrity",
            False,
            False,
        ),
        (
            dict(
                integrity_valid=True,
                zero_dose_control_pass=False,
                selected_multiplier=0.5,
                controls_pass=True,
                isotropic_joint_population_pass=True,
                all_laws_joint_population_pass=True,
            ),
            "invalid_zero_dose_control",
            False,
            False,
        ),
        (
            dict(
                integrity_valid=True,
                zero_dose_control_pass=True,
                selected_multiplier=None,
                controls_pass=None,
                isotropic_joint_population_pass=None,
                all_laws_joint_population_pass=None,
            ),
            "no_common_nonzero_utility_window",
            False,
            False,
        ),
        (
            dict(
                integrity_valid=True,
                zero_dose_control_pass=True,
                selected_multiplier=0.5,
                controls_pass=False,
                isotropic_joint_population_pass=True,
                all_laws_joint_population_pass=True,
            ),
            "nonspecific_target_changing_control",
            False,
            False,
        ),
        (
            dict(
                integrity_valid=True,
                zero_dose_control_pass=True,
                selected_multiplier=0.5,
                controls_pass=True,
                isotropic_joint_population_pass=False,
                all_laws_joint_population_pass=False,
            ),
            "isotropic_closure_fails_at_selected_dose",
            False,
            True,
        ),
        (
            dict(
                integrity_valid=True,
                zero_dose_control_pass=True,
                selected_multiplier=0.5,
                controls_pass=True,
                isotropic_joint_population_pass=True,
                all_laws_joint_population_pass=False,
            ),
            "asymmetric_law_breaks_within_isotropic_window",
            False,
            True,
        ),
        (
            dict(
                integrity_valid=True,
                zero_dose_control_pass=True,
                selected_multiplier=0.5,
                controls_pass=True,
                isotropic_joint_population_pass=True,
                all_laws_joint_population_pass=True,
            ),
            "additive_noise_closed_at_selected_dose",
            True,
            True,
        ),
    ],
)
def test_registered_classification_order(
    arguments: dict[str, object],
    classification: str,
    primary: bool,
    evaluable: bool,
) -> None:
    assert dose.classify_campaign(**arguments) == (
        classification,
        primary,
        evaluable,
    )


def test_stage2_aggregate_counts_joint_and_control_populations() -> None:
    joint = {
        (condition, law): 5
        for condition in dose.CONDITIONS
        for law in dose.LAWS
    }
    joint[("learned_calibrated_equivariant", "lab_biased")] = 3
    controls = {
        (condition, law): 0
        for condition in dose.CONDITIONS
        for law in dose.LAWS
    }
    aggregate = dose.aggregate_stage2(
        _stage2_population(joint_counts=joint, control_counts=controls),
        dose.NoiseLawDoseLocalizationConfig(),
    )
    assert aggregate["integrity_valid"]
    assert aggregate["controls_pass"]
    assert aggregate["isotropic_joint_population_pass"]
    assert not aggregate["all_laws_joint_population_pass"]
    assert (
        aggregate["arms"]["learned_calibrated_equivariant"]["laws"][
            "lab_biased"
        ]["joint_pass_count"]
        == 3
    )


def test_campaign_reuse_requires_result_and_diagnostics_hashes(
    tmp_path: Path,
) -> None:
    config = replace(
        dose.NoiseLawDoseLocalizationConfig(),
        conditions=("analytic_calibrated",),
        seeds=(7,),
        required_seed_passes=1,
        sample_limit=64,
        allow_underpowered=True,
    )
    result_path = tmp_path / "result.json"
    diagnostics_path = tmp_path / "diagnostics.npz"
    result_path.write_text("{}\n", encoding="utf-8")
    diagnostics_path.write_bytes(b"diagnostics")
    entries = [
        {
            "path": str(result_path),
            "result_sha256": dose._sha256(result_path),
            "diagnostics_path": str(diagnostics_path),
            "diagnostics_sha256": dose._sha256(diagnostics_path),
        }
    ]
    campaign = {
        "status": "completed",
        "schema_version": dose.SCHEMA_VERSION,
        "hypothesis_id": dose.HYPOTHESIS_ID,
        "configuration": dose._json_config(config),
        "implementation_sha256": "implementation",
        "source_campaign_sha256": dose.SOURCE_CAMPAIGN_SHA256,
        "source_noise_file_sha256": dose.SOURCE_NOISE_FILE_SHA256,
        "source_noise_content_sha256": dose.SOURCE_NOISE_CONTENT_SHA256,
        "stage1_results": entries,
        "stage2_results": [],
        "result_manifest_sha256": dose._json_hash(entries),
    }
    assert dose._campaign_reusable(campaign, config, "implementation")
    result_path.write_text(json.dumps({"changed": True}), encoding="utf-8")
    assert not dose._campaign_reusable(campaign, config, "implementation")


def test_primary_campaign_is_pinned_complete_and_resumable() -> None:
    campaign_path = PRIMARY_ROOT / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    assert dose._sha256(campaign_path) == PRIMARY_CAMPAIGN_SHA256
    assert dose._sha256(Path(dose.__file__)) == PRIMARY_RUNNER_SHA256
    assert (
        dose._sha256(dose.PREREGISTRATION_PATH)
        == dose.PREREGISTRATION_SHA256
    )
    assert campaign["status"] == "completed"
    assert campaign["implementation_sha256"] == dose._implementation_digest()
    assert dose._campaign_reusable(
        campaign,
        dose.NoiseLawDoseLocalizationConfig(device="cuda:2"),
        dose._implementation_digest(),
    )
    assert campaign["summary"] == {
        "completed_stage1": 10,
        "completed_stage2": 10,
        "excluded": 0,
        "failed": 0,
        "fitted_actions": 0,
        "fitted_noise_models": 0,
        "fitted_observers": 0,
        "fitted_probes": 0,
        "requested_stage1": 10,
        "requested_stage2": 10,
        "reused_stage1": 0,
        "reused_stage2": 0,
        "trained_frontends": 0,
        "trained_models": 0,
        "trained_task_heads": 0,
    }


def test_primary_localizes_the_largest_common_prefix_valid_dose() -> None:
    campaign = json.loads(
        (PRIMARY_ROOT / "campaign_results.json").read_text(encoding="utf-8")
    )
    stage1 = campaign["stage1"]
    assert stage1["zero_dose_control_pass"]
    assert stage1["integrity_valid"]
    assert not stage1["selection_uses_asymmetric_outcomes"]
    assert stage1["selected_multiplier"] == 0.625
    assert stage1["selected_noise_sigma"] == 0.03125
    assert stage1["doses"]["0.625"]["prefix_valid"]
    assert not stage1["doses"]["0.750"]["prefix_valid"]
    assert stage1["doses"]["0.625"]["arms"] == {
        "analytic_calibrated": {"natural_utility_pass_count": 5},
        "learned_calibrated_equivariant": {
            "natural_utility_pass_count": 5
        },
    }
    assert stage1["doses"]["0.750"]["arms"] == {
        "analytic_calibrated": {"natural_utility_pass_count": 2},
        "learned_calibrated_equivariant": {
            "natural_utility_pass_count": 5
        },
    }


def test_primary_finds_a_bias_specific_natural_utility_failure() -> None:
    campaign = json.loads(
        (PRIMARY_ROOT / "campaign_results.json").read_text(encoding="utf-8")
    )
    assert campaign["aggregates"] == {
        "classification": "asymmetric_law_breaks_within_isotropic_window",
        "integrity_valid": True,
        "maximum_control_seed_passes": 1,
        "primary_evaluable": True,
        "primary_hypothesis_pass": False,
        "required_seed_passes": 4,
        "selected_multiplier": 0.625,
        "selected_noise_sigma": 0.03125,
        "valid": True,
        "zero_dose_control_pass": True,
    }
    arms = campaign["stage2"]["arms"]
    assert {
        condition: {
            law: arms[condition]["laws"][law]["joint_pass_count"]
            for law in dose.LAWS
        }
        for condition in dose.CONDITIONS
    } == {
        "analytic_calibrated": {
            "isotropic": 5,
            "lab_anisotropic": 4,
            "lab_biased": 1,
        },
        "learned_calibrated_equivariant": {
            "isotropic": 5,
            "lab_anisotropic": 5,
            "lab_biased": 3,
        },
    }
    for condition in dose.CONDITIONS:
        for law in dose.LAWS:
            record = arms[condition]["laws"][law]
            assert record["joint_pass_count"] == record[
                "natural_utility_pass_count"
            ]
            assert record["control_pass_count"] == 0
            assert record["action_pass_counts"] == {
                "full": 5,
                "pre_block": 5,
            }
            assert record["twirl_pass_counts"] == {
                "full": 5,
                "pre_block": 5,
            }
