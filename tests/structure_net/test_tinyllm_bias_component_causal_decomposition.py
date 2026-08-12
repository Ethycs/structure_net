from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

import experiments.structure_net.tinyllm_bias_component_causal_decomposition as bias


def _population(
    counts: dict[tuple[str, str], int]
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for condition in bias.CONDITIONS:
        for index, seed in enumerate(bias.SEEDS):
            results.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "variant_seed_gates": {
                        variant: index < counts[(condition, variant)]
                        for variant in bias.VARIANTS
                    },
                    "gates": {"validity": True},
                }
            )
    return results


def test_default_config_is_registered_primary() -> None:
    config = bias.BiasComponentCausalConfig()
    assert config.conditions == bias.CONDITIONS
    assert config.seeds == bias.SEEDS
    assert config.selected_multiplier == 0.625
    assert config.selected_noise_sigma == 0.03125
    assert config.sample_limit is None
    assert not config.allow_underpowered


def test_primary_rejects_scale_and_population_changes() -> None:
    with pytest.raises(ValueError, match="selected multiplier"):
        bias.BiasComponentCausalConfig(
            selected_multiplier=0.5, allow_underpowered=True
        )
    with pytest.raises(ValueError, match="required seed count"):
        bias.BiasComponentCausalConfig(seeds=(7,))


def test_component_construction_replays_frozen_biased_array() -> None:
    config = bias.BiasComponentCausalConfig()
    arrays = bias._load_base_noise(config)
    components = bias.construct_components(arrays, config)
    contract = bias.component_contract(components, arrays, config)
    assert contract["pass"]
    assert contract["no_new_random_draws"]
    for regime in bias.REGIMES:
        record = contract["regimes"][regime]
        assert (
            record["full_plus_reconstruction_maximum_absolute_error"]
            <= config.component_reconstruction_tolerance
        )
        assert record["centered_reconstruction_maximum_absolute_error"] == 0.0
        assert record["mean_plus_reconstruction_maximum_absolute_error"] == 0.0
        assert (
            record["sign_energy_relative_difference"]
            <= config.sign_energy_relative_tolerance
        )


def test_selected_dose_source_is_pinned_with_bias_failure_controls() -> None:
    campaign, path, details, diagnostics = bias._load_source_campaign(
        bias.BiasComponentCausalConfig()
    )
    assert bias._sha256(path) == bias.SOURCE_CAMPAIGN_SHA256
    assert campaign["aggregates"]["classification"] == (
        "asymmetric_law_breaks_within_isotropic_window"
    )
    assert len(details) == 10
    assert len(diagnostics) == 10
    assert sum(
        detail["law_seed_gates"]["lab_biased"]
        for detail in details.values()
        if detail["condition"] == "analytic_calibrated"
    ) == 1
    assert sum(
        detail["law_seed_gates"]["lab_biased"]
        for detail in details.values()
        if detail["condition"] == "learned_calibrated_equivariant"
    ) == 3


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            dict(
                integrity_valid=False,
                component_contract_pass=True,
                centered_passes_both_arms=True,
                mean_plus_fails_both_arms=True,
                full_plus_fails_both_arms=True,
                mean_plus_passes_both_arms=False,
            ),
            ("invalid_integrity", False),
        ),
        (
            dict(
                integrity_valid=True,
                component_contract_pass=False,
                centered_passes_both_arms=True,
                mean_plus_fails_both_arms=True,
                full_plus_fails_both_arms=True,
                mean_plus_passes_both_arms=False,
            ),
            ("invalid_component_reconstruction", False),
        ),
        (
            dict(
                integrity_valid=True,
                component_contract_pass=True,
                centered_passes_both_arms=False,
                mean_plus_fails_both_arms=True,
                full_plus_fails_both_arms=True,
                mean_plus_passes_both_arms=False,
            ),
            ("centered_stochastic_breaks_utility", False),
        ),
        (
            dict(
                integrity_valid=True,
                component_contract_pass=True,
                centered_passes_both_arms=True,
                mean_plus_fails_both_arms=True,
                full_plus_fails_both_arms=True,
                mean_plus_passes_both_arms=False,
            ),
            ("deterministic_mean_sufficient", True),
        ),
        (
            dict(
                integrity_valid=True,
                component_contract_pass=True,
                centered_passes_both_arms=True,
                mean_plus_fails_both_arms=False,
                full_plus_fails_both_arms=True,
                mean_plus_passes_both_arms=True,
            ),
            ("mean_noise_interaction", False),
        ),
        (
            dict(
                integrity_valid=True,
                component_contract_pass=True,
                centered_passes_both_arms=True,
                mean_plus_fails_both_arms=False,
                full_plus_fails_both_arms=True,
                mean_plus_passes_both_arms=False,
            ),
            ("arm_specific_or_underdetermined", False),
        ),
    ],
)
def test_registered_classification_order(
    arguments: dict[str, bool], expected: tuple[str, bool]
) -> None:
    assert bias.classify_campaign(**arguments) == expected


def test_population_aggregate_identifies_mean_sufficiency_and_sign() -> None:
    counts = {
        (condition, variant): 5
        for condition in bias.CONDITIONS
        for variant in bias.VARIANTS
    }
    counts[("analytic_calibrated", "full_plus")] = 1
    counts[("learned_calibrated_equivariant", "full_plus")] = 3
    counts[("analytic_calibrated", "mean_plus")] = 2
    counts[("learned_calibrated_equivariant", "mean_plus")] = 3
    counts[("analytic_calibrated", "full_minus")] = 2
    counts[("learned_calibrated_equivariant", "full_minus")] = 3
    aggregate = bias.aggregate_results(
        _population(counts), bias.BiasComponentCausalConfig()
    )
    assert aggregate["integrity_valid"]
    assert aggregate["centered_passes_both_arms"]
    assert aggregate["mean_plus_fails_both_arms"]
    assert aggregate["full_plus_fails_both_arms"]
    assert aggregate["sign_classification"] == "bidirectional_mean_magnitude"


def test_campaign_reuse_requires_exact_artifact_hashes(tmp_path: Path) -> None:
    config = replace(
        bias.BiasComponentCausalConfig(),
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
            "result_sha256": bias._sha256(result_path),
            "diagnostics_path": str(diagnostics_path),
            "diagnostics_sha256": bias._sha256(diagnostics_path),
        }
    ]
    campaign = {
        "status": "completed",
        "schema_version": bias.SCHEMA_VERSION,
        "hypothesis_id": bias.HYPOTHESIS_ID,
        "configuration": bias._json_config(config),
        "implementation_sha256": "implementation",
        "source_campaign_sha256": bias.SOURCE_CAMPAIGN_SHA256,
        "source_selected_arrays_sha256": bias.SOURCE_SELECTED_ARRAYS_SHA256,
        "results": entries,
        "result_manifest_sha256": bias._json_hash(entries),
    }
    assert bias._campaign_reusable(campaign, config, "implementation")
    result_path.write_text(json.dumps({"changed": True}), encoding="utf-8")
    assert not bias._campaign_reusable(campaign, config, "implementation")

