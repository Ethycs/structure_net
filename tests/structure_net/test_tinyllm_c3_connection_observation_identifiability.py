from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net import (
    tinyllm_c3_connection_observation_identifiability as study,
)


def test_primary_configuration_and_sources_are_frozen() -> None:
    config = study.AuditConfig()
    assert config.source_seeds == study.SOURCE_SEEDS
    assert config.evaluation_samples == 1_024
    assert config.noise_probabilities == study.NOISE_PROBABILITIES
    with pytest.raises(ValueError, match="protocol is frozen"):
        study.AuditConfig(evaluation_samples=512)
    hashes, campaign, records = study.validate_sources()
    assert hashes["preregistration"] == study.PREREGISTRATION_SHA256
    assert hashes["source_campaign"] == study.SOURCE_CAMPAIGN_SHA256
    assert campaign["aggregates"]["primary_hypothesis_pass"] is False
    assert set(records) == set(study.SOURCE_SEEDS)


def test_canonical_total_connection_is_the_same_holonomy() -> None:
    generator = torch.Generator().manual_seed(11)
    connection = torch.randint(
        0,
        study.rel.CHANNELS,
        (31, study.rel.TIME_STEPS - 1),
        generator=generator,
    )
    total = study.canonical_total_connection(connection)
    assert torch.equal(
        connection.sum(dim=1) % study.rel.CHANNELS,
        total.sum(dim=1) % study.rel.CHANNELS,
    )
    assert torch.count_nonzero(total[:, :-1]) == 0


def test_every_single_edge_erasure_has_an_exact_observation_collision() -> None:
    result = study.erasure_audit()
    assert result["pass"] is True
    assert result["pass_count"] == study.rel.TIME_STEPS - 1
    for index, witness in enumerate(result["witnesses"]):
        assert witness["erased_edge_index"] == index
        assert witness["token_maximum_integer_error"] == 0
        assert witness["calibration_maximum_error"] == 0.0
        assert witness["visible_connection_maximum_integer_error"] == 0
        assert witness["changed_connection_indices"] == [index]
        assert witness["absolute_target_separation"] == pytest.approx(1.5)


@pytest.mark.parametrize("probability", study.NOISE_PROBABILITIES)
def test_known_noise_enumeration_matches_closed_form(probability: float) -> None:
    result = study.enumerate_noise_law(probability)
    assert result["pattern_count"] == 3 ** (study.rel.TIME_STEPS - 1)
    assert result["probability_mass"] == pytest.approx(1.0, abs=1e-12)
    assert result["coefficient_maximum_error"] <= study.ENUMERATION_ERROR_MAXIMUM
    assert (
        result["conditional_mean_maximum_error"]
        <= study.ENUMERATION_ERROR_MAXIMUM
    )
    assert result["pass"] is True


def test_current_scalar_gates_have_tiny_exact_noise_tolerances() -> None:
    tolerances = study.gate_noise_tolerances()
    assert (
        tolerances["rmse_gate_maximum_per_edge_error_probability"]
        < tolerances["correlation_gate_maximum_per_edge_error_probability"]
    )
    assert (
        tolerances["joint_scalar_gate_maximum_per_edge_error_probability"]
        < 1e-5
    )
    noise = study.noise_audit(study.AuditConfig())
    assert noise["enumeration_pass"] is True
    assert noise["current_gate_consequence_pass"] is True


def test_frozen_learned_and_analytic_modules_use_only_total_holonomy() -> None:
    _hashes, _campaign, records = study.validate_sources()
    config = study.AuditConfig(
        source_seeds=(study.SOURCE_SEEDS[0],),
        evaluation_samples=128,
        noise_probabilities=(1e-3,),
        allow_underpowered=True,
    )
    result = study.total_holonomy_audit(config, records)
    assert result["cell_count"] == 2
    assert result["maximum_full_to_total_prediction_error"] <= 1e-6
    assert result["analytic_clean_pass_count"] == 2


def test_primary_result_matches_locked_gate() -> None:
    assert study.PRIMARY_OUTPUT.exists()
    result = json.loads(study.PRIMARY_OUTPUT.read_text(encoding="utf-8"))
    assert result["schema_version"] == study.SCHEMA_VERSION
    assert result["hypothesis_id"] == study.HYPOTHESIS_ID
    assert result["status"] == "completed"
    assert all(result["gates"].values())
    assert result["aggregates"]["primary_hypothesis_pass"] is True
    assert result["aggregates"]["classification"] == study.CLASSIFICATION_PASS
    assert result["accounting"]["optimizer_steps"] == 0
    assert result["accounting"]["tinyllm_models_instantiated"] == 0
    assert Path(result["source_campaign"]["path"]) == study.SOURCE_CAMPAIGN_PATH
