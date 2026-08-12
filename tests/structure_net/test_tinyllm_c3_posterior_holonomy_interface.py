from __future__ import annotations

import json

import pytest
import torch

from experiments.structure_net import (
    tinyllm_c3_posterior_holonomy_interface as study,
)


def _small_config() -> study.InterfaceConfig:
    return study.InterfaceConfig(
        simplex_denominator=9,
        phase_points=64,
        source_seeds=(study.SOURCE_SEEDS[0],),
        registered_posteriors=study.REGISTERED_POSTERIORS,
        allow_underpowered=True,
    )


def test_primary_configuration_and_sources_are_frozen() -> None:
    config = study.InterfaceConfig()
    assert config.simplex_denominator == 63
    assert config.phase_points == 2_048
    assert config.source_seeds == study.SOURCE_SEEDS
    with pytest.raises(ValueError, match="protocol is frozen"):
        study.InterfaceConfig(simplex_denominator=21)
    hashes, predecessor, records = study.validate_sources()
    assert hashes["preregistration"] == study.PREREGISTRATION_SHA256
    assert hashes["predecessor_result"] == study.PREDECESSOR_RESULT_SHA256
    assert predecessor["aggregates"]["primary_hypothesis_pass"] is True
    assert set(records) == set(study.SOURCE_SEEDS)


def test_simplex_grid_count_and_moment_inverse_are_exact() -> None:
    config = study.InterfaceConfig()
    posterior = study.posterior_simplex(config.simplex_denominator)
    assert posterior.shape == (2_080, 3)
    moment = study.posterior_moment(posterior)
    reconstructed = study.inverse_posterior_moment(moment)
    assert torch.max(torch.abs(reconstructed - posterior)) <= 1e-12
    assert torch.min(reconstructed) >= -1e-12
    result = study.simplex_inverse_audit(config)
    assert result["pass"] is True
    assert result["barycenter"]["moment_magnitude"] <= 1e-12


def test_conditional_mean_and_risk_factorization_on_small_grid() -> None:
    result = study.factorization_and_risk_audit(_small_config())
    assert result["pass"] is True
    assert result["maximum_conditional_mean_factorization_error"] <= 1e-12
    assert result["maximum_soft_risk_formula_error"] <= 1e-12
    assert result["maximum_hard_risk_formula_error"] <= 1e-12
    assert result["minimum_hard_minus_soft_regret"] >= -1e-12
    assert result["minimum_nonvertex_hard_minus_soft_regret"] > 0.0


def test_error_coordinate_relabeling_is_covariant() -> None:
    result = study.covariance_audit(_small_config())
    assert result["pass"] is True
    assert len(result["rows"]) == 3
    assert all(item["pass"] for item in result["rows"])


def test_frozen_linear_heads_accept_the_soft_interface_without_state_change() -> None:
    _hashes, predecessor, records = study.validate_sources()
    result = study.frozen_replay_audit(_small_config(), predecessor, records)
    assert result["cell_count"] == 2
    assert result["pass_count"] == 2
    assert result["maximum_replay_error"] <= 1e-6
    assert result["all_source_states_unchanged"] is True


def test_primary_result_matches_locked_gate() -> None:
    assert study.PRIMARY_OUTPUT.exists()
    result = json.loads(study.PRIMARY_OUTPUT.read_text(encoding="utf-8"))
    assert result["schema_version"] == study.SCHEMA_VERSION
    assert result["hypothesis_id"] == study.HYPOTHESIS_ID
    assert result["status"] == "completed"
    assert all(result["gates"].values())
    assert result["aggregates"]["classification"] == study.CLASSIFICATION_PASS
    assert result["aggregates"]["posterior_estimator_training_licensed"] is False
    assert result["aggregates"]["unrestricted_tinyllm_training_licensed"] is False
    assert result["accounting"]["optimizer_steps"] == 0
    assert result["accounting"]["tinyllm_models_instantiated"] == 0
