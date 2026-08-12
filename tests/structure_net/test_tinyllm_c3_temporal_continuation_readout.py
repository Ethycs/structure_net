import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.structure_net.tinyllm_c3_temporal_continuation_readout import (
    C3ReadoutConfig,
    FIT_ARMS,
    SOURCE_ROOT,
    _source_campaign,
    _task_gate,
    aggregate_details,
    deterministic_target_permutation,
)
from experiments.structure_net.tinyllm_frozen_interval_readout_decomposition import (
    apply_ridge_map,
    fit_ridge_map,
    fixed_interval_posterior,
)
from experiments.structure_net.tinyllm_c3_temporal_quotient_training import (
    _targets,
)


def test_fixed_interval_decoder_replays_c3_target_chart() -> None:
    target = torch.linspace(-1.0, 1.0, 257)
    expected, _ = _targets(target, 16)
    measured = fixed_interval_posterior(target, 16)
    assert torch.max(torch.abs(measured - expected.double())) <= 2e-6


def test_closed_form_ridge_recovers_affine_scalar() -> None:
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(512, 4, generator=generator, dtype=torch.float64)
    target = 0.7 * features[:, 0] - 1.2 * features[:, 2] + 0.3
    fitted = fit_ridge_map(
        features,
        target,
        ridge_lambda=1e-8,
        standardization_floor=1e-6,
    )
    prediction = apply_ridge_map(fitted, features).reshape(-1)
    assert torch.sqrt(torch.mean((prediction - target).square())) < 1e-6


def test_target_permutation_is_deterministic_nonidentity() -> None:
    first = deterministic_target_permutation(4096, 7)
    second = deterministic_target_permutation(4096, 7)
    other = deterministic_target_permutation(4096, 17)
    assert torch.equal(first, second)
    assert not torch.equal(first, torch.arange(4096))
    assert not torch.equal(first, other)
    assert torch.equal(torch.sort(first).values, torch.arange(4096))


def test_primary_configuration_rejects_post_registration_changes() -> None:
    config = C3ReadoutConfig()
    with pytest.raises(ValueError, match="primary C3 readout configuration"):
        replace(config, ridge_lambda=1e-3)


def test_task_gate_requires_all_four_endpoints() -> None:
    config = C3ReadoutConfig(allow_underpowered=True)
    passing = {
        "posterior_mean_correlation": 0.90,
        "exact_bin_accuracy": 0.50,
        "target_cross_entropy": 1.80,
        "predicted_bin_coverage": 14,
    }
    assert _task_gate(passing, "composition", config)
    for name, failing_value in (
        ("posterior_mean_correlation", 0.899),
        ("exact_bin_accuracy", 0.499),
        ("target_cross_entropy", 1.801),
        ("predicted_bin_coverage", 13),
    ):
        failing = dict(passing)
        failing[name] = failing_value
        assert not _task_gate(failing, "composition", config)


def _mock_detail(
    *, typed: bool, untyped: bool, recalibration: bool, shuffled: bool = False
) -> dict:
    values = {
        "typed_final_readout": typed,
        "untyped_final_readout": untyped,
        "output_scalar_recalibration": recalibration,
    }
    return {
        "gates": {
            "validity": True,
            "exact_temporal_bypass": True,
            "arms": {
                arm: {
                    "true_both_shifts": values[arm],
                    "shuffled_both_shifts": shuffled,
                }
                for arm in FIT_ARMS
            },
        }
    }


def test_aggregate_classifies_typed_and_untyped_outcomes() -> None:
    config = C3ReadoutConfig(allow_underpowered=True)
    typed = [
        _mock_detail(typed=index < 4, untyped=True, recalibration=False)
        for index in range(5)
    ]
    aggregate = aggregate_details(typed, config)
    assert aggregate["primary_hypothesis_pass"] is True
    assert aggregate["classification"] == (
        "frozen_continuation_typed_state_sufficient_answer_rows_inadequate"
    )

    untyped = [
        _mock_detail(typed=False, untyped=index < 4, recalibration=False)
        for index in range(5)
    ]
    aggregate = aggregate_details(untyped, config)
    assert aggregate["primary_hypothesis_pass"] is False
    assert aggregate["classification"] == (
        "frozen_continuation_task_state_untyped_only"
    )


def test_aggregate_rejects_nonspecific_success() -> None:
    config = C3ReadoutConfig(allow_underpowered=True)
    details = [
        _mock_detail(typed=True, untyped=True, recalibration=True, shuffled=True)
        for _ in range(5)
    ]
    aggregate = aggregate_details(details, config)
    assert aggregate["primary_hypothesis_pass"] is False
    assert aggregate["classification"] == "invalid_target_shuffle_specificity"


def test_sealed_source_population_revalidates() -> None:
    campaign, details = _source_campaign(SOURCE_ROOT)
    assert campaign["aggregates"]["classification"] == (
        "c3_positive_control_task_failure"
    )
    assert set(details) == {7, 17, 29, 41, 53}
    assert sum(detail["gates"]["natural_task"] for detail in details.values()) == 2


def test_source_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    source = json.loads((SOURCE_ROOT / "campaign_results.json").read_text())
    source["aggregates"]["classification"] = "c3_d6_structured_quotient_supported"
    (tmp_path / "campaign_results.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="source C3 campaign changed"):
        _source_campaign(tmp_path)
