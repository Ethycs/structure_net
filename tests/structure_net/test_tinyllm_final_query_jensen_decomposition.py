from __future__ import annotations

import torch

import experiments.structure_net.tinyllm_final_query_jensen_decomposition as study


def _small_config(**changes: object) -> study.FinalQueryJensenConfig:
    values: dict[str, object] = {
        "regimes": ("training_support",),
        "fiber_pairs": 8,
        "batch_size": 16,
        "device": "cpu",
        "allow_underpowered": True,
    }
    values.update(changes)
    return study.FinalQueryJensenConfig(**values)


def test_primary_configuration_and_registration_are_locked() -> None:
    config = study.FinalQueryJensenConfig()
    assert config.regimes == study.REGIMES
    assert config.fiber_pairs == 512
    assert config.jensen_coverage_floor == 0.90
    assert config.near_complete_remainder_ratio == 0.10
    assert study._sha256(study.PREREGISTRATION_PATH) == (
        study.PREREGISTRATION_SHA256
    )


def test_source_artifacts_and_valid_d8_parent_are_locked() -> None:
    task, hashes, detail = study._validate_sources(_small_config())
    assert task.phase_bins == 16
    assert hashes["posterior_campaign"] == study.POSTERIOR_CAMPAIGN_SHA256
    assert hashes["checkpoint_d8_cosine_interval"] == study.CHECKPOINT_SHA256
    assert detail["gates"]["validity"] is True
    assert detail["summary"]["mature_causal_front"] is None


def test_logit_midpoint_has_exact_jensen_accounting() -> None:
    generator = torch.Generator().manual_seed(20260810)
    endpoint = torch.randn((32, 2, 7), generator=generator, dtype=torch.float64)
    targets = torch.softmax(
        torch.randn((32, 7), generator=generator, dtype=torch.float64), -1
    )
    midpoint = endpoint.mean(1)
    parts = study.decompose_pair_logits(endpoint, midpoint, targets)
    assert torch.min(parts["jensen_gain"]).item() >= -1e-12
    assert torch.max(parts["accounting_residual"].abs()).item() <= 1e-14
    assert torch.max(parts["nonlinear_remainder"].abs()).item() <= 1e-14
    assert torch.max(
        torch.abs(parts["observed_gain"] - parts["jensen_gain"])
    ).item() <= 1e-14


def test_layernorm_remainder_is_separated_from_generic_jensen_gain() -> None:
    endpoint = torch.tensor(
        [
            [[3.0, 0.0, -1.0], [0.0, 3.0, -1.0]],
            [[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor(
        [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=torch.float64
    )
    activation = endpoint.mean(1) + torch.tensor(
        [[0.25, 0.25, -0.5], [-0.25, -0.25, 0.5]], dtype=torch.float64
    )
    parts = study.decompose_pair_logits(endpoint, activation, targets)
    assert torch.min(parts["jensen_gain"]).item() >= -1e-12
    assert torch.max(parts["accounting_residual"].abs()).item() <= 1e-14
    assert torch.any(parts["nonlinear_remainder"].abs() > 1e-4)
    aggregate = study._aggregate(parts)
    assert abs(
        aggregate["observed_activation_gain"]
        - (
            aggregate["jensen_gain"]
            - aggregate["nonlinear_remainder"]
        )
    ) <= 1e-14


def test_jensen_contract_is_nonspecific_to_semantic_pairing() -> None:
    generator = torch.Generator().manual_seed(31)
    endpoint = torch.randn((16, 2, 5), generator=generator, dtype=torch.float64)
    wrong_target = torch.softmax(
        torch.randn((16, 5), generator=generator, dtype=torch.float64), -1
    )
    partner = torch.arange(16) ^ 1
    reassigned = endpoint.index_select(0, partner)
    parts = study.decompose_pair_logits(
        reassigned, reassigned.mean(1), wrong_target
    )
    assert torch.min(parts["jensen_gain"]).item() >= -1e-12
    assert torch.max(parts["accounting_residual"].abs()).item() <= 1e-14


def test_negative_observed_gain_has_no_positive_gain_ratio() -> None:
    endpoint = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0]]], dtype=torch.float64
    )
    target = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    activation = torch.tensor([[8.0, -8.0]], dtype=torch.float64)
    result = study._aggregate(
        study.decompose_pair_logits(endpoint, activation, target)
    )
    assert result["observed_activation_gain"] < 0.0
    assert result["jensen_over_observed_gain"] is None


def test_locked_classification_distinguishes_remainder_regimes() -> None:
    config = study.FinalQueryJensenConfig()
    base = {
        "gates": {
            "material_activation_gain": True,
            "near_complete_jensen": True,
        }
    }
    regimes = {name: base for name in study.REGIMES}
    assert (
        study._classification(config, True, True, regimes)
        == "generic_jensen_near_complete"
    )
    modulated = {
        name: {
            "gates": {
                "material_activation_gain": True,
                "near_complete_jensen": name != "outside_range",
            }
        }
        for name in study.REGIMES
    }
    assert (
        study._classification(config, True, True, modulated)
        == "generic_jensen_sufficient_with_layernorm_modulation"
    )
    assert (
        study._classification(config, True, False, modulated)
        == "activation_geometry_materially_assists_jensen"
    )
