from __future__ import annotations

import pytest
import torch

from experiments.structure_net.tinyllm_affine_gauge_transport import (
    AffineGaugeConfig,
    canonicalize,
    classify_aggregates,
    embedding_transport_errors,
    fit_affine_chart,
)


def test_primary_configuration_is_locked() -> None:
    config = AffineGaugeConfig()
    assert config.presets == ("d6", "d10")
    assert config.seeds == (7, 17, 29, 41, 53)
    assert config.slope_absolute_minimum == 1e-4
    assert config.transport_tolerance == 2e-6
    with pytest.raises(ValueError, match="configuration changed"):
        AffineGaugeConfig(transport_tolerance=1e-4)


@pytest.mark.parametrize(("alpha", "beta"), [(2.5, -0.3), (-0.125, 0.07)])
def test_affine_fit_and_canonicalization(alpha: float, beta: float) -> None:
    target = torch.linspace(-1.0, 1.0, 257)
    scalar = alpha * target + beta
    fit = fit_affine_chart(scalar, target)
    assert fit["alpha"] == pytest.approx(alpha, abs=1e-7)
    assert fit["beta"] == pytest.approx(beta, abs=1e-7)
    assert fit["r_squared"] == pytest.approx(1.0)
    restored = canonicalize(scalar, fit["alpha"], fit["beta"])
    assert torch.allclose(restored, target, atol=1e-6, rtol=0.0)


def test_inverse_embedding_transport_preserves_injection() -> None:
    generator = torch.Generator().manual_seed(17)
    scalar = torch.randn(512, generator=generator)
    weight = torch.randn(32, 1, generator=generator)
    bias = torch.randn(32, generator=generator)
    errors = embedding_transport_errors(
        [scalar], weight, bias, alpha=-0.15, beta=0.08
    )
    assert errors["maximum_float32_absolute_error"] <= 1e-6
    assert errors["maximum_float64_absolute_error"] <= 1e-12


def _strata(
    d6_joint: bool,
    d10_joint: bool,
    *,
    d6_front: bool = False,
    d10_front: bool = False,
    specificity: bool = True,
) -> dict:
    return {
        "d6": {
            "family_gate": d6_joint,
            "canonical_front_family_gate": d6_front,
            "shuffled_specificity_gate": specificity,
        },
        "d10": {
            "family_gate": d10_joint,
            "canonical_front_family_gate": d10_front,
            "shuffled_specificity_gate": specificity,
        },
    }


@pytest.mark.parametrize(
    ("strata", "classification", "primary"),
    [
        (
            _strata(True, True),
            "affine_gauge_transport_sufficient",
            True,
        ),
        (
            _strata(True, False),
            "architecture_conditional_affine_gauge_transport",
            False,
        ),
        (
            _strata(False, False, d6_front=True, d10_front=True),
            "front_gauge_repaired_continuation_insufficient",
            False,
        ),
        (
            _strata(False, False, d6_front=True, d10_front=False),
            "support_relative_affine_gauge_insufficient",
            False,
        ),
        (
            _strata(True, True, specificity=False),
            "specificity_control_failed",
            False,
        ),
    ],
)
def test_locked_classification(
    strata: dict, classification: str, primary: bool
) -> None:
    measured, passed = classify_aggregates(strata, valid=True)
    assert measured == classification
    assert passed is primary
