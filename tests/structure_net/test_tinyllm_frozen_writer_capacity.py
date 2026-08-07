from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import tinyllm_frozen_writer_capacity as capacity


def test_primary_configuration_is_locked() -> None:
    config = capacity.FrozenWriterCapacityConfig(device="cpu")
    assert config.seeds == (7, 29, 53)
    assert config.orbit_count == 64
    assert config.low_orders == (1, 2, 3, 4)
    assert config.matched_orders == (6, 10, 14, 18)
    assert config.context_rank == 3
    with pytest.raises(ValueError, match="fixed"):
        capacity.FrozenWriterCapacityConfig(seeds=(7,), device="cpu")
    pilot = capacity.FrozenWriterCapacityConfig(
        seeds=(7,), orbit_count=8, device="cpu", allow_underpowered=True
    )
    assert capacity._evidence_role(pilot) == (
        "systems_lifecycle_only_not_quality_evidence"
    )


def test_fourier_ladder_is_nested_and_has_explicit_constant() -> None:
    theta = torch.tensor([0.2, 1.1], dtype=torch.float64)
    order1 = capacity.fourier_features(theta, 1)
    order4 = capacity.fourier_features(theta, 4)
    order18 = capacity.fourier_features(theta, 18)
    assert order1.shape == (2, 3)
    assert order4.shape == (2, 9)
    assert order18.shape == (2, 37)
    assert torch.all(order1[:, 0] == 1.0)
    assert torch.allclose(order1, order4[:, :3])
    assert torch.allclose(order4, order18[:, :9])


def test_context_features_have_preregistered_width_and_nested_prefix() -> None:
    theta = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    context = torch.randn(5, 3, dtype=torch.float64)
    order1 = capacity.context_features(theta, context, 1)
    order4 = capacity.context_features(theta, context, 4)
    assert order1.shape == (5, 12)
    assert order4.shape == (5, 36)
    assert torch.allclose(order1, order4[:, :12])
    assert torch.allclose(order1[:, :4], torch.cat((torch.ones(5, 1), context), 1))


def test_ridge_writer_recovers_a_linear_target() -> None:
    features = torch.randn(80, 13, dtype=torch.float64)
    expected = torch.randn(13, 3, dtype=torch.float64)
    target = features @ expected
    mapping = capacity.fit_ridge_writer(features, target, 1e-10)
    predicted = features @ mapping["linear"] + mapping["intercept"]
    assert torch.allclose(predicted, target, atol=1e-8, rtol=1e-8)


def test_invariant_context_fit_is_orthonormal_and_source_standardized() -> None:
    propagated = torch.randn(64, 2, 5, dtype=torch.float64)
    model = capacity.fit_invariant_context(propagated, rank=3, scale_floor=1e-10)
    value = capacity.apply_invariant_context(propagated, model)
    assert value.shape == (64, 3)
    assert torch.allclose(
        value.mean(0), torch.zeros(3, dtype=torch.float64), atol=1e-12
    )
    assert torch.allclose(
        value.std(0, unbiased=False),
        torch.ones(3, dtype=torch.float64),
        atol=1e-12,
    )
    assert torch.allclose(
        model["basis"] @ model["basis"].T,
        torch.eye(3, dtype=torch.float64),
        atol=1e-12,
    )


def test_context_basis_signs_are_canonical() -> None:
    basis = torch.tensor(
        [[-3.0, 1.0, 0.0], [0.5, -4.0, 1.0]], dtype=torch.float64
    )
    value = capacity._canonicalize_basis_signs(basis)
    for row in value:
        pivot = int(torch.argmax(row.abs()))
        assert float(row[pivot]) > 0.0


def test_embedded_mapping_uses_only_declared_feature_block() -> None:
    mapping = {
        "linear": torch.randn(12, 3, dtype=torch.float64),
        "intercept": torch.zeros(3, dtype=torch.float64),
    }
    embedded = capacity._embed_mapping(mapping, offset=37, total_features=73)
    assert torch.all(embedded["linear"][:37] == 0.0)
    assert torch.allclose(embedded["linear"][37:49], mapping["linear"])
    assert torch.all(embedded["linear"][49:] == 0.0)


def _continuous(passed: bool, mean: float, p95: float = 0.2) -> dict[str, object]:
    return {
        "continuous_pass": passed,
        "mean_moment_shift_bins": mean,
        "p95_moment_shift_bins": p95,
    }


def test_writer_summary_requires_causal_and_shuffled_specificity() -> None:
    heldout = [
        {
            "states": {
                "candidate": {"continuous": _continuous(True, 0.05)},
                "candidate_shuffled": {"continuous": _continuous(False, 1.0)},
            }
        }
        for _ in range(4)
    ]
    summary = capacity._writer_summary(
        heldout, "candidate", "candidate_shuffled", 0.125
    )
    assert summary["checkpoint_pass"] is True
    assert summary["specificity"] is True
    assert summary["complete_pass"] is True


def _summaries(
    *,
    passing: str | None = None,
    context_mean: float = 0.5,
    baseline_mean: float = 0.5,
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for order in capacity.ALL_FOURIER_ORDERS:
        name = f"fourier_m{order:02d}"
        output[name] = {
            "complete_pass": name == passing,
            "aggregate_mean_shift_bins": baseline_mean,
        }
    for order in capacity.LOW_ORDERS:
        name = f"context_m{order:02d}"
        output[name] = {
            "complete_pass": name == passing,
            "aggregate_mean_shift_bins": context_mean,
        }
    return output


@pytest.mark.parametrize(
    ("valid", "summaries", "expected", "selected"),
    [
        (False, _summaries(), "invalid", None),
        (
            True,
            _summaries(passing="fourier_m02"),
            "low_order_curvature_sufficient",
            "fourier_m02",
        ),
        (
            True,
            _summaries(passing="fourier_m10"),
            "high_order_quotient_capacity_sufficient",
            "fourier_m10",
        ),
        (
            True,
            _summaries(passing="context_m03"),
            "invariant_context_required",
            "context_m03",
        ),
        (
            True,
            _summaries(context_mean=0.30, baseline_mean=0.50),
            "context_helpful_not_decisive",
            "context_m01",
        ),
        (True, _summaries(), "small_writer_insufficient", None),
    ],
)
def test_checkpoint_classification_is_preregistered(
    valid: bool,
    summaries: dict[str, dict[str, object]],
    expected: str,
    selected: str | None,
) -> None:
    assert capacity.classify_checkpoint(
        valid=valid,
        summaries=summaries,
        specificity_margin_bins=0.125,
    ) == (expected, selected)


def test_predecessor_campaign_is_hash_locked() -> None:
    campaign, path, details = capacity._load_predecessor(
        capacity.FrozenWriterCapacityConfig(device="cpu")
    )
    assert capacity._sha256(path) == capacity.PREDECESSOR_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == (
        capacity.PREDECESSOR_IMPLEMENTATION_SHA256
    )
    assert set(details) == {7, 29, 53}
