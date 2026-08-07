from __future__ import annotations

import math

import pytest
import torch

from experiments.structure_net import tinyllm_local_continuation_tangent_kernel as audit


def test_primary_configuration_is_locked() -> None:
    config = audit.LocalContinuationConfig(
        device="cpu", post_outcome_corrective_replication=True
    )
    assert config.seeds == (7, 29, 53)
    assert config.orbit_count == 64
    assert config.random_controls == 8
    assert config.fourier_order == 4
    with pytest.raises(ValueError, match="fixed"):
        audit.LocalContinuationConfig(
            seeds=(7,), device="cpu", post_outcome_corrective_replication=True
        )
    with pytest.raises(ValueError, match="eight"):
        audit.LocalContinuationConfig(
            random_controls=4,
            device="cpu",
            post_outcome_corrective_replication=True,
        )
    with pytest.raises(ValueError, match="corrective"):
        audit.LocalContinuationConfig(device="cpu")
    pilot = audit.LocalContinuationConfig(
        seeds=(7,),
        orbit_count=8,
        random_controls=2,
        device="cpu",
        allow_underpowered=True,
    )
    assert audit._evidence_role(pilot) == (
        "systems_lifecycle_only_not_quality_evidence"
    )


def test_complex_posterior_moment_uses_bin_circle() -> None:
    posterior = torch.eye(4, dtype=torch.float64)
    moment = audit.complex_posterior_moment(posterior)
    expected = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=torch.float64,
    )
    assert torch.allclose(moment, expected, atol=1e-12)


def test_tangent_kernel_decomposition_recovers_row_and_null_spaces() -> None:
    jacobian = torch.tensor(
        [[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]], dtype=torch.float64
    )
    residual = torch.tensor([[1.0, -2.0, 4.0]], dtype=torch.float64)
    result = audit.tangent_kernel_decomposition(
        jacobian, residual, rtol=1e-6, atol=1e-10
    )
    assert torch.allclose(
        result["tangent"], torch.tensor([[1.0, -2.0, 0.0]], dtype=torch.float64)
    )
    assert torch.allclose(
        result["kernel"], torch.tensor([[0.0, 0.0, 4.0]], dtype=torch.float64)
    )
    assert int(result["rank"][0]) == 2
    assert float(result["relative_kernel_leakage"].max()) < 1e-12
    assert float(result["relative_tangent_mismatch"].max()) < 1e-12


def test_tangent_kernel_decomposition_handles_rank_one_batches() -> None:
    jacobian = torch.tensor(
        [
            [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
            [[0.0, 0.0, 4.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    residual = torch.tensor([[2.0, 0.0, 1.0], [1.0, 2.0, 3.0]], dtype=torch.float64)
    result = audit.tangent_kernel_decomposition(
        jacobian, residual, rtol=1e-6, atol=1e-10
    )
    assert result["rank"].tolist() == [1, 1]
    assert torch.allclose(
        torch.einsum("nij,nj->ni", jacobian, result["kernel"]),
        torch.zeros(2, 2, dtype=torch.float64),
        atol=1e-12,
    )


def test_random_controls_are_deterministic_and_norm_matched() -> None:
    component = torch.tensor(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [1.0, -2.0, 2.0]],
        dtype=torch.float64,
    )
    first = audit.norm_matched_random_controls(component, 8, seed=123)
    second = audit.norm_matched_random_controls(component, 8, seed=123)
    assert torch.equal(first, second)
    expected = torch.linalg.vector_norm(component, dim=-1)
    actual = torch.linalg.vector_norm(first, dim=-1)
    assert torch.allclose(actual, expected[None, :], atol=1e-12)
    assert torch.all(first[:, 1] == 0.0)


def test_normalized_task_metric_has_unit_trace_and_ordered_spectrum() -> None:
    jacobian = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    metric = audit.normalized_task_metric(jacobian)
    assert math.isclose(float(torch.trace(metric["metric"])), 1.0)
    assert metric["eigenvalues"].tolist() == pytest.approx([0.8, 0.2, 0.0])
    assert torch.allclose(
        metric["top_eigenvector"].abs(),
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
    )


def test_metric_stability_is_sign_invariant_for_top_direction() -> None:
    common = {
        "metric": torch.diag(torch.tensor([0.8, 0.2, 0.0], dtype=torch.float64)),
        "eigenvalues": torch.tensor([0.8, 0.2, 0.0], dtype=torch.float64),
        "trace": torch.tensor(1.0, dtype=torch.float64),
    }
    left = {**common, "top_eigenvector": torch.tensor([1.0, 0.0, 0.0])}
    right = {**common, "top_eigenvector": torch.tensor([-1.0, 0.0, 0.0])}
    result = audit.metric_stability([left, right])
    assert result["maximum_normalized_frobenius_distance"] == 0.0
    assert result["minimum_top_eigenvector_absolute_cosine"] == 1.0


@pytest.mark.parametrize(
    (
        "valid",
        "tangent",
        "kernel_inert",
        "specificity",
        "stable",
        "kernel_active",
        "full",
        "expected",
    ),
    [
        (False, True, True, True, True, False, True, "invalid"),
        (
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            "stable_task_tangent_sufficient",
        ),
        (
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            "checkpoint_local_task_tangent_sufficient",
        ),
        (
            True,
            False,
            False,
            True,
            False,
            True,
            True,
            "nominal_kernel_causally_active",
        ),
        (
            True,
            False,
            True,
            True,
            False,
            False,
            True,
            "nonlinear_tangent_kernel_interaction_required",
        ),
        (
            True,
            True,
            True,
            False,
            True,
            False,
            True,
            "mixed_local_continuation_geometry",
        ),
    ],
)
def test_classification_is_fixed(
    valid: bool,
    tangent: bool,
    kernel_inert: bool,
    specificity: bool,
    stable: bool,
    kernel_active: bool,
    full: bool,
    expected: str,
) -> None:
    assert (
        audit.classify_checkpoint(
            valid=valid,
            tangent_pass=tangent,
            kernel_inert=kernel_inert,
            tangent_specificity=specificity,
            kernel_specificity=specificity,
            metric_stable=stable,
            kernel_causally_active=kernel_active,
            full_pass=full,
        )
        == expected
    )


def test_random_seed_separates_cells_and_families() -> None:
    values = {
        audit._random_seed(7, cohort, regime, family)
        for cohort in ("heldout_a", "heldout_b")
        for regime in ("composition", "extrapolation")
        for family in ("tangent", "kernel")
    }
    assert len(values) == 8
