from __future__ import annotations

import pytest
import torch

from experiments.structure_net import tinyllm_source_task_covector_portability as portability


def test_primary_configuration_and_fresh_split_are_locked() -> None:
    config = portability.SourceTaskCovectorPortabilityConfig(device="cpu")
    assert config.seeds == (7, 29, 53)
    assert config.orbit_count == 64
    assert config.carrier_rank == 3
    assert config.writer_order == 4
    assert config.fine_step_std == pytest.approx(0.025)
    assert config.coarse_step_std == pytest.approx(0.05)
    assert portability.FRESH_COHORT_SEEDS == {
        "composition": 430007,
        "extrapolation": 430008,
    }
    source_seeds = {
        value
        for cohort in portability.fixed.COHORT_SEEDS.values()
        for value in cohort.values()
    }
    assert not source_seeds.intersection(portability.FRESH_COHORT_SEEDS.values())
    with pytest.raises(ValueError, match="fixed"):
        portability.SourceTaskCovectorPortabilityConfig(seeds=(7,), device="cpu")
    with pytest.raises(ValueError, match="64 exact orbits"):
        portability.SourceTaskCovectorPortabilityConfig(
            seeds=(7,), orbit_count=8, device="cpu", allow_underpowered=True
        )
    pilot = portability.SourceTaskCovectorPortabilityConfig(
        seeds=(7,), device="cpu", allow_underpowered=True
    )
    assert portability._evidence_role(pilot) == (
        "systems_lifecycle_only_not_quality_evidence"
    )


def test_task_inverse_is_minimum_norm_signed_correction() -> None:
    gradient = torch.tensor(
        [[1.0, 2.0, -2.0], [3.0, 0.0, 4.0]], dtype=torch.float64
    )
    error = torch.tensor([4.5, -2.0], dtype=torch.float64)
    correction = portability.task_inverse_correction(
        gradient, error, denominator_floor=1e-12
    )
    assert torch.allclose((gradient * correction).sum(1), error, atol=1e-12)
    for row in range(2):
        assert torch.allclose(
            correction[row],
            gradient[row] * error[row] / gradient[row].square().sum(),
            atol=1e-12,
        )


def test_zero_covector_is_numerically_safe() -> None:
    correction = portability.task_inverse_correction(
        torch.zeros((5, 3), dtype=torch.float64),
        torch.ones(5, dtype=torch.float64),
        denominator_floor=1e-12,
    )
    assert torch.equal(correction, torch.zeros_like(correction))
    assert torch.isfinite(correction).all()


def test_source_maps_recover_exact_linear_laws() -> None:
    generator = torch.Generator().manual_seed(91)
    features = torch.randn((32, 9), generator=generator, dtype=torch.float64)
    features[:, -1] = 1.0
    gradient_weights = torch.randn((9, 3), generator=generator, dtype=torch.float64)
    error_weights = torch.randn((9, 1), generator=generator, dtype=torch.float64)
    gradient = features @ gradient_weights
    error = (features @ error_weights).reshape(-1)
    maps = portability.fit_source_maps(features, gradient, error, ridge=1e-12)
    predicted_gradient = portability.transport.apply_affine(
        features, maps["covector"]
    )
    predicted_error = portability.transport.apply_affine(
        features, maps["signed_error"]
    ).reshape(-1)
    assert torch.allclose(predicted_gradient, gradient, atol=1e-9)
    assert torch.allclose(predicted_error, error, atol=1e-9)


def test_random_direction_and_permutation_are_deterministic() -> None:
    source = torch.tensor(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float64
    )
    first = portability.norm_matched_random(source, 311)
    second = portability.norm_matched_random(source, 311)
    assert torch.equal(first, second)
    assert torch.allclose(
        torch.linalg.vector_norm(first, dim=1),
        torch.linalg.vector_norm(source, dim=1),
        atol=1e-12,
    )
    first_permutation = portability.fixed_permutation(64, 77, torch.device("cpu"))
    second_permutation = portability.fixed_permutation(64, 77, torch.device("cpu"))
    assert torch.equal(first_permutation, second_permutation)
    assert torch.equal(torch.sort(first_permutation).values, torch.arange(64))


def test_regression_metrics_preserve_signed_and_vector_diagnostics() -> None:
    target_vector = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64
    )
    vector = portability.regression_metrics(target_vector, target_vector)
    assert vector["zero_referenced_r2"] == pytest.approx(1.0)
    assert vector["mean_row_cosine"] == pytest.approx(1.0)
    scalar_target = torch.tensor([1.0, -2.0, 0.001], dtype=torch.float64)
    scalar_prediction = torch.tensor([0.5, -1.0, -0.001], dtype=torch.float64)
    scalar = portability.regression_metrics(
        scalar_prediction, scalar_target, sign_floor=0.01
    )
    assert scalar["sign_evaluation_count"] == 2
    assert scalar["sign_agreement"] == pytest.approx(1.0)


def _continuous(passed: bool, mean: float) -> dict[str, object]:
    return {"continuous_pass": passed, "mean_moment_shift_bins": mean}


def _cell(
    *, primary_pass: bool = True, control_pass: bool = False, control_mean: float = 0.50
) -> dict[str, object]:
    return {
        "states": {
            "order4": {"continuous": _continuous(False, 0.60)},
            **{
                name: {"continuous": _continuous(primary_pass, 0.05)}
                for name in portability.PRIMARY_NAMES
            },
            **{
                name: {"continuous": _continuous(control_pass, control_mean)}
                for name in portability.CONTROL_NAMES
            },
        }
    }


def test_state_gate_requires_both_fresh_cells_and_specific_controls() -> None:
    summary = portability.state_gate_summary([_cell(), _cell()], 0.125)
    assert summary["all_primary_pass"] is True
    assert summary["all_controls_specific"] is True
    assert all(value["specific"] for value in summary["controls"].values())

    cells = [_cell(), _cell()]
    cells[0]["states"]["source_predicted"]["continuous"] = _continuous(False, 0.20)
    primary_failure = portability.state_gate_summary(cells, 0.125)
    assert primary_failure["primary_all_fresh_cells_pass"]["source_predicted"] is False

    low_margin = portability.state_gate_summary(
        [_cell(control_mean=0.10), _cell(control_mean=0.10)], 0.125
    )
    assert low_margin["all_controls_specific"] is False


def _state_gates(
    *,
    local: bool = True,
    covector: bool = True,
    scalar: bool = True,
    source: bool = True,
    specific: bool = True,
) -> dict[str, object]:
    primary = {
        "local_oracle": local,
        "source_covector_oracle_error": covector,
        "local_covector_source_error": scalar,
        "source_predicted": source,
    }
    return {
        "primary_all_fresh_cells_pass": primary,
        "all_primary_pass": all(primary.values()),
        "all_controls_specific": specific,
    }


@pytest.mark.parametrize(
    ("contracts", "gates", "expected", "passed"),
    [
        (False, _state_gates(), "invalid", False),
        (True, _state_gates(local=False), "local_tangent_not_replicated", False),
        (True, _state_gates(), "portable_source_covector_and_scalar", True),
        (True, _state_gates(specific=False), "nonspecific_source_correction", False),
        (
            True,
            _state_gates(source=False),
            "source_components_portable_joint_not",
            False,
        ),
        (
            True,
            _state_gates(scalar=False, source=False),
            "source_covector_portable_scalar_not",
            False,
        ),
        (
            True,
            _state_gates(covector=False, source=False),
            "source_scalar_portable_covector_not",
            False,
        ),
        (
            True,
            _state_gates(covector=False, scalar=False, source=False),
            "local_oracle_only",
            False,
        ),
    ],
)
def test_checkpoint_classification_is_preregistered(
    contracts: bool,
    gates: dict[str, object],
    expected: str,
    passed: bool,
) -> None:
    assert portability.classify_checkpoint(
        contracts_valid=contracts, state_gates=gates
    ) == (expected, passed)


def test_campaign_decision_requires_three_complete_checkpoint_gates() -> None:
    supported = portability._campaign_decision(
        ["portable_source_covector_and_scalar"] * 3, 3, False
    )
    assert supported["supported"] is True
    partial = portability._campaign_decision(
        ["portable_source_covector_and_scalar"] * 2 + ["local_oracle_only"],
        2,
        False,
    )
    assert partial["supported"] is False
    assert partial["conclusion"] == "checkpoint_stratified_source_portability"
    systems = portability._campaign_decision(
        ["portable_source_covector_and_scalar"], 1, True
    )
    assert systems["conclusion"] == "systems_lifecycle_only_not_quality_evidence"


def test_predecessor_and_known_local_campaign_are_hash_locked() -> None:
    config = portability.SourceTaskCovectorPortabilityConfig(device="cpu")
    campaign, path, details = portability.local._load_predecessor(config)
    assert portability._sha256(path) == portability.PREDECESSOR_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == (
        portability.PREDECESSOR_IMPLEMENTATION_SHA256
    )
    assert set(details) == {7, 29, 53}
    tangent, tangent_path = portability._load_local_tangent_provenance(config)
    assert portability._sha256(tangent_path) == portability.LOCAL_TANGENT_CAMPAIGN_SHA256
    assert tangent["hypothesis_id"] == portability.local.HYPOTHESIS_ID
