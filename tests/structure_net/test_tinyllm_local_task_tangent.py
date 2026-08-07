from __future__ import annotations

import pytest
import torch

from experiments.structure_net import tinyllm_local_task_tangent as tangent


def test_primary_configuration_and_geometry_are_locked() -> None:
    config = tangent.LocalTaskTangentConfig(device="cpu")
    assert config.seeds == (7, 29, 53)
    assert config.orbit_count == 64
    assert config.carrier_rank == 3
    assert config.writer_order == 4
    assert config.fine_step_std == pytest.approx(0.025)
    assert config.coarse_step_std == pytest.approx(0.05)
    with pytest.raises(ValueError, match="fixed"):
        tangent.LocalTaskTangentConfig(seeds=(7,), device="cpu")
    with pytest.raises(ValueError, match="64 exact orbits"):
        tangent.LocalTaskTangentConfig(
            seeds=(7,), orbit_count=8, device="cpu", allow_underpowered=True
        )
    pilot = tangent.LocalTaskTangentConfig(
        seeds=(7,), device="cpu", allow_underpowered=True
    )
    assert tangent._evidence_role(pilot) == (
        "systems_lifecycle_only_not_quality_evidence"
    )


def test_tangent_kernel_decomposition_reconstructs_and_is_orthogonal() -> None:
    residual = torch.tensor(
        [[2.0, -1.0, 4.0], [0.5, 3.0, -2.0]], dtype=torch.float64
    )
    gradient = torch.tensor(
        [[1.0, 2.0, -1.0], [-2.0, 1.0, 0.5]], dtype=torch.float64
    )
    scale = torch.tensor([2.0, 0.5, 4.0], dtype=torch.float64)
    parts = tangent.tangent_kernel_decomposition(
        residual, gradient, scale, denominator_floor=1e-12
    )
    assert torch.allclose(
        parts["tangent_raw"] + parts["kernel_raw"], residual, atol=1e-12
    )
    assert torch.allclose(
        (parts["tangent_std"] * parts["kernel_std"]).sum(1),
        torch.zeros(2, dtype=torch.float64),
        atol=1e-12,
    )
    diagnostics = tangent.decomposition_diagnostics(parts)
    assert diagnostics["reconstruction_relative_error"] < 1e-12
    assert diagnostics["maximum_tangent_kernel_absolute_cosine"] < 1e-12


def test_zero_gradient_places_the_entire_residual_in_the_kernel() -> None:
    residual = torch.randn(8, 3, dtype=torch.float64)
    parts = tangent.tangent_kernel_decomposition(
        residual,
        torch.zeros_like(residual),
        torch.ones(3, dtype=torch.float64),
        denominator_floor=1e-12,
    )
    assert torch.all(parts["tangent_raw"] == 0.0)
    assert torch.allclose(parts["kernel_raw"], residual)


def test_random_direction_is_deterministic_and_norm_matched() -> None:
    source = torch.tensor(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    first = tangent.norm_matched_random(source, seed=311)
    second = tangent.norm_matched_random(source, seed=311)
    different = tangent.norm_matched_random(source, seed=312)
    assert torch.equal(first, second)
    assert not torch.equal(first[:2], different[:2])
    assert torch.allclose(
        torch.linalg.vector_norm(first, dim=1),
        torch.linalg.vector_norm(source, dim=1),
        atol=1e-12,
    )


def test_fixed_permutation_is_deterministic_and_complete() -> None:
    first = tangent.fixed_permutation(64, 77, torch.device("cpu"))
    second = tangent.fixed_permutation(64, 77, torch.device("cpu"))
    assert torch.equal(first, second)
    assert torch.equal(torch.sort(first).values, torch.arange(64))


def _continuous(passed: bool, mean: float) -> dict[str, object]:
    return {"continuous_pass": passed, "mean_moment_shift_bins": mean}


def _cell(
    *, tangent_pass: bool = True, control_pass: bool = False
) -> dict[str, object]:
    return {
        "states": {
            "order4": {"continuous": _continuous(False, 0.50)},
            "tangent_only": {
                "continuous": _continuous(tangent_pass, 0.05)
            },
            **{
                name: {"continuous": _continuous(control_pass, 0.50)}
                for name in tangent.CONTROL_NAMES
            },
        }
    }


def test_state_gate_requires_all_cells_and_each_specific_control() -> None:
    summary = tangent.state_gate_summary([_cell() for _ in range(4)], 0.125)
    assert summary["tangent_all_cells_pass"] is True
    assert summary["all_controls_specific"] is True
    assert all(value["specific"] for value in summary["controls"].values())

    cells = [_cell() for _ in range(4)]
    cells[0]["states"]["kernel_only"]["continuous"] = _continuous(True, 0.06)
    for cell in cells[1:]:
        cell["states"]["kernel_only"]["continuous"] = _continuous(False, 0.06)
    failed = tangent.state_gate_summary(cells, 0.125)
    assert failed["controls"]["kernel_only"]["any_failure"] is True
    assert failed["controls"]["kernel_only"]["specific"] is False
    assert failed["all_controls_specific"] is False


def _state_gates(
    *, tangent_pass: bool, specific: bool, order4: float = 0.50, tangent_mean: float = 0.05
) -> dict[str, object]:
    return {
        "tangent_all_cells_pass": tangent_pass,
        "all_controls_specific": specific,
        "aggregate_mean_shift_bins": {
            "order4": order4,
            "tangent_only": tangent_mean,
        },
    }


@pytest.mark.parametrize(
    ("contracts", "local", "gates", "expected", "passed"),
    [
        (
            False,
            True,
            _state_gates(tangent_pass=True, specific=True),
            "invalid",
            False,
        ),
        (
            True,
            True,
            _state_gates(tangent_pass=True, specific=True),
            "local_task_tangent_sufficient",
            True,
        ),
        (
            True,
            True,
            _state_gates(tangent_pass=True, specific=False),
            "nonunique_or_curved_correction",
            False,
        ),
        (
            True,
            True,
            _state_gates(
                tangent_pass=False, specific=False, order4=0.50, tangent_mean=0.20
            ),
            "tangent_helpful_not_sufficient",
            False,
        ),
        (
            True,
            True,
            _state_gates(
                tangent_pass=False, specific=False, order4=0.30, tangent_mean=0.20
            ),
            "task_tangent_insufficient",
            False,
        ),
        (
            True,
            False,
            _state_gates(tangent_pass=True, specific=True),
            "nonlinear_at_writer_residual_scale",
            False,
        ),
    ],
)
def test_checkpoint_classification_is_preregistered(
    contracts: bool,
    local: bool,
    gates: dict[str, object],
    expected: str,
    passed: bool,
) -> None:
    assert tangent.classify_checkpoint(
        contracts_valid=contracts,
        local_adequate=local,
        state_gates=gates,
        helpful_improvement_bins=0.125,
    ) == (expected, passed)


def test_campaign_decision_requires_three_complete_checkpoint_gates() -> None:
    supported = tangent._campaign_decision(
        ["local_task_tangent_sufficient"] * 3, 3, False
    )
    assert supported["supported"] is True
    partial = tangent._campaign_decision(
        ["local_task_tangent_sufficient"] * 2 + ["task_tangent_insufficient"],
        2,
        False,
    )
    assert partial["supported"] is False
    assert partial["conclusion"] == "checkpoint_stratified_local_geometry"
    systems = tangent._campaign_decision(
        ["local_task_tangent_sufficient"], 1, True
    )
    assert systems["conclusion"] == "systems_lifecycle_only_not_quality_evidence"


def test_predecessor_campaign_and_results_are_hash_locked() -> None:
    campaign, path, details = tangent._load_predecessor(
        tangent.LocalTaskTangentConfig(device="cpu")
    )
    assert tangent._sha256(path) == tangent.PREDECESSOR_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == (
        tangent.PREDECESSOR_IMPLEMENTATION_SHA256
    )
    assert set(details) == {7, 29, 53}
    for detail, result_path in details.values():
        matching = next(
            item
            for item in campaign["results"]
            if int(item["seed"]) == int(detail["seed"])
        )
        assert tangent._sha256(result_path) == matching["result_sha256"]
