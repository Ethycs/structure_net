from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net import tinyllm_local_continuation_tangent as local


def _config() -> local.LocalContinuationTangentConfig:
    return local.LocalContinuationTangentConfig(
        seeds=(7,), device="cpu", allow_underpowered=True
    )


def test_primary_configuration_is_locked() -> None:
    config = local.LocalContinuationTangentConfig(device="cpu")
    assert config.seeds == (7, 29, 53)
    assert config.orbit_count == 64
    assert config.carrier_rank == 3
    assert config.writer_name == "context_m04"
    assert config.fine_step_std == 0.025
    assert config.coarse_step_std == 0.05
    with pytest.raises(ValueError, match="context_m04"):
        replace(config, writer_name="context_m03")
    with pytest.raises(ValueError, match="64 exact orbits"):
        replace(config, orbit_count=8, allow_underpowered=True)


def test_task_tangent_decomposition_reconstructs_and_nulls_gradient() -> None:
    residual = torch.tensor(
        [[3.0, 4.0, 5.0], [-2.0, 1.0, 7.0]], dtype=torch.float64
    )
    gradient = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]], dtype=torch.float64
    )
    result = local.decompose_task_tangent(residual, gradient, 1e-12)
    tangent = result["tangent"]
    kernel = result["kernel"]
    assert isinstance(tangent, torch.Tensor)
    assert isinstance(kernel, torch.Tensor)
    torch.testing.assert_close(tangent + kernel, residual, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        (gradient * kernel).sum(1), torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0
    )
    assert result["nondegenerate"] is True
    assert result["reconstruction_relative_error"] <= 1e-12
    assert result["kernel_orthogonality_relative_error"] <= 1e-12


def test_task_tangent_marks_degenerate_gradient() -> None:
    result = local.decompose_task_tangent(
        torch.ones((2, 3), dtype=torch.float64),
        torch.zeros((2, 3), dtype=torch.float64),
        1e-8,
    )
    assert result["nondegenerate"] is False


def test_matched_random_directions_are_deterministic_and_norm_matched() -> None:
    norms = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64)
    first = local.matched_random_directions(norms, 3, 1234, torch.device("cpu"))
    second = local.matched_random_directions(norms, 3, 1234, torch.device("cpu"))
    different = local.matched_random_directions(norms, 3, 1235, torch.device("cpu"))
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(first, dim=1), norms, atol=1e-12, rtol=0
    )
    assert not torch.equal(first[1:], different[1:])


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            dict(
                valid=False,
                local_adequate=True,
                tangent_checkpoint_pass=True,
                tangent_specificity=True,
                kernel_change_fraction=0.5,
            ),
            "invalid",
        ),
        (
            dict(
                valid=True,
                local_adequate=True,
                tangent_checkpoint_pass=True,
                tangent_specificity=True,
                kernel_change_fraction=0.5,
            ),
            "local_task_tangent_sufficient",
        ),
        (
            dict(
                valid=True,
                local_adequate=False,
                tangent_checkpoint_pass=False,
                tangent_specificity=False,
                kernel_change_fraction=0.5,
            ),
            "local_linearization_inadequate",
        ),
        (
            dict(
                valid=True,
                local_adequate=True,
                tangent_checkpoint_pass=False,
                tangent_specificity=False,
                kernel_change_fraction=0.10,
            ),
            "nominal_kernel_causally_active",
        ),
        (
            dict(
                valid=True,
                local_adequate=True,
                tangent_checkpoint_pass=False,
                tangent_specificity=False,
                kernel_change_fraction=0.099,
            ),
            "tangent_kernel_interaction_or_endpoint_curvature",
        ),
    ],
)
def test_checkpoint_classification_precedence(
    arguments: dict[str, object], expected: str
) -> None:
    assert (
        local.classify_checkpoint(
            **arguments, material_kernel_fraction=0.10  # type: ignore[arg-type]
        )
        == expected
    )


def test_fingerprint_changes_with_outcome_relevant_inputs() -> None:
    config = _config()
    first = local._fingerprint(config, 7, "a", "b", "c")
    assert first == local._fingerprint(config, 7, "a", "b", "c")
    assert first != local._fingerprint(config, 7, "a", "b", "d")
    assert first != local._fingerprint(replace(config, fine_step_std=0.02), 7, "a", "b", "c")


def test_authoritative_predecessor_identity_and_arm() -> None:
    campaign, path, details = local._load_predecessor(_config())
    assert local._sha256(path) == local.PREDECESSOR_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == local.PREDECESSOR_IMPLEMENTATION_SHA256
    detail, detail_path = details[7]
    assert detail_path.is_file()
    assert detail["classification"] == "small_writer_insufficient"
    assert local.WRITER_NAME in detail["alignment_fit"]["mappings"]


def test_completed_campaign_reuse_requires_result_hash(tmp_path: Path) -> None:
    config = _config()
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"status": "completed"}) + "\n")
    implementation = "implementation"
    campaign = {
        "schema_version": local.SCHEMA_VERSION,
        "hypothesis_id": local.HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": local._evidence_role(config),
        "implementation_sha256": implementation,
        "configuration": local._json_compatible(config.__dict__),
        "summary": {"completed": 1},
        "results": [{"path": str(result), "result_sha256": local._sha256(result)}],
    }
    assert local._campaign_is_reusable(campaign, config, implementation)
    campaign["results"][0]["result_sha256"] = "wrong"
    assert not local._campaign_is_reusable(campaign, config, implementation)
