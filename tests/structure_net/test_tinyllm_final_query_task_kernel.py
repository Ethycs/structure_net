from __future__ import annotations

import torch

import experiments.structure_net.tinyllm_final_query_task_kernel as study
from experiments.structure_net.tinyllm_predictive_circle import (
    CircleTaskConfig,
    _tinyllm_config,
)
from structure_net.components.models import TinyLLMModel


def _small_config(**changes: object) -> study.FinalQueryKernelConfig:
    values: dict[str, object] = {
        "regimes": ("training_support",),
        "fiber_pairs": 8,
        "batch_size": 16,
        "jacobian_batch_pairs": 4,
        "finite_difference_pairs": 4,
        "device": "cpu",
        "allow_underpowered": True,
    }
    values.update(changes)
    return study.FinalQueryKernelConfig(**values)


def test_primary_configuration_and_preregistration_are_locked() -> None:
    config = study.FinalQueryKernelConfig()
    assert config.regimes == study.REGIMES
    assert config.fiber_pairs == 512
    assert config.kernel_contraction_ratio_ceiling == 0.25
    assert config.random_js_disadvantage_floor == 0.01
    assert study._sha256(study.PREREGISTRATION_PATH) == (
        study.PREREGISTRATION_SHA256
    )


def test_parent_valid_d8_null_and_sources_are_locked() -> None:
    task, hashes, detail = study._validate_parent(_small_config())
    assert task.phase_bins == 16
    assert hashes["parent_campaign"] == study.PARENT_CAMPAIGN_SHA256
    assert hashes["checkpoint_d8_cosine_interval"] == study.CHECKPOINT_SHA256
    assert detail["gates"]["validity"] is True
    assert detail["summary"]["mature_causal_front"] is None


def test_analytic_centered_logit_jacobian_matches_autograd() -> None:
    task = CircleTaskConfig()
    model = TinyLLMModel(_tinyllm_config("tiny", task, seed=23)).eval()
    answer_ids = torch.tensor(task.answer_token_ids[:5], dtype=torch.long)
    query = torch.randn((2, model.config.n_embd), dtype=torch.float64)
    analytic = study.centered_logit_jacobian(model, query, answer_ids)
    expected = []
    for row in query:
        value = row.detach().clone().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(
            lambda item: study.centered_logits_double(
                model, item[None], answer_ids
            )[0],
            value,
        )
        expected.append(jacobian)
    expected_tensor = torch.stack(expected)
    assert torch.max(torch.abs(analytic - expected_tensor)).item() <= 1e-10


def test_task_kernel_decomposition_reconstructs_and_annihilates() -> None:
    generator = torch.Generator().manual_seed(29)
    independent = torch.randn(
        (6, 4, 12), generator=generator, dtype=torch.float64
    )
    jacobian = torch.cat(
        (independent, -independent.sum(1, keepdim=True)), dim=1
    )
    displacement = torch.randn(
        (6, 12), generator=generator, dtype=torch.float64
    )
    parts = study.decompose_displacement(
        jacobian, displacement, rtol=1e-6, atol=1e-10
    )
    assert torch.equal(parts["rank"], torch.full((6,), 4))
    assert torch.max(parts["reconstruction_error"]).item() <= 1e-12
    assert torch.max(parts["kernel_leakage"]).item() <= 1e-10
    inner = (parts["task"] * parts["kernel"]).sum(1).abs()
    assert torch.max(inner).item() <= 1e-10


def test_random_rowspace_is_deterministic_and_rank_matched() -> None:
    first = study.random_rowspace_basis(24, 5, 31, device=torch.device("cpu"))
    second = study.random_rowspace_basis(24, 5, 31, device=torch.device("cpu"))
    assert torch.equal(first, second)
    assert torch.max(torch.abs(first.T @ first - torch.eye(5))).item() <= 1e-12
    displacement = torch.randn((7, 24), dtype=torch.float64)
    kernel = study.random_kernel_displacement(displacement, first)
    assert torch.max(torch.abs(kernel @ first)).item() <= 1e-12


def test_effect_attribution_uses_actual_nonlinear_changes() -> None:
    baseline = torch.zeros((16, 5), dtype=torch.float64)
    full = torch.arange(80, dtype=torch.float64).reshape(16, 5)
    full = full - full.mean(1, keepdim=True)
    task_only = 0.90 * full
    kernel_only = 0.10 * full
    result, arrays = study._effect_attribution(
        baseline, full, task_only, kernel_only, _small_config()
    )
    assert result["pass"] is True
    assert result["mean_relative_task_residual"] == torch.mean(
        arrays["relative_task_residual"]
    ).item()
    assert result["median_task_full_cosine"] >= 0.999999
