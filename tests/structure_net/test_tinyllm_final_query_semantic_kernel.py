from __future__ import annotations

import torch

import experiments.structure_net.tinyllm_final_query_semantic_kernel as study
from experiments.structure_net.tinyllm_predictive_circle import (
    CircleTaskConfig,
    _tinyllm_config,
)
from structure_net.components.models import TinyLLMModel


def _small_config(**changes: object) -> study.SemanticKernelConfig:
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
    return study.SemanticKernelConfig(**values)


def test_primary_configuration_and_preregistration_are_locked() -> None:
    config = study.SemanticKernelConfig()
    assert config.regimes == study.REGIMES
    assert config.fiber_pairs == 512
    assert config.semantic_mean_change_ceiling == 0.01
    assert config.semantic_p95_change_ceiling == 0.03
    assert config.kernel_contraction_ratio_ceiling == 0.25
    assert config.posterior_kernel_ratio_floor == 0.75
    assert study._sha256(study.PREREGISTRATION_PATH) == (
        study.PREREGISTRATION_SHA256
    )


def test_frozen_parent_and_posterior_null_sources_are_locked() -> None:
    task, hashes, parent, posterior = study._validate_sources(_small_config())
    assert task.phase_bins == 16
    assert hashes["posterior_campaign"] == study.POSTERIOR_CAMPAIGN_SHA256
    assert hashes["checkpoint_d8_cosine_interval"] == study.CHECKPOINT_SHA256
    assert parent["gates"]["validity"] is True
    assert posterior["gates"] == {
        "primary_hypothesis_pass": False,
        "state_unchanged": True,
        "validity": True,
    }
    assert posterior["classification"] == (
        "task_rowspace_contains_material_fiber_chord"
    )


def test_analytic_semantic_jacobian_matches_autograd() -> None:
    task = CircleTaskConfig()
    model = TinyLLMModel(_tinyllm_config("tiny", task, seed=37)).eval()
    answer_ids = torch.tensor(task.answer_token_ids[:6], dtype=torch.long)
    centers = torch.linspace(-1.0, 1.0, len(answer_ids), dtype=torch.float64)
    query = torch.randn((3, model.config.n_embd), dtype=torch.float64)
    analytic, _, _ = study.semantic_coordinate_jacobian(
        model, query, answer_ids, centers
    )
    expected = []
    for row in query:
        value = row.detach().clone().requires_grad_(True)
        gradient = torch.autograd.functional.jacobian(
            lambda item: study.scalar_coordinate_double(
                model, item[None], answer_ids, centers
            )[0],
            value,
        )
        expected.append(gradient)
    assert torch.max(
        torch.abs(analytic[:, 0] - torch.stack(expected))
    ).item() <= 1e-10


def test_semantic_jacobian_is_nested_in_centered_logit_rowspace() -> None:
    task = CircleTaskConfig()
    model = TinyLLMModel(_tinyllm_config("tiny", task, seed=41)).eval()
    answer_ids = torch.tensor(task.answer_token_ids[:6], dtype=torch.long)
    centers = torch.linspace(-1.0, 1.0, len(answer_ids), dtype=torch.float64)
    query = torch.randn((4, model.config.n_embd), dtype=torch.float64)
    scalar, logits, _ = study.semantic_coordinate_jacobian(
        model, query, answer_ids, centers
    )
    parts = study.posterior.decompose_displacement(
        logits, scalar[:, 0], rtol=1e-6, atol=1e-10
    )
    leakage = torch.linalg.vector_norm(parts["kernel"], dim=1) / (
        torch.linalg.vector_norm(scalar[:, 0], dim=1).clamp_min(1e-12)
    )
    assert torch.equal(parts["rank"], torch.full((4,), 5))
    assert torch.max(leakage).item() <= 1e-10


def test_scalar_preservation_uses_coordinate_and_target_metrics() -> None:
    baseline = torch.linspace(-0.8, 0.8, 32)
    candidate = baseline + 0.005
    baseline_metrics = {"task_map_score": 0.95, "root_mean_squared_error": 0.10}
    candidate_metrics = {"task_map_score": 0.945, "root_mean_squared_error": 0.105}
    result = study.scalar_preservation(
        baseline,
        candidate,
        baseline_metrics,
        candidate_metrics,
        _small_config(),
    )
    assert result["pass"] is True
    assert result["mean_absolute_coordinate_change"] < 0.01


def test_scalar_effect_attribution_uses_actual_nonlinear_coordinate() -> None:
    baseline = torch.zeros(64, dtype=torch.float64)
    full = torch.linspace(-1.0, 1.0, 64, dtype=torch.float64)
    semantic = 0.90 * full
    kernel = 0.10 * full
    result, arrays = study.scalar_effect_attribution(
        baseline, full, semantic, kernel, _small_config()
    )
    assert result["pass"] is True
    assert result["material_row_count"] == 64
    assert result["semantic_full_sign_agreement"] == 1.0
    assert len(arrays["relative_semantic_residual"]) == 64


def test_classification_preserves_scalar_kernel_failure_mode() -> None:
    regimes = {
        regime: {
            "gates": {
                "semantic_kernel_scalar_preservation": True,
                "semantic_kernel_pair_contraction": False,
                "complete_posterior_separation": True,
                "random_projector_specificity": True,
                "semantic_effect_attribution": True,
            }
        }
        for regime in study.REGIMES
    }
    assert study._classification(
        study.SemanticKernelConfig(), True, False, regimes
    ) == "scalar_kernel_preserves_but_does_not_contain_fiber_chord"
