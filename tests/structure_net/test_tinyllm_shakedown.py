"""End-to-end test for the TinyLLM feedback shakedown experiment."""

import json

import pytest
import torch

from experiments.structure_net.tinyllm_feedback_shakedown import (
    ShakedownConfig,
    main,
    run_shakedown,
)
from structure_net.components.models import TinyLLMModel


def test_shakedown_is_reproducible_matched_and_restorable(tmp_path):
    config = ShakedownConfig(
        training_steps=2,
        batch_size=3,
        sequence_length=6,
        vocab_size=24,
        tiny_layers=2,
        tiny_heads=2,
        tiny_width=16,
        feedback_connections=1,
        feedback_neurons=2,
        seed=4,
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    original_threads = torch.get_num_threads()
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_rng_state = torch.random.get_rng_state()
    bundle = run_shakedown(config, first_dir)
    repeated = run_shakedown(config, second_dir)

    assert torch.get_num_threads() == original_threads
    assert torch.are_deterministic_algorithms_enabled() == original_deterministic
    assert torch.equal(torch.random.get_rng_state(), original_rng_state)

    assert bundle["claim_scope"] == "systems_lifecycle_only_not_quality_evidence"
    assert bundle == repeated
    assert (first_dir / "results.json").read_bytes() == (
        second_dir / "results.json"
    ).read_bytes()
    assert bundle["comparisons"]["all_arms_share_pre_intervention_loss"]
    assert bundle["comparisons"]["recompute_matches_baseline_initial"]
    assert bundle["comparisons"]["recompute_matches_baseline_final"]
    assert bundle["comparisons"]["all_checkpoint_round_trips_exact"]
    assert [result["arm"] for result in bundle["results"]] == [
        "baseline",
        "recompute_control",
        "random_feedback",
    ]
    stored = json.loads((first_dir / "results.json").read_text())
    assert stored["schema_version"] == "nal.tinyllm-feedback-shakedown.v1"
    feedback_result = next(
        result for result in bundle["results"] if result["arm"] == "random_feedback"
    )
    assert feedback_result["growth_plan"]["type"] == "add_random_feedback_connections"
    assert feedback_result["growth_result"]["optimizer_parameters_added"] > 0

    for result in bundle["results"]:
        restored = TinyLLMModel.from_checkpoint(
            first_dir / result["model_checkpoint"]
        )
        assert restored.get_feedback_topology() == result["model_architecture"][
            "feedback_connections"
        ]

    feedback = TinyLLMModel.from_checkpoint(first_dir / "random_feedback.pt")
    assert len(feedback.feedback_connections) == 1
    assert feedback.refinement_steps == 1
    assert len(bundle["unsupported_boundaries"]) == 5


def test_shakedown_accepts_cuda_and_rejects_other_accelerators():
    assert ShakedownConfig(device="cuda").device == "cuda"
    assert ShakedownConfig(device="cuda:2").device == "cuda:2"
    with pytest.raises(ValueError, match="supports CPU or CUDA"):
        ShakedownConfig(device="mps")


def test_shakedown_reports_unavailable_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cannot access a CUDA driver"):
        run_shakedown(ShakedownConfig(device="cuda"), tmp_path / "cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_shakedown_cuda_lifecycle(tmp_path):
    bundle = run_shakedown(
        ShakedownConfig(
            device="cuda",
            training_steps=1,
            batch_size=2,
            sequence_length=4,
            vocab_size=16,
            tiny_layers=2,
            tiny_heads=2,
            tiny_width=16,
            feedback_connections=1,
            feedback_neurons=2,
        ),
        tmp_path / "cuda",
    )
    assert bundle["comparisons"]["all_arms_share_pre_intervention_loss"]
    assert bundle["comparisons"]["all_checkpoint_round_trips_exact"]


def test_documented_shakedown_cli_arguments(tmp_path):
    output = tmp_path / "cli"
    assert main(
        [
            "--preset",
            "tiny",
            "--steps",
            "1",
            "--seed",
            "9",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    ) == 0
    assert (output / "results.json").is_file()
