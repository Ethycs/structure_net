from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net import tinyllm_c3_temporal_sensor_mechanism as study


@pytest.fixture(scope="module")
def primary_result() -> dict:
    if study.PRIMARY_RESULT_PATH.is_file():
        return json.loads(study.PRIMARY_RESULT_PATH.read_text(encoding="utf-8"))
    if torch.cuda.is_available():
        return study.build_result("cuda:0")
    pytest.skip("primary result absent and source-matched CUDA replay unavailable")


def test_frozen_source_campaign_and_preregistration_validate() -> None:
    campaign, details = study.validate_source_campaign()
    assert campaign["aggregates"]["classification"] == (
        "task_only_sensor_acquisition_supported"
    )
    assert set(details) == set(study.SEEDS)
    assert study._sha256(study.PREREGISTRATION_PATH) == (
        study.PREREGISTRATION_SHA256
    )


def test_complex_affine_fit_recovers_exact_response() -> None:
    scalar = torch.linspace(-3.0, 3.0, 1024)
    slope = torch.tensor(1.25 - 0.75j, dtype=torch.complex128)
    intercept = torch.tensor(-0.4 + 0.2j, dtype=torch.complex128)
    response = slope * scalar.double() + intercept
    fit = study.fit_complex_affine(scalar, response)
    assert fit["slope"] == pytest.approx(slope.item(), abs=1e-12)
    assert fit["intercept"] == pytest.approx(intercept.item(), abs=1e-12)
    assert fit["source_complex_r2"] == pytest.approx(1.0, abs=1e-12)
    assert fit["source_response_rmse"] <= 1e-12


def test_locked_population_classifications() -> None:
    def cells(
        true_affine: int,
        true_residual: int,
        shuffled_affine: int,
    ) -> list[dict]:
        values = []
        for index, seed in enumerate(study.SEEDS):
            values.append(
                {
                    "seed": seed,
                    "source_arm": "learned_true",
                    "full_replay_pass": True,
                    "response_seed_gates": {
                        "full_replay": True,
                        "affine_only": index < true_affine,
                        "nonlinear_residual_only": index < true_residual,
                    },
                }
            )
            values.append(
                {
                    "seed": seed,
                    "source_arm": "learned_target_shuffled",
                    "full_replay_pass": True,
                    "response_seed_gates": {
                        "full_replay": True,
                        "affine_only": index < shuffled_affine,
                        "nonlinear_residual_only": False,
                    },
                }
            )
        return values

    assert study.classify(cells(4, 1, 1), True)["classification"] == (
        "affine_identity_character_carries_learned_solution"
    )
    assert study.classify(cells(3, 4, 0), True)["classification"] == (
        "nonlinear_shared_response_required"
    )
    assert study.classify(cells(4, 4, 0), True)["classification"] == (
        "affine_and_nonlinear_paths_redundant"
    )
    assert study.classify(cells(5, 0, 2), True)["classification"] == (
        "affine_mechanism_specificity_failed"
    )
    assert study.classify(cells(5, 0, 0), False)["classification"] == (
        "invalid_source_contract"
    )


def test_primary_ten_checkpoint_replay_is_valid(primary_result: dict) -> None:
    assert primary_result["status"] == "completed"
    assert len(primary_result["cells"]) == 10
    aggregate = primary_result["aggregates"]
    assert aggregate["classification"] == (
        "affine_identity_character_carries_learned_solution"
    )
    assert aggregate["primary_hypothesis_pass"] is True
    assert aggregate["full_replay_pass_count"] == 10
    assert aggregate["true_affine_only_pass_count"] == 5
    assert aggregate["true_nonlinear_residual_only_pass_count"] == 0
    assert aggregate["shuffled_affine_only_pass_count"] == 0
    assert primary_result["accounting"] == {
        "checkpoints_loaded": 10,
        "optimizer_steps": 0,
        "parameters_changed": 0,
        "tinyllm_models_instantiated": 0,
        "target_using_fits": 0,
        "target_free_complex_affine_fits": 10,
    }
    for cell in primary_result["cells"]:
        assert cell["valid"] is True
        assert cell["state_unchanged"] is True
        assert cell["coefficient_reconstruction_pass"] is True
        for regime in study.REGIMES:
            measured = cell["regimes"][regime]
            assert measured["direct_encoder_posterior_maximum_error"] <= 2e-6
            assert measured["stored_metric_replay"]["pass"] is True
            for arm in study.RESPONSE_ARMS:
                response = measured["responses"][arm]
                assert set(response["deck_action_maximum_errors"]) == {"1", "2"}
                assert response["maximum_deck_action_error"] >= 0.0
                assert isinstance(response["action_pass"], bool)


def test_main_writes_strict_json_without_changing_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "status": "completed",
        "aggregates": {
            "classification": "test_classification",
            "true_affine_only_pass_count": 4,
            "true_nonlinear_residual_only_pass_count": 0,
            "shuffled_affine_only_pass_count": 0,
        },
    }
    monkeypatch.setattr(study, "build_result", lambda _device: expected)
    output = tmp_path / "result.json"
    assert study.main(["--output", str(output)]) == 0
    restored = json.loads(output.read_text(encoding="utf-8"))
    assert restored == expected
