from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch

from experiments.structure_net import (
    tinyllm_c3_full_local_gauge_closure as study,
)


PRIMARY_RESULT = Path(
    "data/experiments/tinyllm_c3_full_local_gauge_closure/"
    "20260811_artifact_audit/result.json"
)
PRIMARY_RESULT_SHA256 = (
    "31f7c7301c889db67e436fba4b8de1909dfbc372573681a9950b5b330f22db35"
)


def _pilot_tokens() -> tuple[torch.Tensor, torch.Tensor]:
    dataset = study.source_study.generate_dataset(
        "composition",
        study.source_study.PILOT_SEED,
        sample_count=64,
        allow_pilot=True,
    )
    donor, frame = study.source_study.corruption_plan(
        "composition", study.source_study.PILOT_SEED, 64
    )
    return (
        study.base.corruption.corrupt_frames(dataset.tokens, donor, frame),
        dataset.calibration,
    )


def test_sources_and_algebraic_scope_are_frozen() -> None:
    hashes, result = study.validate_sources()
    assert hashes == {
        "preregistration": study.PREREGISTRATION_SHA256,
        "joint_typed_score_preregistration": (
            study.SOURCE_PREREGISTRATION_SHA256
        ),
        "joint_typed_score_runner": study.SOURCE_RUNNER_SHA256,
        "joint_typed_score_result": study.SOURCE_RESULT_SHA256,
        "joint_typed_score_report": study.SOURCE_REPORT_SHA256,
    }
    assert result["aggregates"]["invariant_fixed_seed_pass_count"] == 5
    assert len(study.GENERATOR_ACTIONS) == 16
    assert study.GROUP_ORDER == 3**8 == 6561


def test_local_action_group_law_and_pointwise_cube_are_exact() -> None:
    tokens, _ = _pilot_tokens()
    first = study.action_stream("composition", 1171, 64)
    second = study.action_stream("composition", 1171, 64, second=True)
    contracts = study.token_group_contracts(tokens, first, second)
    assert contracts["pass"] is True
    assert sum(
        value
        for key, value in contracts.items()
        if key.endswith("error_count")
    ) == 0
    reference = study.pointwise_cube_pairs(tokens)
    transformed = study.pointwise_cube_pairs(
        study.apply_local_action(tokens, first)
    )
    assert torch.equal(reference[0], transformed[0])
    assert torch.equal(reference[1], transformed[1])


def test_generator_action_preserves_invariant_decoder() -> None:
    tokens, calibration = _pilot_tokens()
    base_cube = study.pointwise_cube_pairs(tokens)
    invariant, prediction, selection, _ = study.invariant_decoder(
        tokens, calibration
    )
    for time, element in study.GENERATOR_ACTIONS:
        action = torch.zeros((64, study.TIME_STEPS), dtype=torch.int64)
        action[:, time] = element
        measurement = study._action_measurement(
            tokens,
            calibration,
            action,
            base_cube,
            invariant,
            prediction,
            selection,
        )
        assert measurement["pass"] is True
        assert measurement["exact_cube_integer_error_count"] == 0
        assert measurement["maximum_cubic_carrier_error"] <= 2e-12
        assert measurement["maximum_invariant_forecast_error"] <= 2e-12


def test_lifecycle_exhausts_all_local_action_vectors() -> None:
    contract = study.exhaustive_pilot_cube_contract(sample_count=2)
    assert contract == {
        "sample_count": 2,
        "action_count": 6561,
        "expected_group_order": 6561,
        "exact_cube_integer_error_count": 0,
        "pass": True,
    }


def _classification_cells(
    *, inherited: int, exact: int, numeric: int
) -> list[dict]:
    cells = []
    for index, seed in enumerate(study.SEEDS):
        for regime in study.REGIMES:
            cells.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "source_fixed_invariant_pass": index < inherited,
                    "exact_algebraic_closure": index < exact,
                    "numeric_local_gauge_closure": index < numeric,
                }
            )
    return cells


def test_classifications_close_training_in_every_valid_row() -> None:
    closed = study.classify(
        _classification_cells(inherited=5, exact=5, numeric=5), True
    )
    assert closed["classification"] == (
        "pointwise_cubic_quotient_closes_full_local_c3_gauge"
    )
    assert closed["multiple_jump_experiment_licensed"] is False
    assert closed["compact_connection_model_licensed"] is False
    assert closed["tinyllm_training_licensed"] is False

    numeric = study.classify(
        _classification_cells(inherited=5, exact=5, numeric=3), True
    )
    assert numeric["classification"] == (
        "algebraic_local_gauge_closure_numeric_implementation_defect"
    )
    insufficient = study.classify(
        _classification_cells(inherited=3, exact=5, numeric=5), True
    )
    assert insufficient["classification"] == (
        "local_gauge_invariant_but_task_decoder_insufficient"
    )
    assert study.classify(
        _classification_cells(inherited=5, exact=5, numeric=5), False
    )["classification"] == "invalid_full_local_c3_gauge_audit"


def test_main_writes_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "status": "completed",
        "aggregates": {
            "classification": "test_classification",
            "inherited_fixed_invariant_seed_pass_count": 5,
            "numeric_local_gauge_seed_pass_count": 5,
            "certified_local_group_order": 6561,
            "tinyllm_training_licensed": False,
        },
    }
    monkeypatch.setattr(study, "build_result", lambda: expected)
    output = tmp_path / "result.json"
    assert study.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected


def test_authoritative_result_certifies_full_local_group_and_stop() -> None:
    assert hashlib.sha256(PRIMARY_RESULT.read_bytes()).hexdigest() == (
        PRIMARY_RESULT_SHA256
    )
    result = json.loads(PRIMARY_RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["schema_version"] == study.SCHEMA_VERSION
    assert result["hypothesis_id"] == study.HYPOTHESIS_ID
    assert result["source_hashes"] == {
        "joint_typed_score_preregistration": (
            study.SOURCE_PREREGISTRATION_SHA256
        ),
        "joint_typed_score_report": study.SOURCE_REPORT_SHA256,
        "joint_typed_score_result": study.SOURCE_RESULT_SHA256,
        "joint_typed_score_runner": study.SOURCE_RUNNER_SHA256,
        "preregistration": study.PREREGISTRATION_SHA256,
        "runner": (
            "dae960759a2412451f8b15e15b0b6fb479603938a323ea54c86c77c68839005d"
        ),
    }
    assert result["aggregates"] == {
        "certified_local_group_order": 6561,
        "classification": (
            "pointwise_cubic_quotient_closes_full_local_c3_gauge"
        ),
        "compact_connection_model_licensed": False,
        "exact_local_gauge_seed_pass_count": 5,
        "generator_action_count": 16,
        "inherited_fixed_invariant_seed_pass_count": 5,
        "multiple_jump_experiment_licensed": False,
        "numeric_local_gauge_seed_pass_count": 5,
        "required_seed_passes": 4,
        "tinyllm_training_licensed": False,
        "valid": True,
    }
    assert result["accounting"] == {
        "arbitrary_local_counterfactual_evaluations": 40_960,
        "checkpoints_loaded": 0,
        "fresh_examples": 0,
        "generator_counterfactual_evaluations": 655_360,
        "models_instantiated": 0,
        "optimizer_steps": 0,
        "parameters_changed": 0,
        "reusable_parameter_fits": 0,
        "source_examples_reused": 40_960,
        "target_using_fits": 0,
    }
    assert len(result["cells"]) == 10
    assert all(
        cell["valid"] is True
        and cell["source_hash_match"] is True
        and cell["identity_metric_replay"] is True
        and cell["exact_algebraic_closure"] is True
        and cell["numeric_local_gauge_closure"] is True
        for cell in result["cells"]
    )
    assert sum(cell["total_selector_changes"] for cell in result["cells"]) == 42
    assert max(
        cell["maximum_cubic_carrier_error"] for cell in result["cells"]
    ) <= 2e-12
    assert max(
        cell["maximum_invariant_forecast_error"] for cell in result["cells"]
    ) <= 2e-12
    assert sum(
        measurement["exact_cube_integer_error_count"]
        for cell in result["cells"]
        for measurement in (
            *cell["generator_measurements"].values(),
            cell["arbitrary_local_action"],
        )
    ) == 0
