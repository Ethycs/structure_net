from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.structure_net import (
    tinyllm_c3_relational_connection_readout_audit as audit,
)


RESULT = Path(
    "data/experiments/tinyllm_c3_relational_connection_readout_audit/"
    "20260811_artifact_audit/result.json"
)


def test_corrective_audit_sources_and_scope_are_sealed() -> None:
    sources, campaign, details = audit.validate_sources()
    assert sources["primary_runner"] == audit.PRIMARY_RUNNER_SHA256
    assert campaign["aggregates"]["classification"] == audit.PRIMARY_CLASSIFICATION
    assert campaign["aggregates"]["primary_hypothesis_pass"] is False
    assert tuple(details) == audit.study.SEEDS


def test_linear_fit_and_application_are_exact_on_affine_data() -> None:
    feature = torch.linspace(-1.0, 1.0, 64, dtype=torch.float64)[:, None]
    target = 2.5 * feature[:, 0] - 0.125
    fit = audit.fit_linear(feature, target)
    prediction = audit.apply_linear(feature, fit["coefficients"])
    assert fit["training_rank"] == 2
    assert torch.allclose(prediction, target.double(), atol=1e-12, rtol=0.0)


def test_stored_audit_is_valid_and_cannot_rescue_primary() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    aggregates = result["aggregates"]
    assert aggregates["valid"] is True
    assert aggregates["classification"] == (
        "posthoc_public_scale_readout_reaches_four_of_five_one_wrong_winding_remains"
    )
    assert aggregates["joint_pass_counts"]["scalar_affine"] == {
        "learned_connection_shuffled": 0,
        "learned_no_connection": 0,
        "learned_target_shuffled": 0,
        "learned_true": 4,
    }
    assert aggregates["newly_repaired_true_seed_counts"]["scalar_affine"] == 3
    assert aggregates["persistent_true_failure_seeds"]["scalar_affine"] == [1453]
    assert aggregates["primary_classification_unchanged"] is True
    assert aggregates["further_optimizer_tuning_licensed"] is False
    assert aggregates["unrestricted_tinyllm_training_licensed"] is False


def test_all_primary_predictions_replay_on_registered_devices() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert len(result["cells"]) == 5
    assert all(cell["pass"] is True for cell in result["cells"])
    assert all(
        arm["replay_pass"] is True
        for cell in result["cells"]
        for arm in cell["arms"].values()
    )
