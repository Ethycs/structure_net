"""Tests for the pending A5--A7 NAL registrations."""

from __future__ import annotations

import json

import pytest

from neural_architecture_lab.core import LabConfig
from neural_architecture_lab.dynamic_ttno_followup_hypotheses import (
    A5_HYPOTHESIS_ID,
    A6_HYPOTHESIS_ID,
    A7_HYPOTHESIS_ID,
    REGISTRY_SCHEMA,
    get_dynamic_ttno_followup_hypotheses,
    register_dynamic_ttno_followups,
)
from neural_architecture_lab.lab import NeuralArchitectureLab


EXPECTED_IDS = [A5_HYPOTHESIS_ID, A6_HYPOTHESIS_ID, A7_HYPOTHESIS_ID]


def test_builds_frozen_pending_a5_through_a7() -> None:
    hypotheses = get_dynamic_ttno_followup_hypotheses()

    assert [item.id for item in hypotheses] == EXPECTED_IDS
    assert [item.control_parameters["menu_id"] for item in hypotheses] == [
        "A5",
        "A6",
        "A7",
    ]
    assert all(item.tested is False and item.results == [] for item in hypotheses)
    assert all(
        item.parameter_space["evaluation_seed"] == [101, 211, 307, 401, 503]
        for item in hypotheses
    )
    assert all(item.control_parameters["frozen_cell_count"] == 120 for item in hypotheses)
    assert all(item.control_parameters["lengths"] == [32, 64, 128, 256] for item in hypotheses)
    assert all(
        len(item.control_parameters["preregistration_sha256"]) == 64
        for item in hypotheses
    )

    for hypothesis in hypotheses:
        with pytest.raises(
            RuntimeError, match="registered but has not been implemented or run"
        ):
            hypothesis.test_function({})

    a7 = hypotheses[2].control_parameters
    assert a7["primary_lengths"] == [64, 128, 256]
    assert a7["extension_requires_new_fingerprinted_checkpoint"] is True
    assert a7["probe"]["seed"] == 1707
    assert a7["partition_integrity"]["256"] == {
        "admissible": 33,
        "dense": 31,
        "zero": 15,
        "sha256": "1535f675b18b8493d5da1c4e1048eefb73c884e700953451de7f9b8b380688f2",
    }


def test_registers_persists_and_reads_back(tmp_path) -> None:
    results_dir = tmp_path / "nal"
    output_path = tmp_path / "registry.json"
    lab = NeuralArchitectureLab(
        LabConfig(
            project_name="test_dynamic_ttno_followup_registration",
            results_dir=str(results_dir),
            device_ids=[-1],
            max_parallel_experiments=1,
            enable_wandb=False,
            enable_adaptive_hypotheses=False,
            auto_balance=False,
            verbose=False,
        )
    )

    record = register_dynamic_ttno_followups(lab, output_path)
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert record["schema"] == REGISTRY_SCHEMA
    assert persisted == record
    assert persisted["status"] == "registered_not_run"
    assert [item["id"] for item in persisted["hypotheses"]] == EXPECTED_IDS
    assert all(item["tested"] is False for item in persisted["hypotheses"])
    assert lab.pending_hypotheses == EXPECTED_IDS
    assert record["storage"]["readback"] == {
        "verified": True,
        "hypothesis_ids": EXPECTED_IDS,
        "tested": False,
    }

    chroma_readback = lab.logger.hypotheses_collection.get(
        ids=EXPECTED_IDS, include=["metadatas"]
    )
    assert set(chroma_readback["ids"]) == set(EXPECTED_IDS)
    assert all(item["tested"] is False for item in chroma_readback["metadatas"])
