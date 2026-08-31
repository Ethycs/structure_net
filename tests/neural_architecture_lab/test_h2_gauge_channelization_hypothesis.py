from __future__ import annotations

import json

import pytest

from neural_architecture_lab.core import LabConfig
from neural_architecture_lab.h2_gauge_channelization_hypothesis import (
    HYPOTHESIS_ID,
    PREREGISTRATION_SHA256,
    REGISTRY_SCHEMA,
    build_hypothesis,
    register_hypothesis,
)
from neural_architecture_lab.lab import NeuralArchitectureLab


def test_builds_frozen_pending_a8() -> None:
    hypothesis = build_hypothesis()
    controls = hypothesis.control_parameters
    assert hypothesis.id == HYPOTHESIS_ID
    assert hypothesis.tested is False
    assert hypothesis.results == []
    assert controls["menu_id"] == "A8"
    assert controls["block_widths"] == [1, 2, 4]
    assert controls["optimizer"]["updates"] == 96
    assert controls["preregistration_sha256"] == PREREGISTRATION_SHA256
    with pytest.raises(RuntimeError, match="registered but not finalized"):
        hypothesis.test_function({})


def test_registers_and_reads_back_a8(tmp_path) -> None:
    results_dir = tmp_path / "nal"
    output = tmp_path / "registry.json"
    lab = NeuralArchitectureLab(
        LabConfig(
            project_name="test_h2_gauge_channelization",
            results_dir=str(results_dir),
            device_ids=[-1],
            max_parallel_experiments=1,
            enable_wandb=False,
            enable_adaptive_hypotheses=False,
            auto_balance=False,
            verbose=False,
        )
    )
    record = register_hypothesis(lab, output)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert record == persisted
    assert record["schema"] == REGISTRY_SCHEMA
    assert record["status"] == "registered_not_run"
    assert record["hypotheses"][0]["id"] == HYPOTHESIS_ID
    assert record["hypotheses"][0]["tested"] is False
    readback = lab.logger.hypotheses_collection.get(
        ids=[HYPOTHESIS_ID], include=["metadatas"]
    )
    assert readback["ids"] == [HYPOTHESIS_ID]
    assert readback["metadatas"][0]["tested"] is False
