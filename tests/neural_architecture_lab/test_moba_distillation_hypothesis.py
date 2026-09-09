"""Conservative confirmation and exact storage read-back contracts."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from neural_architecture_lab.moba_distillation_hypothesis import (
    HYPOTHESIS_ID, MoBADistillationEvidence, build_hypothesis, store_evidence,
)
from experiments.structure_net.tinyllm_moba_distillation import sha256


@pytest.fixture
def evidence(tmp_path):
    campaign = tmp_path / "campaign.json"
    report = tmp_path / "report.md"
    campaign.write_text('{"classification":"distillation_not_confirmed"}')
    report.write_text("A completed negative result.\n")
    return MoBADistillationEvidence("distillation_not_confirmed", 40, 0, str(campaign), sha256(campaign),
                                    str(report), sha256(report), tuple(f"cell-{i}" for i in range(40)))


class FakeCollection:
    def __init__(self):
        self.stored = None
        self.corrupt = False

    def get(self, ids, include):
        assert ids == [HYPOTHESIS_ID]
        metadata = dict(self.stored)
        if self.corrupt:
            metadata["confirmation_status"] = "unverified"
        return {"ids": ids, "metadatas": [metadata]}


class FakeLogger:
    def __init__(self):
        self.hypotheses_collection = FakeCollection()

    def log_hypothesis(self, record):
        self.hypotheses_collection.stored = record


def test_negative_and_partial_results_cannot_be_upgraded(evidence):
    assert build_hypothesis(evidence).tested
    assert build_hypothesis(evidence).control_parameters["confirmed"] is False
    assert replace(evidence, classification="behavioral_gain_without_full_joint_gate").confirmed is False
    with pytest.raises(ValueError, match="full joint gate"):
        replace(evidence, classification="routing_distillation_supported", joint_passes=3)
    assert replace(evidence, classification="routing_distillation_supported", joint_passes=4).confirmed is True
    with pytest.raises(ValueError):
        replace(evidence, classification="systems_lifecycle_only_not_quality_evidence")
    with pytest.raises(ValueError):
        replace(evidence, completed_cells=39)


def test_exact_storage_and_idempotent_readback(evidence, tmp_path):
    logger = FakeLogger()
    path = tmp_path / "registry.json"
    record = store_evidence(evidence, logger, path)
    assert record["hypotheses"][0]["id"] == HYPOTHESIS_ID
    assert record["hypotheses"][0]["confirmed"] is False
    assert record["storage"]["readback"]["verified"] is True
    assert json.loads(path.read_text()) == store_evidence(evidence, logger, path)


def test_hash_mismatch_or_bad_readback_prevents_registry(evidence, tmp_path):
    logger = FakeLogger()
    path = tmp_path / "registry.json"
    logger.hypotheses_collection.corrupt = True
    with pytest.raises(RuntimeError, match="read-back"):
        store_evidence(evidence, logger, path)
    assert not path.exists()
    Path(evidence.report_path).write_text("changed interpretation")
    with pytest.raises(RuntimeError, match="changed"):
        store_evidence(evidence, logger, path)
    assert not path.exists()


def test_different_existing_registry_blocks_search_mutation(evidence, tmp_path):
    logger = FakeLogger()
    path = tmp_path / "registry.json"
    path.write_text('{"prior_record":true}')
    with pytest.raises(RuntimeError, match="preserve existing"):
        store_evidence(evidence, logger, path)
    assert logger.hypotheses_collection.stored is None
    assert json.loads(path.read_text()) == {"prior_record": True}
