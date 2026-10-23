import json

import pytest
import torch

from experiments.neural_architecture_lab.finalize_ncm_training_diagnosis import Evidence
from experiments.structure_net.ncm_memory_data import file_hash
from experiments.structure_net.diagnose_ncm_training import (
    code_metrics, reconstruction_counts, released_penalty_tokens)


def test_reconstruction_separates_padding_and_eos():
    target = torch.tensor([[3, 4, 50256, 50257, 50257]])
    predicted = torch.tensor([[3, 9, 50256, 50257, 50257]])
    assert reconstruction_counts(predicted, target) == {
        "content": {"correct": 1, "total": 2},
        "nonpad": {"correct": 2, "total": 3},
        "canvas": {"correct": 4, "total": 5}}


def test_per_level_code_metrics_preserve_denominators():
    labels = torch.tensor([[0, 1, 2]])
    logits = torch.tensor([[[5., 0., 0.], [5., 0., 0.], [0., 0., 5.]]])
    result = code_metrics(logits, labels, [1, 2])
    assert result["correct"] == 2 and result["total"] == 3
    assert [(r["correct"], r["total"]) for r in result["per_level"]] == [(1, 1), (1, 2)]
    assert abs(result["ce"] - (result["per_level"][0]["ce"] +
                                2*result["per_level"][1]["ce"])/3) < 1e-6


def test_released_penalties_discourage_immediate_repetition():
    logits = torch.tensor([[[3., 2.], [3., 2.], [3., 2.]]])
    assert logits.argmax(-1).tolist() == [[0, 0, 0]]
    assert released_penalty_tokens(logits, torch.tensor([.5, .5])).tolist() == [[0, 1, 0]]


def test_registry_record_preserves_exploratory_boundary_and_checks_evidence(tmp_path):
    evidence = tmp_path / "result.json"
    evidence.write_text('{"trained":false}')
    campaign = tmp_path / "campaign_results.json"
    campaign.write_text(json.dumps({"classification": "exploratory_code_generation_failure_diagnosed",
        "artifacts": {str(evidence): file_hash(evidence)}}))
    report = tmp_path / "report.md"
    report.write_text("Forward-only diagnosis; no new task accuracy.")
    record = Evidence(tmp_path, report).record()
    assert record["tested"] and not record["confirmed"]
    assert record["new_training_runs"] == 0 and not record["task_accuracy_measured"]
    assert record["report_sha256"] == file_hash(report)
    evidence.write_text('{"trained":true}')
    with pytest.raises(ValueError, match="evidence changed"):
        Evidence(tmp_path, report).record()
