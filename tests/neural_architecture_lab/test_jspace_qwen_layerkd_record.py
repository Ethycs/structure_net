"""Conservative result indexing must reject unfair or incomplete comparisons."""
from copy import deepcopy

import pytest

from neural_architecture_lab.jspace_qwen_layerkd_record import (
    ARMS, CLASSIFICATION, JSpaceQwenLayerKDRecord,
)


def complete_aggregate():
    result = {"study_id": "JSPACE-QWEN-LAYERKD-01", "status": "completed", "confirmed": False,
              "fresh_evaluation_count": 512, "fingerprint": "f" * 64, "results": {}}
    training, reload = {}, {}
    for arm, correct in (("native", 387), ("endpoint", 385), ("euclidean", 388), ("jacobian", 385)):
        accuracy = {"correct": correct, "count": 512, "overall": correct / 512}
        result["results"][arm] = {"accuracy": accuracy, "tensor_bytes": 6000,
                                  "runtime": {"timings": {"cuda_graph": {"tokens_per_second": 40.}}}}
        if arm in ARMS:
            training[arm] = {"initialized_checkpoint_sha256": "same-initializer", "completed_epochs": 12,
                             "updates": 1536, "selected_epoch": 7, "tensor_bytes": 6000}
            reload[arm] = {"records_exact": True, "accuracy": deepcopy(accuracy)}
    return {"classification": CLASSIFICATION, "result": result, "training": training, "reload": reload}


def test_index_retains_all_arms_without_promoting_best_arm_to_confirmation():
    record = JSpaceQwenLayerKDRecord.from_aggregate(complete_aggregate())
    assert record.correct_by_arm == {"endpoint": 385, "euclidean": 388, "jacobian": 385}
    assert record.trained_arms == 3 and record.independent_confirmations == 0
    assert record.confirmed is False and record.classification == CLASSIFICATION
    assert "single_final_block" in record.scope and "no_staged_joint_recovery" in record.scope


@pytest.mark.parametrize("mutation", ["confirmed", "budget", "initializer", "footprint", "reload", "accuracy", "selection", "missing_arm"])
def test_invalid_comparison_cannot_be_registered(mutation):
    aggregate = complete_aggregate()
    if mutation == "confirmed":
        aggregate["result"]["confirmed"] = True
    elif mutation == "budget":
        aggregate["training"]["euclidean"]["updates"] = 1535
    elif mutation == "initializer":
        aggregate["training"]["jacobian"]["initialized_checkpoint_sha256"] = "different"
    elif mutation == "footprint":
        aggregate["training"]["euclidean"]["tensor_bytes"] = 5900
        aggregate["result"]["results"]["euclidean"]["tensor_bytes"] = 5900
    elif mutation == "reload":
        aggregate["reload"]["jacobian"]["records_exact"] = False
    elif mutation == "accuracy":
        aggregate["result"]["results"]["endpoint"]["accuracy"]["overall"] = .9
    elif mutation == "selection":
        aggregate["training"]["jacobian"]["selected_epoch"] = 13
    else:
        del aggregate["training"]["endpoint"]
    with pytest.raises(ValueError):
        JSpaceQwenLayerKDRecord.from_aggregate(aggregate)
