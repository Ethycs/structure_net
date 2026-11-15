"""Focused policy boundary checks; the frozen training implementation is reused."""
import pytest

from experiments.structure_net import jspace_qwen3b_native_budget_train as study


def test_valid_preserved_seed_passes_native_but_reports_parent_failure():
    value = study.gates(47, native_correct=48, parent_correct=52, count=64)
    assert value["passed"] and value["minimum_correct"] == 45
    assert value["native"] == {"correct": 48, "minimum_correct": 45, "passed": True}
    assert value["parent"] == {"correct": 52, "minimum_correct": 49, "passed": False}
    assert value["policy"] == "original_native_only" and value["parent_gate_required"] is False


@pytest.mark.parametrize("correct,passed", [(44, False), (45, True), (48, True), (64, True)])
def test_native_budget_boundary_does_not_ratchet_with_immediate_parent(correct, passed):
    for parent in (45, 47, 52, 64):
        value = study.gates(correct, native_correct=48, parent_correct=parent, count=64)
        assert value["passed"] is passed and value["minimum_correct"] == 45
        assert value["parent"]["passed"] is (correct >= value["parent"]["minimum_correct"])


@pytest.mark.parametrize("changes", [
    {"correct": True}, {"correct": 47.0}, {"correct": -1}, {"correct": 65},
    {"native_correct": False}, {"parent_correct": 65}, {"count": 0},
    {"count": 64.0}, {"maximum_drop": .06},
])
def test_frozen_input_and_budget_validation_preserved(changes):
    args = dict(correct=47, native_correct=48, parent_correct=52, count=64)
    args.update(changes)
    with pytest.raises(ValueError):
        study.gates(**args)


def test_trainer_dispatch_changes_only_acceptance_not_training_or_selection():
    # Avoid constructing a model/optimizer: the override only reads these
    # already-validated fields. Actual fit dispatch is inherited unchanged.
    trainer = object.__new__(study.NativeBudgetTrainer)
    trainer.native_correct, trainer.parent_correct, trainer.development_count = 48, 52, 64
    assert trainer.quality(47) == study.gates(47, native_correct=48, parent_correct=52, count=64)
    assert trainer.quality(44)["passed"] is False
    assert study.CONFIG is study.prior.CONFIG
    for method in ("__init__", "fit", "master_update_probe", "packed_development", "train_window", "logits", "snapshot"):
        assert getattr(study.NativeBudgetTrainer, method) is getattr(study.prior.ProgressiveTrainer, method)
