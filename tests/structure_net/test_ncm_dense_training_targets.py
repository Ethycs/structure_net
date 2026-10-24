from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest
import torch

from experiments.structure_net import ncm_dense_training_targets as run
from experiments.neural_architecture_lab import finalize_ncm_dense_training_targets as audit


def training_rows():
    return [{"id": str(i), "split": "train", "word_ids": [7, 50256] + [50257] * 190,
        **({"training_origin": "reviewed_draft"} if i >= 252 else {})} for i in range(690)]


def test_complete_train_binding_preserves_all_cases_targets_and_fixed_strata():
    rows = training_rows()
    result = run.bind_rows(rows)
    assert [r["id"] for r in result] == [r["id"] for r in rows]
    assert sum(r["diagnostic_group"] == "original_252" for r in result) == 252
    assert sum(r["diagnostic_group"] == "reviewed_438" for r in result) == 438
    assert "diagnostic_group" not in rows[0]
    for mutation in (lambda r: r.pop(), lambda r: r[0].update(split="development"),
            lambda r: r[0]["word_ids"].__setitem__(1, 9),
            lambda r: r[0]["word_ids"].__setitem__(3, 8),
            lambda r: r[0].update(training_origin="moved"),
            lambda r: r[1].update(id=r[0]["id"])):
        changed = deepcopy(rows)
        mutation(changed)
        with pytest.raises(ValueError):
            run.bind_rows(changed)


def test_fresh_pilot_excludes_all_reused_targets_and_uses_metadata_only():
    rows = run.bind_rows(training_rows())
    previous = {str(i): {} for i in range(32)}
    rows[90]["word_ids"] = [7] * 190 + [50256, 50257]
    assert [r["id"] for r in run.probes(rows, previous)] == ["32", "90"]
    rows[90]["target_text"] = "bad label text cannot alter pilot selection"
    assert [r["id"] for r in run.probes(rows, previous)] == ["32", "90"]


def components(a=199, b=199):
    return {key: {"cases": n, "correct": correct, "total": total, "exact_target_texts": 0}
        for key, n, correct, total in zip(run.STRATA, (252, 438, 690), (a, b, a + b), (200, 200, 400))}


def test_threshold_is_inclusive_in_both_populations_and_union():
    assert run.fidelity_gate(components())
    assert not run.fidelity_gate(components(198, 200))
    broken = components()
    broken["all_690"]["correct"] += 1
    with pytest.raises(ValueError, match="compose"):
        run.fidelity_gate(broken)
    broken = components()
    broken["reviewed_438"]["cases"] -= 1
    with pytest.raises(ValueError, match="complete strata"):
        run.fidelity_gate(broken)


def test_aggregate_keeps_failed_labels_in_the_complete_denominator():
    values = [{"diagnostic_group": run.STRATA[i >= 252],
        "arms": {run.prior.ARMS[1]: {"correct": 2 if i else 0, "total": 2, "exact_target_text": i != 0}}}
        for i in range(690)]
    result = run.aggregate(values)
    assert result["all_690"]["cases"] == 690
    assert result["all_690"]["correct"] == 1378
    assert result["all_690"]["total"] == 1380
    assert result["original_252"]["exact_target_texts"] == 251


def test_reuse_requires_exact_optimizer_latents_traces_and_readouts():
    row = {"id": "a", "diagnostic_group": "original_252"}
    old = {"id": "a", "diagnostic_group": "original_16", "trace": [{"step": 256}],
        "arms": {"refined_encoder": {"correct": 2, "total": 2}}}
    saved = {"initial": torch.zeros(1, 3, 2), "refined": torch.ones(1, 3, 2),
        "optimizer": {"step": torch.tensor(256)}}
    value = {**deepcopy(old), "diagnostic_group": "original_252", "label_origin": "registered256_exact_reuse"}
    audit.replay_reuse(value, saved, row, {"a": old}, {"a": saved})
    changed = deepcopy(saved)
    changed["optimizer"]["step"] += 1
    with pytest.raises(ValueError, match="reused256"):
        audit.replay_reuse(value, changed, row, {"a": old}, {"a": saved})
    value["arms"]["refined_encoder"]["correct"] = 1
    with pytest.raises(ValueError, match="reused256"):
        audit.replay_reuse(value, saved, row, {"a": old}, {"a": saved})


def test_fixed_configuration_preserves_the_established_latent_objective():
    config = run.Config()
    assert config.updates == 256 and config.pilot_updates == 2
    assert config.lr == .02 and config.anchor_weight == .001
    assert config.word_vocab == config.pad == 50257
    with pytest.raises(FrozenInstanceError):
        config.updates = 512
