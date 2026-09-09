"""Leakage, scoring, optimization, and checkpoint contracts for accuracy95."""
from dataclasses import replace
import json

import pytest
import torch
import torch.nn.functional as F

from experiments.structure_net import tinyllm_moba_accuracy95 as a
from experiments.structure_net import tinyllm_moba_distillation as base


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return a.AccuracyConfig(device="cpu", shakedown=True)


def record(pair_id, good, bad, suite="suite"):
    return {"pairID": str(pair_id), "sentence_good": good, "sentence_bad": bad, "UID": suite}


def test_split_excludes_good_and_bad_sentence_leakage():
    rows = [record(0, "Old good", "Old bad"), record(900, "Fresh good", "Fresh bad"),
            record(800, "Val good", "Val bad"), record(801, "fresh BAD", "different"),
            record(32, "Train good", "Train bad"), record(33, "other", "Old good"),
            record(34, "val GOOD", "different2")]
    parts = a.split_pairs(rows)
    assert [r["pairID"] for r in parts["train"]] == ["32"]
    assert [r["pairID"] for r in parts["validation"]] == ["800"]
    with pytest.raises(RuntimeError, match="tests overlap"):
        a.split_pairs([record(0, "same", "bad"), record(900, "same", "other")])


def test_real_splits_and_original_evaluation_preserved(config):
    parts = a.load_pairs(config)
    assert {k:len(v) for k,v in parts.items()} == {"historical":128,"train":3072,"validation":400,"test":400}
    assert all(int(r["pairID"]) < 32 for r in parts["historical"])
    sets = {k: {r[field].strip().casefold() for r in rows for field in ("sentence_good", "sentence_bad")} for k, rows in parts.items()}
    for first in sets:
        for second in sets:
            if first != second:
                assert not sets[first] & sets[second]


def test_padding_does_not_change_sentence_likelihood(config):
    model = base.TinyLLMModel(base.TinyLLMConfig(block_size=256, vocab_size=64, n_layer=2, n_head=2, n_embd=16)).eval()
    rows = [{"good":[1,2,3],"bad":[1,4,5,6]}, {"good":[1,7,8,9,10],"bad":[1,3]}]
    ids, labels = a.pair_batch(rows, "cpu")
    logits, _ = base.forward(model, ids, config, 2, capture=False)
    _, scores, _ = a.pair_objective(logits, labels, config)
    for index, seq in enumerate([r[k] for r in rows for k in ("good", "bad")]):
        x = torch.tensor([seq[:-1]])
        y = torch.tensor(seq[1:])
        l, _ = base.forward(model, x, config, 2, capture=False)
        expected = -F.cross_entropy(l.flatten(0,1), y, reduction="sum")
        torch.testing.assert_close(scores[index], expected, atol=1e-5, rtol=1e-6)


def test_pair_gradient_prefers_grammatical_sentence(config):
    cfg = replace(config, good_lm_weight=0)
    logits = torch.zeros(2, 1, 3, requires_grad=True)
    labels = torch.tensor([[1], [2]])
    loss, _, _ = a.pair_objective(logits, labels, cfg)
    loss.backward()
    assert logits.grad[0,0,1] < 0  # Raise the grammatical target logit.
    assert logits.grad[1,0,2] > 0  # Lower the ungrammatical target logit.


def test_threshold_inclusivity_and_joint_accuracy(config):
    assert a.validation_gate(.98, 4.2, 4, config)
    assert not a.validation_gate(.9799, 4, 4, config)
    assert a.final_gate(.95,.95,4.2,4,config) == {"accuracy":True,"nll":True,"joint":True}
    assert not a.final_gate(.96,.94,4,4,config)["joint"]
    assert not a.final_gate(.99,.99,4.21,4,config)["joint"]


def test_schedules_are_paired_and_avoid_parent_test_windows(config):
    config = replace(config, validation_windows=4, final_windows=8)
    count=13_101_712
    first=a.make_schedules(count,3072,7,config)
    second=a.make_schedules(count,3072,7,config)
    import numpy as np
    assert all(np.array_equal(first[k],second[k]) for k in first)
    split=count-config.validation_tokens
    assert first["replay"].max()+257 <= split
    assert first["validation"].min() >= split
    assert first["test"].min() >= split+131072
    for seed in (7,17,29,41,53):
        with np.load(a.PARENT/"schedules"/f"seed_{seed}.npz") as old:
            assert not set(first["test"]) & set(old["test"])


def test_cpu_forward_backward_checkpoint_lifecycle(config,tmp_path):
    model=base.TinyLLMModel(base.TinyLLMConfig(block_size=256,vocab_size=64,n_layer=2,n_head=2,n_embd=16))
    before=a.state_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    ids,labels=a.pair_batch([{"good":[1,2,3],"bad":[1,4,3]}],"cpu")
    logits,_=base.forward(model,ids,config,2,capture=False)
    loss,_,_=a.pair_objective(logits,labels,config)
    loss.backward()
    optimizer.step()
    assert a.state_hash(model) != before
    checkpoint=a.save_checkpoint(tmp_path/"model.pt",model,"test-fingerprint",1)
    reloaded=a.load_model(checkpoint["path"],"cpu")
    assert a.state_hash(reloaded)==a.state_hash(model)
    with torch.no_grad():
        x,_=base.forward(model,ids,config,2,capture=False)
        y,_=base.forward(reloaded,ids,config,2,capture=False)
    torch.testing.assert_close(x,y,atol=0,rtol=0)
    base.write_json(tmp_path/"result.json",{"profile":"systems_lifecycle_only_not_quality_evidence","checkpoint":checkpoint,"loss":float(loss.detach())})
    assert json.loads((tmp_path/"result.json").read_text())["checkpoint"]["sha256"]==base.sha256(tmp_path/"model.pt")
