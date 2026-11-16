from types import SimpleNamespace
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event

import pytest
import torch
from torch import nn

from experiments.structure_net import jspace_3b_response_campaign as campaign


def records(n=100):
    return [SimpleNamespace(split="development" if i < 4 else "train", task="task",
        provenance={"source_family": "family"+str(i%4)},
        response_ids=tuple(range(2+i%9)), input_ids=tuple(range(10+i%17))) for i in range(n)]


def test_balanced_order_is_one_pass_train_only_and_retains_whole_responses():
    corpus = records()
    first = campaign.training_windows(corpus, budget=120, accumulation=8, seed=5)
    second = campaign.training_windows(corpus, budget=120, accumulation=8, seed=5)
    assert first == second
    assert len(set(first["order"])) == len(first["order"])
    assert all(corpus[i].split == "train" for i in first["order"])
    assert first["actual_supervised_tokens"] >= 120
    assert sum(len(corpus[i].response_ids) for i in first["order"][:-8]) < 120
    assert len(set(first["family_examples"].values())) == 1
    with pytest.raises(ValueError, match="cannot cover"):
        campaign.training_windows(corpus, budget=100_000)


def test_checkpoint_thresholds_are_crossed_only_at_complete_windows():
    corpus = records()
    order = campaign.training_windows(corpus, budget=120, accumulation=8)
    boundaries = campaign.checkpoint_boundaries(order["windows"], corpus, thresholds=(25, 50, 100))
    assert boundaries[-1]["supervised_tokens"] >= 100
    assert [x for row in boundaries for x in row["crossed_thresholds"]] == [25, 50, 100]
    assert len({row["update"] for row in boundaries}) == len(boundaries)


def test_pilot_includes_longest_inputs_responses_and_all_families():
    corpus = records()
    corpus[80].input_ids = tuple(range(416))
    corpus[81].response_ids = tuple(range(128))
    order = campaign.training_windows(corpus, budget=120, accumulation=16)["order"]
    windows = campaign.pilot_windows(corpus, order)
    included = [i for window in windows for i in window]
    assert len(included) == len(set(included)) == 32
    assert 80 in included and 81 in included
    assert {corpus[i].provenance["source_family"] for i in included} == {"family0", "family1", "family2", "family3"}
    assert all(corpus[i].split == "train" for i in included)


def candidate(update, *, passed=True, semantic=.75, nll=2.):
    return dict(update=update, semantic_task_success=semantic, nll=nll,
                checkpoint={"path":str(update)}, eligibility={"passed":passed})


def test_failed_generation_cannot_be_rescued_by_lower_nll_or_mc():
    gates = dict(maximum_native_nll_ratio=1.1)
    result = campaign.combine_gates({"ce":.1}, {"nll":2.}, {"passed":False},
        {"passed":True}, {"eligibility":{"passed":True}}, gates, task_checks={"passed":True})
    assert not result["passed"] and not result["checks"]["canaries"]
    result = campaign.combine_gates({"ce":2.3}, {"nll":2.}, {"passed":True},
        {"passed":True}, {"eligibility":{"passed":True}}, gates, task_checks={"passed":True})
    assert not result["passed"] and not result["checks"]["response_nll"]
    missing = campaign.combine_gates({"ce":.1}, {"nll":2.}, {"passed":True},
        {"passed":True}, {"eligibility":{"passed":True}}, gates, task_checks={"passed":False})
    assert not missing["passed"] and not missing["checks"]["generative_tasks"]


def test_selection_preserves_initializer_and_earliest_equivalent_checkpoint():
    rows = [candidate(0, passed=False), candidate(20), candidate(40), candidate(60, semantic=1., passed=False)]
    selected = campaign.select_checkpoint(rows)
    assert selected["selected"]["update"] == 20
    assert selected["diagnostic_best"]["update"] == 60
    assert selected["initializer"]["update"] == 0 and not selected["confirmed"]
    failed = campaign.select_checkpoint([candidate(0, passed=False), candidate(20, passed=False)])
    assert failed["selected"] is None and failed["fallback"]["update"] == 0
    with pytest.raises(ValueError, match="Initializer"):
        campaign.select_checkpoint([candidate(20)])


def test_pilot_needs_real_bf16_changes_even_when_fp32_masters_moved():
    blocks = [nn.Linear(2, 2), nn.Linear(2, 2)]
    norm = nn.LayerNorm(2)
    params = [p for block in blocks for p in block.parameters()] + list(norm.parameters())
    trainer = SimpleNamespace(parameters=params, stack=SimpleNamespace(blocks=blocks), norm=norm)
    rows = [dict(loss=1., gradients_present_by_block=[True, True],
        optimizer={"master_changed_by_parameter":[True]*len(params), "unclipped_gradient_norm":1.})]
    report = campaign._pilot_report(trainer,
        {"bf16_changed_by_block":[True, False], "norm_bf16_storage_changed":False}, rows)
    assert not report["passed"]
    report = campaign._pilot_report(trainer,
        {"bf16_changed_by_block":[True, True], "norm_bf16_storage_changed":False}, rows)
    assert report["passed"] and report["norm_fp32_master_changed"]


def test_runtime_configuration_is_full_response_exact_native_kl():
    config = campaign.training.ResponseConfig(max_supervised_tokens=100_131).loss_config()
    assert config.objective == "forward_kl" and config.temperature == 1.
    assert config.kd_weight == 1. and config.ce_weight == 0.
    assert config.max_supervised_tokens == 100_131


def test_selected_checkpoint_restores_earlier_weights_and_replays_saved_metrics(tmp_path, monkeypatch):
    stack, norm = nn.Linear(2, 2), nn.LayerNorm(2)
    identities = dict(fingerprint="study", model={"revision":"teacher"},
                      checkpoint_sha256="initializer", corpus_fingerprint="corpus", cache_fingerprint="cache")
    checkpoint = tmp_path/"earlier.pt"
    torch.save(dict(schema="jspace.complete-response-checkpoint.v1", stack=stack.state_dict(),
                    norm=norm.state_dict(), identities=campaign._identities(identities), updates=1), checkpoint)
    def development():
        return {"ce":float(stack.weight.detach().sum())}, [{"example_id":"dev", "tokens":3}]
    metrics, rows = development()
    directory = tmp_path/"evaluation"
    directory.mkdir()
    campaign.write(directory/"development.json", dict(**metrics, records=rows))
    trainer = SimpleNamespace(stack=stack, norm=norm, readout=None, rotary=None, fixed="fixed",
                              memory={"snapshot_bytes":32}, development=development)
    monkeypatch.setattr(campaign.offload, "fixed_state", lambda *args:"fixed")
    monkeypatch.setattr(campaign.offload, "require_host_memory", lambda *args, **kwargs:None)
    selected = dict(update=1, checkpoint={"path":str(checkpoint), "sha256":campaign.sha(checkpoint)},
                    evaluation_path=str(directory))
    parameters = tuple(map(id, stack.parameters()))
    with torch.no_grad():
        stack.weight.add_(1.)
    result = campaign._reload_selected(trainer, identities, selected)
    assert result["passed"] and result["complete_response_dev_exact"]
    assert tuple(map(id, stack.parameters())) == parameters and development()[0] == metrics
    selected["checkpoint"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="changed"):
        campaign._reload_selected(trainer, identities, selected)


def test_async_judge_uses_immutable_checkpoint_snapshot_while_training_can_advance(monkeypatch, tmp_path):
    started, release = Event(), Event()
    def finish(directory, output, snapshot):
        started.set()
        assert release.wait(2)
        return snapshot
    monkeypatch.setattr(campaign, "_complete_evaluation", finish)
    snapshot = dict(update=0, generation=[{"response":"saved output"}], checkpoint={"sha256":"old"})
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = campaign.queue_judgment(pool, tmp_path, tmp_path/"eval", snapshot)
        assert started.wait(2)
        snapshot["update"] = 1
        snapshot["generation"][0]["response"] = "later output"
        snapshot["checkpoint"]["sha256"] = "new"
        assert not future.done()
        release.set()
        result = future.result(timeout=2)
    assert result == dict(update=0, generation=[{"response":"saved output"}], checkpoint={"sha256":"old"})


def test_pending_judgments_cannot_enter_candidate_selection_before_resolution():
    ready, pending = Future(), Future()
    ready.set_result(candidate(0))
    candidates = []
    remaining = campaign.collect_judgments([ready, pending], candidates)
    assert remaining == [pending] and [row["update"] for row in candidates] == [0]
    pending.set_result(candidate(20, passed=False))
    assert campaign.collect_judgments(remaining, candidates, block=True) == []
    assert [row["update"] for row in candidates] == [0, 20]
