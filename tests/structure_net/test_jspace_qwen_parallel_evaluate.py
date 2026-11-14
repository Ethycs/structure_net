from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.structure_net import jspace_qwen_parallel_evaluate as evaluator


THRESHOLDS = dict(maximum_accuracy_drop=.01, maximum_family_drop=4/128,
                  minimum_size_reduction=.38, minimum_speedup=1.4)


def rows(count=512):
    return [dict(identity=f"case_{i}", family=f"family_{i//128}", answer=0,
                 prediction=0, logits=[1., 0.]) for i in range(count)]


def tasks():
    evaluation = [{**r, "input_ids": [1, 2], "choices": ["a", "b"]} for r in rows()]
    development = [{**evaluation[128*f+i], "identity": f"dev_{f}_{i}"}
                   for f in range(4) for i in range(2)]
    return {"splits": {"evaluation": evaluation, "development": development, "train": []},
            "previous_identities": ["previous"]}


def gates(quality, **changes):
    args = dict(native_bytes=10000, compact_bytes=6200, native_before=10., compact_rate=14.,
                native_after=10., reload_exact=True, graph_static_exact=True, config=THRESHOLDS)
    args.update(changes)
    return evaluator.acceptance_gates(quality, **args)


def test_paired_counts_exact_probability_and_thresholds():
    native, compact = rows(), rows()
    for i in range(4):
        compact[i]["prediction"] = 1
    compact[128]["prediction"] = 1
    quality = evaluator.paired_quality(native, compact)
    assert quality["overall"]["losses"] == 5
    assert quality["overall"]["wins"] == 0
    assert quality["overall"]["exact_binomial_p"] == 1/16
    assert gates(quality)["passed"]  # 5/512 < 1%, family 4/128 exactly allowed.
    compact[129]["prediction"] = 1
    result = gates(evaluator.paired_quality(native, compact))
    assert not result["gates"]["overall_quality"]
    compact = rows()
    for i in range(5):
        compact[i]["prediction"] = 1
    result = gates(evaluator.paired_quality(native, compact))
    assert result["gates"]["overall_quality"] and not result["gates"]["every_family_quality"]


def test_family_drop_is_net_accuracy_not_discordant_losses():
    native, compact = rows(), rows()
    for i in range(5):
        compact[i]["prediction"] = 1
    native[5]["prediction"] = 1
    result = evaluator.paired_quality(native, compact)
    assert result["by_family"]["family_0"]["losses"] == 5
    assert result["by_family"]["family_0"]["wins"] == 1
    assert result["by_family"]["family_0"]["net_lost_correct"] == 4
    assert gates(result)["passed"]


def test_speed_uses_average_native_rates_and_gates_exact_boundaries():
    quality = evaluator.paired_quality(rows(), rows())
    assert gates(quality, native_before=8., native_after=12.)["passed"]
    assert not gates(quality, compact_bytes=6201)["gates"]["deployed_size"]
    assert not gates(quality, compact_rate=13.999)["gates"]["whole_model_speed"]
    assert not gates(quality, reload_exact=False)["passed"]
    assert not gates(quality, graph_static_exact=False)["passed"]
    with pytest.raises(ValueError, match="finite"):
        gates(quality, compact_rate=float("nan"))
    assert evaluator.exact_binomial_p(0, 0) == 1
    assert evaluator.exact_binomial_p(3, 3) == 1
    assert evaluator.exact_binomial_p(0, 512) == 2**-511
    with pytest.raises(ValueError):
        evaluator.exact_binomial_p(-1, 0)


def test_rejects_misaligned_labels_duplicates_and_nonfresh_tasks():
    native, compact = rows(), rows()
    compact[0]["answer"] = 1
    with pytest.raises(ValueError, match="Unpaired"):
        evaluator.paired_quality(native, compact)
    native[1]["identity"] = native[0]["identity"]
    with pytest.raises(ValueError, match="Duplicate"):
        evaluator.paired_quality(native, rows())
    data = tasks()
    assert len(evaluator.evaluation_cases(data)) == 512
    assert len(evaluator.runtime_cases(data)) == 8
    data["previous_identities"].append("case_0")
    with pytest.raises(ValueError, match="previously"):
        evaluator.evaluation_cases(data)
    data = tasks(); data["splits"]["development"][0]["input_ids"] = [1]*385
    with pytest.raises(ValueError, match="416"):
        evaluator.runtime_cases(data)


class TinyStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))
        self.packed, self.stripped, self.layer_indices = True, True, tuple(range(36))
        self.extra = {"packed": True, "layers": list(self.layer_indices)}

    def forward(self, hidden):
        return hidden*self.scale

    def get_extra_state(self):
        return deepcopy(self.extra)

    def set_extra_state(self, state):
        self.extra = deepcopy(state)


class TinyBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 3)
        self.layers = nn.ModuleList([TinyStack()])
        self.norm = nn.Linear(3, 3, bias=False)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.norm(self.layers[0](self.embed_tokens(input_ids))))


class TinyModel(nn.Module):
    def __init__(self, compact=False):
        super().__init__()
        self.model = TinyBody()
        self.lm_head = nn.Linear(3, 4, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.compact = compact

    def generate(self, ids, **kwargs):
        return torch.cat((ids, torch.zeros((1, 32), dtype=torch.long)), 1)


def test_last_state_choice_score_and_tied_tensor_bytes():
    torch.manual_seed(13)
    model = TinyModel()
    case = dict(identity="test", family="f", input_ids=[1, 2], choices=["A", "B"], answer=1)
    records = evaluator.score_cases(model, [case], [3, 1])
    expected = nn.functional.linear(model.model(torch.tensor([[1, 2]])).last_hidden_state[:, -1],
                                    model.lm_head.weight[[3, 1]])[0]
    assert records[0]["logits"] == expected.tolist()
    assert records[0]["prediction"] == int(expected.argmax())
    assert evaluator.tensor_bytes(model) == (4*3+3*3+1)*4
    assert all(p.grad is None for p in model.parameters())


def test_packed_export_strict_reload_metadata_and_append_only(tmp_path):
    model = TinyModel()
    path = tmp_path/"packed.pt"
    evaluator.export_packed(path, model, fingerprint="frozen")
    with pytest.raises(FileExistsError):
        evaluator.export_packed(path, model, fingerprint="frozen")
    fresh = deepcopy(model)
    with torch.no_grad():
        fresh.model.layers[0].scale.add_(.2)
        fresh.model.norm.weight.zero_()
    evaluator.reload_packed(path, fresh, fingerprint="frozen")
    assert torch.equal(fresh.model.layers[0].scale, model.model.layers[0].scale)
    assert torch.equal(fresh.model.norm.weight, model.model.norm.weight)
    assert fresh.model.layers[0].extra == model.model.layers[0].extra
    with pytest.raises(ValueError, match="provenance"):
        evaluator.reload_packed(path, fresh, fingerprint="other")
    fresh.model.layers[0].register_buffer("unexpected", torch.zeros(1))
    with pytest.raises(RuntimeError, match="Missing key"):
        evaluator.reload_packed(path, fresh, fingerprint="frozen")


def test_runtime_measures_24_graph_calls_and_shared_eager_prefill(monkeypatch):
    from experiments.structure_net import jspace_qwen_affine as base

    class Decoder:
        def capture(self):
            return .5

        def generate(self, ids, captured):
            return torch.zeros(32, dtype=torch.long)

        def prefill(self, ids):
            return torch.zeros(1, 3)

    monkeypatch.setattr(base, "graph_decoder", lambda model: Decoder())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(evaluator, "_timed", lambda call: (call(), .25))
    value = evaluator.runtime(TinyModel(), SimpleNamespace(eos_token_id=0),
                              evaluator.runtime_cases(tasks()), seed=33)
    assert value["graph_static_exact"]
    assert len(value["timings"]["cuda_graph"]["samples"]) == 24
    assert len(value["timings"]["eager_static"]["samples"]) == 8
    assert len(value["timings"]["hf_eager"]["samples"]) == 8
    assert value["timings"]["cuda_graph"]["tokens_per_second"] == 128
    assert len(value["prefill"]["samples"]) == 24
    assert "eager" in value["prefill"]["mode"]
    assert all(value["hf_graph_tokens_equal"])


def test_evaluate_orders_models_and_produces_manifest_without_quality_selection(tmp_path, monkeypatch):
    from experiments.structure_net import jspace_qwen_parallel as campaign

    plan = dict(fingerprint="frozen", systems_only=False, label_ids=[0, 1],
                config={**THRESHOLDS, "seed": 33, "scope": "test"})
    (tmp_path/"tasks.json").write_text(json.dumps(tasks()))
    setup = (plan, {"id": "reserved", "allocation": {"card": 1}}, lambda: None)
    monkeypatch.setattr(campaign, "verify", lambda *a, **kw: plan)
    monkeypatch.setattr(campaign, "checked", lambda *a, **kw: {})
    monkeypatch.setattr(campaign, "sha", lambda *a, **kw: "0"*64)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 500)
    order = []
    def build(root, previous):
        order.append(previous)
        return TinyModel(compact=previous is not None), SimpleNamespace(eos_token_id=0)
    monkeypatch.setattr(evaluator, "build_teacher", build)
    monkeypatch.setattr(evaluator, "tensor_bytes", lambda model: 6200 if model.compact else 10000)
    monkeypatch.setattr(evaluator, "score_cases", lambda *a, **kw: rows())
    monkeypatch.setattr(evaluator, "runtime", lambda model, *a, **kw: {
        "graph_static_exact": True,
        "timings": {"cuda_graph": {"tokens_per_second": 14. if model.compact else 10.}}})
    result = evaluator.evaluate(tmp_path, setup=setup)
    assert order == [None, 5, None, 5]
    assert result["targets_met"] and result["packed_reload_exact"]
    assert not result["confirmed"]  # Exploratory evidence stays conservative.
    assert result["classification"] == "all_targets_met_on_restricted_benchmark"
    manifest = json.loads((tmp_path/"evaluation/manifest.json").read_text())
    assert "evaluation/packed.pt" in manifest["artifacts"]
    assert json.loads((tmp_path/"campaign_results.json").read_text()) == result
    with pytest.raises(FileExistsError):
        evaluator.evaluate(tmp_path, setup=setup)
