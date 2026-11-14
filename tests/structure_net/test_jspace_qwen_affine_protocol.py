"""Protocol boundaries: provenance, held-out data and compilation order."""
from contextlib import nullcontext
import copy
import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from experiments.structure_net import jspace_qwen_affine as study


@pytest.fixture
def frozen_run(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    source = tmp_path / "producer.py"
    source.write_text("# frozen implementation\n")
    run = tmp_path / "run"
    (run / "producer").mkdir(parents=True)
    (run / "producer" / source.name).write_bytes(source.read_bytes())
    tasks = {"splits": {"evaluation": [{"identity": "held-out", "answer": 1}]}}
    (run / "tasks.json").write_text(json.dumps(tasks))
    plan = {"study_id": study.STUDY_ID, "config": copy.deepcopy(study.CONFIG),
            "source_sha256": {str(source): study.sha(source)},
            "task_digest": study.digest(tasks), "input_sha256": {}}
    plan["fingerprint"] = study.digest(plan)
    (run / "plan.json").write_text(json.dumps(plan))
    return run, source


@pytest.mark.parametrize("copy_name", ["live", "archived"])
def test_frozen_producer_rejects_changes_in_either_copy(frozen_run, copy_name):
    run, source = frozen_run
    study.verify(run)
    target = source if copy_name == "live" else run / "producer" / source.name
    target.write_text("# changed implementation\n")
    with pytest.raises(ValueError, match="Changed producer"):
        study.verify(run)


def test_changed_optimizer_or_held_out_labels_invalidate_fingerprint(frozen_run, monkeypatch):
    run, _ = frozen_run
    with monkeypatch.context() as patch:
        patch.setitem(study.CONFIG, "learning_rate", study.CONFIG["learning_rate"] * 2)
        with pytest.raises(ValueError, match="Changed protocol"):
            study.verify(run)
    tasks = json.loads((run / "tasks.json").read_text())
    tasks["splits"]["evaluation"][0]["answer"] = 0
    (run / "tasks.json").write_text(json.dumps(tasks))
    with pytest.raises(ValueError, match="Changed tasks"):
        study.verify(run)


def test_fresh_evaluation_excludes_seen_long_and_duplicate_questions(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(study, "DATASETS", {"boolq": None})
    monkeypatch.setitem(study.CONFIG, "fresh_evaluation_per_task", 2)
    monkeypatch.setitem(study.CONFIG, "max_tokens", 3)
    manifest = {"boolq": {"evaluation_source_split": "validation",
                          "files": {"validation": {"local": "unused.parquet"}}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    def row(question, answer=False):
        return {"passage": "A passage", "question": question, "answer": answer}
    source = [row("unsupported"), row("seen"), row("first"), row("first", True), row("second"), row("oversized")]
    native_normalize = study.normalize
    def normalize(family, item):
        if item["question"] == "unsupported":
            raise ValueError("Unsupported or unlabeled multiple-choice item")
        return native_normalize(family, item)
    monkeypatch.setattr(study, "normalize", normalize)
    monkeypatch.setattr(study.random, "Random", lambda seed: SimpleNamespace(shuffle=lambda order: None))
    arrow = ModuleType("pyarrow")
    parquet = ModuleType("pyarrow.parquet")
    parquet.read_table = lambda _: SimpleNamespace(to_pylist=lambda: source)
    arrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", arrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    question, choices, _ = study.normalize("boolq", source[1])
    seen = hashlib.sha256(json.dumps([question, choices]).encode()).hexdigest()
    old = {"splits": {"train": [{"identity": seen}], "development": [], "evaluation": []}}
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1] * (5 if "oversized" in messages[-1]["content"] else 2)
    actual, exclusions = study.fresh_cases(Tokenizer(), old)
    assert exclusions["boolq"]["unsupported_choice_format"] == 1
    assert len(actual) == 2
    assert len({r["identity"] for r in actual}) == 2
    assert seen not in {r["identity"] for r in actual}
    assert all(len(r["input_ids"]) <= 3 and "oversized" not in r["prompt"] for r in actual)
    assert actual == study.fresh_cases(Tokenizer(), old)[0]


def test_groebner_arm_has_a_matched_non_groebner_control_in_fingerprint():
    assert study.CONFIG["arms"] == study.ARMS
    treated = study.ARMS["r1536_h64_gb"]
    assert treated["groebner"] is True
    assert {k: v for k, v in treated.items() if k != "groebner"} == study.ARMS["r1536_h64"]
    changed = copy.deepcopy(study.CONFIG)
    changed["arms"]["r1536_h64_gb"]["groebner"] = False
    assert study.digest(changed) != study.digest(study.CONFIG)


def test_polynomial_compilation_cannot_load_an_unverified_fit(tmp_path, monkeypatch):
    class Tensor:
        def __getitem__(self, _): return self
        def cuda(self): return self
    compiled = []
    class Block:
        def __init__(self, *args, **kwargs): pass
        def compile_polynomial(self, *args): compiled.append(args)
    monkeypatch.setattr(study, "AffineQwenBlock", Block)
    def reject(root, stage):
        assert stage == "relations"
        raise ValueError("Unverified relations")
    monkeypatch.setattr(study, "checked", reject)
    monkeypatch.setattr(study.torch, "load", lambda *a, **k: pytest.fail("Fit loaded before verification"))
    stage = object.__new__(study.Stage)
    stage.root = tmp_path
    stage.model = SimpleNamespace(model=SimpleNamespace(layers=[object()]))
    stage.reset = lambda: None
    with pytest.raises(ValueError, match="Unverified relations"):
        stage.install({"basis": Tensor(), "means": Tensor()}, 1536, groebner=True)
    assert compiled == []


def test_primary_evaluation_waits_for_every_arm_to_complete(tmp_path, monkeypatch):
    seen = []
    last = list(study.ARMS)[-1]
    def checked(root, name):
        seen.append(name)
        if name == "train_" + last:
            raise ValueError("Last training arm incomplete")
    monkeypatch.setattr(study, "checked", checked)
    monkeypatch.setattr(study, "Stage", lambda *a: pytest.fail("Evaluation accessed before all training completed"))
    with pytest.raises(ValueError, match="training arm incomplete"):
        study.evaluate(tmp_path)
    assert seen == ["train_" + arm for arm in study.ARMS]


def test_packed_validation_restores_live_parameters_even_after_failure(monkeypatch):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.001, 2.003]))
        def pack(self, dtype, buffer_dtype):
            return self.to(dtype=dtype).requires_grad_(False)
    block = Block()
    norm = torch.nn.LayerNorm(2)
    originals = [*block.parameters(), *norm.parameters()]
    before = [p.detach().clone() for p in originals]
    model = SimpleNamespace(model=SimpleNamespace(layers=[block], norm=norm))
    stage = SimpleNamespace(model=model)
    def fail_validation(current, tag):
        assert current.model.model.layers[-1] is not block
        assert current.model.model.norm is not norm
        assert all(not p.requires_grad for p in current.model.model.layers[-1].parameters())
        raise RuntimeError("Validation failed")
    monkeypatch.setattr(study, "development", fail_validation)
    with pytest.raises(RuntimeError, match="Validation failed"):
        study.packed_development(stage, "test")
    assert model.model.layers[-1] is block and model.model.norm is norm
    assert all(p.requires_grad and torch.equal(p, saved) for p, saved in zip(originals, before))


def test_development_reads_only_development_cache_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(study.torch, "autocast", lambda *a, **k: nullcontext())
    accessed = []
    def logits(stage, case):
        accessed.append(case["identity"])
        return torch.tensor([[2., -1.], [2., -1.]])
    monkeypatch.setattr(study, "cached_logits", logits)
    monkeypatch.setattr(study, "read_bits", lambda *a, **k: torch.zeros(2, 2))
    class ForbiddenTasks(dict):
        def __getitem__(self, key):
            pytest.fail("Validation accessed task split " + key)
    def case(identity, split):
        return {"identity": identity, "split": split, "family": "boolq", "answer": 0,
                "choices": ["No", "Yes"], "targets": [0, 0]}
    stage = SimpleNamespace(items=[case("train", "train"), case("dev", "development"),
                                   case("heldout", "evaluation")],
                            tasks=ForbiddenTasks(), targets=None, label_ids=torch.tensor([0, 1]), path=tmp_path)
    metric, rows = study.development(stage, "test")
    assert accessed == ["dev"]
    assert metric["accuracy"]["count"] == 1 and [r["identity"] for r in rows] == ["dev"]
