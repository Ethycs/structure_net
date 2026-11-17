"""CPU-only controls for streaming, exact saved evidence and paired accounting."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net import jspace_3b_mmlu_redux as study
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack
from experiments.structure_net import jspace_qwen3b_progressive_model as builder
from test_jspace_3b_ablation32_compile import fixture, context


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_stream_matches_independent_full_native_forward_and_exact_boundary(dtype):
    model, _, _ = fixture(dtype)
    stream = study.NativeStream(model.model.layers, model.model.embed_tokens.weight,
        model.model.norm, model.model.rotary_emb, guard=lambda: None, cuts=(1,))
    ids = [1, 3, 6, 8, 4]
    with torch.inference_mode():
        expected = model.model(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[:, -1]
        h, pos, mask, rope = context(model, torch.tensor([ids]))
        boundary = model.model.layers[0](h, attention_mask=mask, position_ids=pos,
            position_embeddings=rope, cache_position=pos[0], use_cache=False)[0]
        actual, boundaries = stream.forward(ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(boundaries[1], boundary, rtol=0, atol=0)
    assert stream.calls == 1
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_host_storage_restored_after_compute_failure_and_guard_failure():
    module = torch.nn.Linear(3, 2).eval().requires_grad_(False)
    original = [(p, p.data_ptr(), p.clone()) for p in module.parameters()]
    with pytest.raises(RuntimeError, match="compute"):
        with study.host_weights(module, "cpu", lambda: None):
            # Change storage, not the original frozen host tensor.
            module.weight.data = module.weight.clone()
            raise RuntimeError("compute")
    for p, pointer, value in original:
        assert p.data_ptr() == pointer
        assert torch.equal(p, value)
    calls = []
    def guard():
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("lease")
    with pytest.raises(RuntimeError, match="lease"):
        with study.host_weights(module, "cpu", guard):
            pytest.fail("Admission failure must not execute computation")
    assert all(p.data_ptr() == pointer for p, pointer, _ in original)


def test_choice_argmax_is_four_way_not_full_vocabulary():
    embedding = torch.tensor([[99.], [1.], [4.], [3.], [2.]])
    scores = study.choice_scores(torch.ones(1, 1), embedding, [1, 2, 3, 4])
    assert scores == dict(choice_logits=[1., 4., 3., 2.], prediction=1)
    with pytest.raises(ValueError):
        study.choice_scores(torch.ones(1, 1), embedding, [1, 1, 2, 3])


def cpu_runner(tmp_path, monkeypatch):
    model, _, full = fixture(torch.bfloat16)
    block = AffineQwenBlock(model.model.layers[1], full.blocks[1].basis,
        torch.zeros(1, 8), parameter_dtype=torch.bfloat16)
    suffix = CompactQwenStack([block], geometry_mode="fp32", enabled=False)
    compact = study.packed.PackedResponseModel(builder.packed_clone(suffix, device="cpu"),
        deepcopy(model.model.norm), model.model.embed_tokens.weight,
        deepcopy(model.model.rotary_emb), config=model.config, max_length=416)
    export = tmp_path/"export"
    study.packed.export_packed_snapshot(compact, export, identities={"fixture": True})
    monkeypatch.setattr(study, "CUTS", {"blocks08": 1, "blocks03": 1})
    runner = study.Runner.__new__(study.Runner)
    runner.guard = lambda: None
    runner.plan = dict(max_length=416, models={name: {"export": str(export)} for name in study.CUTS})
    runner.embedding = model.model.embed_tokens.weight
    runner.host_embedding = runner.embedding
    runner.stream = study.NativeStream(model.model.layers, runner.embedding, model.model.norm,
        model.model.rotary_emb, guard=runner.guard, cuts=(1,))
    runner.students = {}
    for name in study.CUTS:
        fresh, _ = study.packed.load_packed_snapshot(export)
        fresh.embedding, fresh.guard = runner.embedding, runner.guard
        runner.students[name] = fresh
    ids, labels = [1, 3, 6, 8, 4], [1, 2, 3, 4]
    with torch.inference_mode():
        hidden = model.model(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[:, -1]
        expected = study.choice_scores(hidden, runner.embedding, labels)
    evidence = tmp_path/"existing_native.json"
    evidence.write_text(json.dumps(expected))
    control = dict(identity="existing-dev", input_ids=ids, label_ids=labels,
        expected_choice_logits=expected["choice_logits"], evidence={str(evidence): study.sha(evidence)})
    return runner, control


def test_pilot_authenticates_independent_cache_and_real_packed_reload(tmp_path, monkeypatch):
    runner, control = cpu_runner(tmp_path, monkeypatch)
    proof = runner.pilot(control)
    assert proof["passed"] and proof["native_control"]["bit_exact"]
    assert proof["benchmark_answers_observed"] == 0
    assert all(v["bit_exact"] and v["production_entrypoint_exact"] for v in proof["packed_reload"].values())
    assert runner.stream.calls == 1
    wrong = deepcopy(control)
    wrong["expected_choice_logits"][0] += 1
    with pytest.raises(ValueError, match="authenticated existing DEV"):
        runner.pilot(wrong)
    Path(next(iter(control["evidence"]))).write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        runner.pilot(control)


def test_actual_prefix_shared_once_and_unsupported_students_preserved(tmp_path, monkeypatch):
    runner, control = cpu_runner(tmp_path, monkeypatch)
    result = runner.evaluate(control["input_ids"], eligible=True, label_ids=control["label_ids"])
    assert runner.stream.calls == 1
    assert all(result[name] is not None for name in study.MODEL_NAMES)
    with pytest.raises(ValueError, match="eligibility"):
        runner.evaluate(control["input_ids"], eligible=False, label_ids=control["label_ids"])
    # A smaller declared envelope exercises unsupported accounting with the
    # real native forward, without extending the tiny native fixture's context.
    runner.plan["max_length"] = 4
    result = runner.evaluate(control["input_ids"], eligible=False, label_ids=control["label_ids"])
    assert result["teacher"] is not None and result["blocks08"] is None and result["blocks03"] is None
    assert runner.stream.calls == 2
    with pytest.raises(ValueError, match="Unsupported"):
        runner.student("blocks03", control["input_ids"], None, control["label_ids"])


def report_fixture():
    cases, rows = [], []
    for index, (teacher, student, eligible, corrected) in enumerate([
            (0, 0, True, 0), (1, 0, True, None), (0, 1, True, 1), (0, None, False, 0)]):
        case = dict(identity=str(index), subject="s"+str(index%2), eligible=eligible,
            input_ids=[1, 2] if eligible else [1]*416, answer_original=0,
            answer_corrected=corrected, contamination={"overlap": index == 0})
        def output(prediction):
            return None if prediction is None else dict(prediction=prediction,
                choice_logits=[float(i == prediction) for i in range(4)])
        rows.append(dict(identity=str(index), subject=case["subject"], eligible=eligible,
            models=dict(teacher=output(teacher), blocks08=output(student), blocks03=output(student))))
        cases.append(case)
    return cases, rows


def test_reports_paired_counts_corrected_denominators_overlap_and_unsupported():
    cases, rows = report_fixture()
    report = study.summarize(cases, rows)
    model = report["all"]["models"]["blocks08"]
    assert model["evaluated"] == 3 and model["unsupported"] == 1
    assert model["original"] == dict(count=3, correct=2, accuracy=2/3)
    assert model["corrected"] == dict(count=2, correct=2, accuracy=1.)
    pair = report["all"]["paired_vs_teacher"]["blocks08"]["original"]
    assert (pair["count"], pair["wins"], pair["losses"], pair["both_correct"]) == (3, 1, 1, 1)
    assert pair["accuracy_delta"] == 0 and pair["bootstrap_draws"] == 10000
    assert report["paired_eligible"]["models"]["teacher"]["evaluated"] == 3
    assert report["contamination"]["known_overlap"]["count"] == 1
    assert report["contamination"]["disjoint_from_audited_sources"]["count"] == 3
    assert not report["published_context"]["matched_protocol"]
    assert study.summarize(cases, rows) == report
    assert study.paired([], {}, "blocks08", "answer_original")["bootstrap_95_ci"] is None


def test_stratified_bootstrap_preserves_fixed_subject_counts():
    cases, rows = report_fixture()
    # Two strata each have a deterministic difference, so subject resampling
    # would incorrectly produce variance while within-subject resampling does not.
    cases = [cases[1], cases[2]]
    outputs = {row["identity"]: row["models"] for row in rows}
    result = study.paired(cases, outputs, "blocks08", "answer_original")
    assert result["bootstrap_95_ci"] == [0., 0.]


def test_output_validation_rejects_missing_teacher_and_unsupported_execution():
    cases, rows = report_fixture()
    study._validate_rows(cases, rows, {"max_length": 416})
    bad = deepcopy(rows)
    bad[0]["models"]["teacher"] = None
    with pytest.raises(ValueError, match="Missing supported"):
        study._validate_rows(cases, bad, {"max_length": 416})
    bad = deepcopy(rows)
    bad[-1]["models"]["blocks08"] = bad[0]["models"]["blocks08"]
    with pytest.raises(ValueError, match="Unsupported student"):
        study._validate_rows(cases, bad, {"max_length": 416})


def test_run_rejects_missing_pilot_before_creating_run_or_reserving_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "_plan", lambda root: dict(fingerprint="f", max_length=416,
        models={name: {} for name in study.CUTS}))
    with pytest.raises(FileNotFoundError):
        study.run(tmp_path)
    assert not (tmp_path/"run").exists()


def test_manifest_requires_result_binding_and_rejects_changed_evidence(tmp_path):
    study.write(tmp_path/"result.json", dict(status="completed", fingerprint="f"))
    study.write(tmp_path/"manifest.json", dict(status="completed", fingerprint="f",
        artifacts={"result.json": study.sha(tmp_path/"result.json")}))
    assert study._receipt(tmp_path, "f")["status"] == "completed"
    (tmp_path/"result.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        study._receipt(tmp_path, "f")
