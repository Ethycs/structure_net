from copy import deepcopy

import pytest
import torch

from experiments.structure_net import jspace_3b_recovery_diagnostics as diagnostic
from experiments.structure_net.jspace_qwen_parallel_model import state_dict_cpu
from test_jspace_qwen3b_progressive_model import source, build, recovered


def trace(output=(2, 3, 2, 3, 2, 3), kind="saved_generated"):
    return diagnostic.history("case", "benchmark:gsm8k", "development", [0, 1], list(output),
                              kind=kind, generating_state="state_a" if kind != "reference" else None)


def row(*, tail="old", insertion=None, contrast="native_boundary", value=1., item=None, norm="fixed"):
    item = item or trace()
    return dict(contrast=contrast, tail_state=tail, norm_state=norm, insertion_state=insertion,
        example_id=item["example_id"], family=item["family"], history_kind=item["history_kind"],
        generating_state=item["generating_state"], split=item["split"], history_sha256=item["history_sha256"],
        tokens=[dict(position=p, forward_kl=value, teacher_probability=.25, teacher_nll=1.4,
                     student_nll=2., greedy_agreement=False, phase=diagnostic.phase_at(item, i))
                for i, p in enumerate(item["predicting_positions"])])


def test_repetition_onset_is_first_repeated_token_not_original_phrase():
    item = trace()
    assert item["repetition"] == dict(start=0, onset=2, confirmed_at=5, end=6, period=2, repeats=3)
    assert [diagnostic.phase_at(item, i) for i in range(6)] == ["before_loop"]*2+["inside_loop"]*4
    assert diagnostic.repetition([1, 2, 1, 3, 1, 4]) is None


def test_cap_eos_and_syntactic_repetition_do_not_infer_correctness():
    clean = diagnostic.classify([1, 2, 3], [9], 3, 8, 2)
    assert clean["phenotype"] == "clean_cap_exhaustion"
    assert clean["correctness"] == "not_inferred"
    eos = diagnostic.classify([1, 9], [9], 3, 8, 2)
    assert eos["termination"] == "natural_eos"
    repeated = diagnostic.classify([1]*6, [9], 6, 8, 2)
    assert repeated["phenotype"] == "repetition_flagged"
    assert repeated["termination"] == "cap_exhausted"
    assert diagnostic.classify([1], [9], 3, 8, 2)["termination"] == "incomplete_unknown"
    with pytest.raises(ValueError, match="after EOS"):
        diagnostic.classify([9, 1], [9], 3, 8, 2)
    with pytest.raises(ValueError, match="positional support"):
        diagnostic.classify([1, 2, 3], [9], 3, 4, 2)


def test_same_length_changed_history_cannot_reuse_reference_teacher():
    reference = trace((2, 3, 4), "reference")
    generated = trace((2, 0, 4))
    with pytest.raises(ValueError, match="Teacher/history identity"):
        diagnostic.bind_history(generated, reference)
    changed = deepcopy(reference)
    changed["predicting_positions"][0] += 1
    with pytest.raises(ValueError, match="Teacher/history identity"):
        diagnostic.bind_history(changed, reference)


def test_full_vocab_metrics_use_predicting_positions_and_exact_target_tokens():
    item = trace((2, 3, 4))
    embedding = torch.eye(5)
    teacher = torch.zeros(1, 5, 5)
    student = torch.zeros_like(teacher)
    for position, target in zip(item["predicting_positions"], item["output_ids"]):
        teacher[0, position, target] = 2.
        student[0, position, target] = 1.
    rows = diagnostic.token_metrics(student, teacher, embedding, item, item, chunk=2)
    p = torch.tensor([2., 0., 0., 0., 0.]).softmax(-1)
    q = torch.tensor([1., 0., 0., 0., 0.]).softmax(-1)
    expected = (p*(p.log()-q.log())).sum().item()
    assert [r["position"] for r in rows] == [1, 2, 3]
    assert [r["token_id"] for r in rows] == [2, 3, 4]
    assert all(r["forward_kl"] == pytest.approx(expected, abs=1e-6) for r in rows)
    assert all(r["teacher_probability"] == pytest.approx(p[0].item()) for r in rows)
    assert all(r["greedy_agreement"] for r in rows)


def test_boundary_pairing_rejects_changed_tail_history_or_norm():
    control = row()
    treatment = row(insertion="new25", contrast="compressed_boundary", value=2.)
    assert diagnostic.paired_effects([control, treatment])[0]["mean_kl_difference"] == 1.
    for key, changed in (("tail_state", "updated"), ("history_sha256", "other_history"), ("norm_state", "updated_norm")):
        bad = dict(treatment, **{key: changed})
        with pytest.raises(ValueError, match="matched native-boundary"):
            diagnostic.paired_effects([control, bad])
    bad = deepcopy(treatment)
    bad["tokens"][0]["position"] += 1
    with pytest.raises(ValueError, match="positions differ"):
        diagnostic.paired_effects([control, bad])


def test_suffix_pairing_holds_insertion_history_and_norm_fixed():
    old = row(insertion="pre25", contrast="compressed_boundary")
    updated = row(tail="updated", insertion="pre25", contrast="compressed_boundary", value=.5)
    effects = diagnostic.suffix_effects([old, updated], "old")
    assert effects[0]["mean_kl_difference"] == -.5
    for key in ("insertion_state", "history_sha256", "norm_state", "generating_state"):
        bad = dict(updated, **{key: "different"})
        with pytest.raises(ValueError, match="same-history/insertion/norm"):
            diagnostic.suffix_effects([old, bad], "old")


def test_summary_preserves_task_position_history_and_loop_strata():
    generated = row()
    reference = row(item=trace(kind="reference"), value=2.)
    summary = diagnostic.aggregate([generated, reference])
    all_rows = [r for r in summary if r["stratum"] == "all"]
    assert len(all_rows) == 2
    assert {r["history_kind"] for r in all_rows} == {"saved_generated", "reference"}
    assert {r["family"] for r in all_rows} == {"math"}
    assert any(r["stratum"] == "inside_loop" and r["tokens"] == 4 for r in summary)
    assert any(r["stratum"] == "position_0000_0031" for r in summary)


def test_native_index_tensor_lineage_detects_dropped_recovery(source):
    parent = recovered(source)
    stack, norm, *_ = build(source, 1, parent)
    child = dict(stack=state_dict_cpu(stack), norm=state_dict_cpu(norm))
    exact = diagnostic.inherited_tensors(parent, child)
    assert exact["inherited_exact"]
    assert exact["parent_layers"] == list(range(2, 9))
    assert exact["child_layers"] == list(range(1, 9))
    wrong = deepcopy(child)
    key = next(k for k, v in wrong["stack"].items() if k.startswith("blocks.1.") and isinstance(v, torch.Tensor) and v.is_floating_point() and v.numel())
    wrong["stack"][key] += .25
    difference = diagnostic.inherited_tensors(parent, wrong)
    assert not difference["inherited_exact"]
    assert difference["changed_tensors"] == [key]
    wrong = deepcopy(child)
    wrong["norm"]["weight"] += .25
    assert not diagnostic.inherited_tensors(parent, wrong)["norm_equal"]


def test_sliced_real_compact_stack_preserves_saved_bridges_and_freezes_weights(source):
    parent = recovered(source)
    stack, _, _, rotary, _ = build(source, 1, parent)
    before = deepcopy(stack.state_dict())
    tail = diagnostic.slice_stack(stack, 1, len(stack.blocks))
    assert tail.layer_indices == tuple(range(2, 9))
    assert tail.packed and tail.stripped
    for actual, wanted in zip(tail.bridges, stack.bridges[1:]):
        assert torch.equal(actual.weight, wanted.weight)
        assert torch.equal(actual.shift, wanted.shift)
    x = torch.randn(1, 4, 8).bfloat16()
    ids = torch.zeros(1, 4, dtype=torch.long)
    value = tail(x, **diagnostic._kwargs(x, ids, rotary))[0]
    assert value.shape == x.shape and torch.isfinite(value).all()
    assert all(not p.requires_grad for p in tail.parameters())
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, stack.state_dict()[key])


def test_check_requires_cpu_hidden(monkeypatch, tmp_path):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        diagnostic.check(tmp_path)


def test_history_rejects_confirmation_data():
    with pytest.raises(ValueError, match="TEST/confirmation"):
        diagnostic.history("sealed", "math", "confirmation", [1], [2], kind="reference")


def test_prepared_check_authenticates_nested_inputs_and_decoded_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    root = tmp_path/"diagnostics"
    root.mkdir()
    checkpoint = tmp_path/"nested"/"checkpoint.bin"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"actual frozen model bytes")
    spec = tmp_path/"spec.json"
    spec.write_text("{}")
    plan = dict(schema=diagnostic.SCHEMA, spec_sha256=diagnostic.sha(spec), spec_path=str(spec),
                inputs={str(checkpoint): diagnostic.sha(checkpoint), str(spec): diagnostic.sha(spec)},
                sources={}, panel_missing=[])
    plan["fingerprint"] = diagnostic.digest(plan)
    diagnostic.write(root/"plan.json", plan)
    diagnostic.write(root/"decoded.json", dict(status="observational_only"))
    diagnostic.write(root/"prepare_manifest.json", dict(fingerprint=plan["fingerprint"],
        spec_sha256=plan["spec_sha256"], files={name: diagnostic.sha(root/name) for name in ("plan.json", "decoded.json")}))
    assert diagnostic.check(root)["status"] == "prepared_no_gpu_evidence"
    checkpoint.write_bytes(b"changed frozen model bytes")
    with pytest.raises(ValueError, match="Frozen input changed"):
        diagnostic.check(root)
    checkpoint.write_bytes(b"actual frozen model bytes")
    (root/"decoded.json").write_text("{}")
    with pytest.raises(ValueError, match="Prepared artifact changed"):
        diagnostic.check(root)
