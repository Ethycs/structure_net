"""CPU numerical/lifecycle evidence; no toy loss is a model-quality claim."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F


def load_module(name, filename):
    path = Path(__file__).resolve().parents[2] / "src/neural_architecture_lab" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The NCM environment intentionally lacks the optional legacy NAL analyzer stack.
runtime = load_module("nal_training_test_runtime", "distillation_training.py")
data = load_module("nal_training_test_data", "distillation_data.py")


class TinyCausalLM(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.embedding = nn.Embedding(11, 6)
        self.blocks = nn.ModuleList([nn.Linear(6, 6), nn.Linear(6, 6)])
        self.norm = nn.LayerNorm(6)
        self.register_buffer("geometry", torch.linspace(.8, 1.2, 6))
        self.to(dtype=dtype)

    def hidden(self, batch):
        h = self.embedding(batch["input_ids"])
        h = h * batch["attention_mask"].to(h.dtype)[..., None]
        denominator = torch.arange(1, h.shape[1] + 1, device=h.device, dtype=h.dtype)[None, :, None]
        for block in self.blocks:
            h = torch.tanh(block(h.cumsum(1) / denominator)) * self.geometry
        return self.norm(h)

    def readout(self, hidden):
        return F.linear(hidden, self.embedding.weight)  # Tests tied head+trunk gradients.


def examples():
    provenance = dict(source_uri="fixture://tiny", source_revision="v1", tokenizer_sha256="tok",
        chat_template_sha256="template", prompt_provenance="fixture", response_provenance="human",
        assistant_boundary_verified=True)
    return [data.ResponseRecord("one", "group1", "train", "instruction", (1, 2), (5, 6, 8), provenance, "eos", 8),
            data.ResponseRecord("two", "group2", "train", "instruction", (2, 3, 4), (7, 8), provenance, "eos", 8)]


def batch(records=None):
    return data.response_batch(examples() if records is None else records, pad_token_id=0, max_length=32)


def models(dtype=torch.float32):
    torch.manual_seed(31); student = TinyCausalLM(dtype)
    torch.manual_seed(72); teacher = TinyCausalLM(dtype)
    return student, teacher


def provider(teacher, temperature=1.):
    return runtime.FrozenTeacherProvider(teacher, teacher.hidden, teacher.readout,
        teacher_id="original-teacher", temperature=temperature)


def identities(student):
    return dict(teacher="original-teacher", student_initializer=runtime.module_fingerprint(student),
        tokenizer="tok", chat_template="template", data=data.dataset_fingerprint(examples()))


@pytest.mark.parametrize("objective,ce_weight", [("forward_kl", 0.), ("forward_kl", .3), ("generalized_jsd", .2)])
@pytest.mark.parametrize("chunk", [1, 2, 99])
def test_bounded_chunks_match_dense_value_and_every_tied_trunk_head_gradient(objective, ce_weight, chunk):
    student, teacher = models(torch.float64); reference = deepcopy(student); b = batch()
    config = runtime.DistillationConfig(objective=objective, temperature=1.7, ce_weight=ce_weight, token_chunk_size=chunk)
    tp = provider(teacher, config.temperature); tp.prepare(b)
    positions = b["loss_mask"].nonzero()
    target = b["targets"][b["loss_mask"]]
    dense_logits = reference.readout(reference.hidden(b)[b["loss_mask"]])
    dense = runtime.loss_vectors(dense_logits, tp(b, positions), target, config)
    dense["loss"].mean().backward()
    result = runtime.chunked_backward(student.hidden(b), student.readout, tp, b, config)
    assert result["supervised_tokens"] == 5
    assert result["loss"] / 5 == pytest.approx(float(dense["loss"].mean()), abs=1e-12)
    for (name, actual), (other, expected) in zip(student.named_parameters(), reference.named_parameters()):
        assert name == other and actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-12, rtol=2e-12)
    assert tp.validate_unchanged() and all(p.grad is None for p in teacher.parameters())


def test_full_response_mask_includes_first_token_eos_and_rejects_holes_or_dev():
    b = batch()
    assert runtime.validate_batch(b) == 5
    assert b["targets"][b["loss_mask"]].tolist() == [5, 6, 8, 7, 8]
    altered = dict(b, loss_mask=b["loss_mask"].clone()); altered["loss_mask"][0, 1] = False
    with pytest.raises(ValueError, match="every eligible"):
        runtime.validate_batch(altered)
    with pytest.raises(ValueError, match="TRAIN"):
        runtime.validate_batch(batch([replace(examples()[0], split="development")]))


def test_real_update_accumulation_teacher_offload_hook_save_fresh_reload_and_complete_budget(tmp_path):
    student, teacher = models(); identity = identities(student); tp = provider(teacher)
    original = runtime.module_fingerprint(teacher); order = []; b = batch()
    prepare = tp.prepare
    def prepare_first(value): order.append("teacher_prepare"); prepare(value)
    tp.prepare = prepare_first
    def hidden(value): order.append("student_forward"); return student.hidden(value)
    config = runtime.DistillationConfig(token_chunk_size=2, accumulation_batches=2,
        max_supervised_tokens=11, max_updates=4, grad_clip_norm=100.)
    optimizer = torch.optim.AdamW(student.parameters(), lr=.01, weight_decay=0.)
    trainer = runtime.DistillationTrainer(student, optimizer, hidden, student.readout, tp,
        config=config, identities=identity)
    events = []
    result = trainer.fit([b, b, b], tmp_path / "run", progress=events.append)
    assert result["updates"] == 1 and result["supervised_tokens"] == 10
    assert result["stop_reason"] == "token_budget" and result["response_tokens"] == 10
    assert result["fixed_state_unchanged"] and result["teacher_unchanged"] and result["in_process_reload_exact"]
    assert result["changed_parameters"] == list(dict(student.named_parameters()))
    assert order == ["teacher_prepare", "student_forward"] * 2
    assert events[0]["phase"] == "initialized" and events[-1]["phase"] == "training"
    assert runtime.module_fingerprint(teacher) == original
    assert all(p.grad is None for p in teacher.parameters())
    fresh = TinyCausalLM(); checkpoint = result["final_checkpoint"]
    runtime.load_checkpoint(checkpoint["path"], fresh, expected_sha256=checkpoint["sha256"], identities=identity)
    assert runtime.module_fingerprint(fresh) == runtime.module_fingerprint(student)
    torch.testing.assert_close(fresh.readout(fresh.hidden(b)), student.readout(student.hidden(b)), rtol=0, atol=0)
    assert result["confirmed"] is False and result["quality_status"] == "systems_only"
    with pytest.raises(ValueError, match="digest"):
        runtime.load_checkpoint(checkpoint["path"], fresh, expected_sha256="wrong", identities=identity)


def test_reference_t1_and_separate_ce_jsd_and_topk_tail_approximation():
    logits = torch.tensor([[.2, -1., 2., .7]], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([[1., 2., -.5, .3]], dtype=torch.float64)
    targets = torch.tensor([3]); config = runtime.DistillationConfig()
    exact = runtime.loss_vectors(logits, teacher, targets, config)["loss"]
    expected = F.kl_div(F.log_softmax(logits, -1), F.softmax(teacher, -1), reduction="none").sum(-1)
    torch.testing.assert_close(exact, expected)
    p = F.softmax(teacher, -1); ids = torch.tensor([[1, 0]])
    top = p.gather(1, ids); tail = 1 - top.sum(-1)
    approximate = runtime.topk_tail_kl(logits, ids, top, tail)
    q = F.softmax(logits, -1); qs = q.gather(1, ids); qt = q[:, 2:].sum(-1)
    grouped = (top * (top.log() - qs.log())).sum(-1) + tail * (tail.log() - qt.log())
    torch.testing.assert_close(approximate, grouped)
    assert float(approximate) < float(exact)  # Coarsening loses within-tail information.
    all_ids = torch.tensor([[0, 1, 2, 3]])
    full = runtime.topk_tail_kl(logits, all_ids, p, torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(full, exact)
    torch.testing.assert_close(torch.autograd.grad(full.sum(), logits, retain_graph=True)[0],
                               torch.autograd.grad(exact.sum(), logits)[0])
    with pytest.raises(ValueError, match="distinct"):
        runtime.topk_tail_kl(logits, torch.tensor([[1, 1]]), top, tail)


def test_nonfinite_gradient_or_changed_fixed_state_cannot_be_success(tmp_path):
    student, teacher = models(); b = batch(); tp = provider(teacher)
    trainer = runtime.DistillationTrainer(student, torch.optim.SGD(student.parameters(), lr=.01),
        student.hidden, student.readout, tp, config=runtime.DistillationConfig(max_updates=1), identities=identities(student))
    next(student.parameters()).register_hook(lambda gradient: gradient * float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        trainer.fit([b], tmp_path / "bad")
    assert (tmp_path / "bad/failure.json").exists() and not (tmp_path / "bad/result.json").exists()
    student, teacher = models(); tp = provider(teacher)
    original_hidden = student.hidden
    def mutate(batch):
        student.geometry.add_(.1)
        return original_hidden(batch)
    trainer = runtime.DistillationTrainer(student, torch.optim.SGD(student.parameters(), lr=.01),
        mutate, student.readout, tp, config=runtime.DistillationConfig(max_updates=1), identities=identities(student))
    with pytest.raises(ValueError, match="Fixed student"):
        trainer.fit([b], tmp_path / "geometry")


def test_initializer_and_fresh_optimizer_guards(tmp_path):
    student, teacher = models(); tp = provider(teacher); identity = identities(student)
    optimizer = torch.optim.AdamW(student.parameters(), lr=.01)
    for p in student.parameters(): p.grad = torch.ones_like(p)
    optimizer.step(); identity["student_initializer"] = runtime.module_fingerprint(student)
    with pytest.raises(ValueError, match="Fresh optimizer"):
        runtime.DistillationTrainer(student, optimizer, student.hidden, student.readout, tp,
            config=runtime.DistillationConfig(), identities=identity)
    identity["student_initializer"] = "wrong"
    with pytest.raises(ValueError, match="initializer"):
        runtime.DistillationTrainer(student, torch.optim.SGD(student.parameters(), lr=.01),
            student.hidden, student.readout, tp, config=runtime.DistillationConfig(), identities=identity)


def test_short_rollouts_exact_teacher_prefixes_disk_targets_and_real_recovery_update(tmp_path):
    student, teacher = models(); source = [examples()[0]]; observed = []
    initial = runtime.module_fingerprint(student)
    def next_logits(ids):
        observed.append(ids[0].tolist())
        value = torch.zeros(11); value[4 if len(observed) == 1 else 8] = 9
        return value
    rolled, receipt = runtime.collect_rollouts(source, next_logits, student_checkpoint=initial,
        teacher_id="original-teacher", round_id="round0", eos_token_id=8, max_new_tokens=4, max_length=32)
    assert observed == [[1, 2], [1, 2, 4]]
    assert rolled[0].response_ids == (4, 8) and rolled[0].finish_reason == "eos"
    assert receipt["supervised_tokens"] == 2
    b = batch(rolled); tp = provider(teacher); tp.prepare(b); positions = b["loss_mask"].nonzero()
    targets = tp(b, positions)
    for i, position in enumerate(positions[:, 1]):
        # Independently score exactly the prefix preceding this generated token.
        prefix = rolled[0].input_ids[:int(position) + 1]
        prefix_batch = dict(input_ids=torch.tensor([prefix]), attention_mask=torch.ones(1, len(prefix), dtype=torch.long))
        expected = teacher.readout(teacher.hidden(prefix_batch)[:, -1])
        torch.testing.assert_close(targets[i], expected[0])
    cache_dir = tmp_path / "targets"
    data.write_target_cache(cache_dir, rolled[0], teacher_id="original-teacher", temperature=1.,
        vocabulary_size=11, logits=targets)
    store = data.TeacherTargetStore({data.record_fingerprint(rolled[0]): cache_dir},
        teacher_id="original-teacher", temperature=1., vocabulary_size=11)
    config = runtime.DistillationConfig(max_updates=1, token_chunk_size=1, rollout_policy="offline_round",
        rollout_student_checkpoint=initial, rollout_round_id="round0", max_stale_updates=0)
    result = runtime.DistillationTrainer(student, torch.optim.AdamW(student.parameters(), lr=.01),
        student.hidden, student.readout, store, config=config, identities=identities(student)).fit([b], tmp_path / "rollout_run")
    assert result["updates"] == 1 and result["supervised_tokens"] == 2
    assert result["changed_parameters"] and tp.validate_unchanged()
    altered = replace(rolled[0], response_ids=(6, 8))
    with pytest.raises(ValueError, match="exact record"):
        store(batch([altered]), positions)


def test_rollout_cap_natural_repetition_and_explicit_staleness_guard(tmp_path):
    student, teacher = models(); initial = runtime.module_fingerprint(student)
    def constant(ids):
        logits = torch.zeros(11); logits[4] = 1; return logits
    rolled, receipt = runtime.collect_rollouts([examples()[0]], constant, student_checkpoint=initial,
        teacher_id="original-teacher", round_id="r", eos_token_id=8, max_new_tokens=12, max_length=32)
    assert rolled[0].response_ids == (4,) * 12 and rolled[0].finish_reason == "length"
    assert receipt["records"][0]["repetition"]["flagged"]
    assert not runtime.repetition_report([1, 2, 1, 3, 1, 4, 1, 5])["flagged"]
    assert runtime.validate_batch(batch(rolled)) == 12
    config = runtime.DistillationConfig(max_updates=2, rollout_policy="offline_round",
        rollout_student_checkpoint=initial, rollout_round_id="r", max_stale_updates=0)
    tp = provider(teacher)
    trainer = runtime.DistillationTrainer(student, torch.optim.SGD(student.parameters(), lr=.001),
        student.hidden, student.readout, tp, config=config, identities=identities(student))
    with pytest.raises(ValueError, match="stale-update"):
        trainer.fit([batch(rolled), batch(rolled)], tmp_path / "stale")
    assert (tmp_path / "stale/failure.json").exists()
    with pytest.raises(ValueError, match="TRAIN"):
        runtime.collect_rollouts([replace(examples()[0], split="test")], constant,
            student_checkpoint=initial, teacher_id="original-teacher", round_id="r",
            eos_token_id=8, max_new_tokens=4)
    with pytest.raises(ValueError, match="reservation"):
        runtime.collect_rollouts([examples()[0]], constant, student_checkpoint=initial,
            teacher_id="original-teacher", round_id="r", eos_token_id=8, device="cuda:0")


def test_real_student_generated_round_then_frozen_teacher_recovery(tmp_path):
    student, teacher = models(); student.eval(); identity = identities(student)
    seen = []
    def actual_next_logits(ids):
        seen.append(ids.clone())
        context = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        return student.readout(student.hidden(context)[:, -1])[0]
    rolled, receipt = runtime.collect_rollouts([examples()[0]], actual_next_logits,
        student_checkpoint=identity["student_initializer"], teacher_id=identity["teacher"],
        round_id="real", eos_token_id=8, max_new_tokens=6, max_length=32)
    assert 1 <= len(seen) == len(rolled[0].response_ids) <= 6
    for i, ids in enumerate(seen):
        assert ids[0].tolist() == list(rolled[0].prompt_ids + rolled[0].response_ids[:i])
    config = runtime.DistillationConfig(max_updates=1, token_chunk_size=2,
        rollout_policy="offline_round", rollout_student_checkpoint=identity["student_initializer"],
        rollout_round_id="real", max_stale_updates=0)
    tp = provider(teacher)
    result = runtime.DistillationTrainer(student, torch.optim.AdamW(student.parameters(), lr=.002),
        student.hidden, student.readout, tp, config=config, identities=identity).fit([batch(rolled)], tmp_path / "real")
    assert result["supervised_tokens"] == receipt["supervised_tokens"]
    assert result["teacher_unchanged"] and result["changed_parameters"] and result["in_process_reload_exact"]
