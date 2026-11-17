"""Consumer-visible CPU checks; real tiny Qwen suffixes and CPU Adam updates."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2RMSNorm, Qwen2RotaryEmbedding

from experiments.structure_net import jspace_3b_recovery_session as runtime
from experiments.structure_net import jspace_3b_recovery_study_eval as evaluation
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack


def record(index=0, *, split="train", response_ids=(7, 8, 15), task="math"):
    return runtime.data.ResponseRecord(example_id=f"example{index}", group_id=f"group{index}", split=split,
        task=task, prompt_ids=(1, 3 + index % 3), response_ids=response_ids, finish_reason="eos", eos_token_id=15,
        provenance=dict(source_uri="fixture://train", source_revision="v1", tokenizer_sha256="tokenizer",
            chat_template_sha256="template", prompt_provenance="reference", response_provenance="reference",
            assistant_boundary_verified=True))


def corpus_file(tmp_path):
    path = tmp_path / "train.jsonl"
    if not path.exists():
        rows = [record(i, task="math" if i % 2 else "code",
                       response_ids=(7, 8, 15) if i % 2 else (6, 5, 8, 15)) for i in range(6)]
        path.write_text("".join(json.dumps(asdict(row)) + "\n" for row in rows))
    return path


def tiny_teacher(seed=31, *, layers=2, cut=1):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=16, hidden_size=8, intermediate_size=12,
        num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=32, attention_dropout=0.)
    cfg._attn_implementation = "eager"
    full = Qwen2Model(cfg).bfloat16().eval().requires_grad_(False)
    teacher = runtime.ExactHistoryTeacher(list(full.layers), full.embed_tokens.weight.detach(), full.norm,
        full.rotary_emb, cut=cut, max_length=12, teacher_id="a" * 40, guard=lambda: None)
    return cfg, full, teacher


def make_session(tmp_path, *, arm="student_history"):
    cfg, full, teacher = tiny_teacher()
    native = deepcopy(full.layers[1])
    basis = torch.linalg.qr(torch.randn(8, 8)).Q[:, :4]
    block = AffineQwenBlock(native, basis, torch.randn(12, 8) / 10,
                            trainable=True, parameter_dtype=torch.bfloat16)
    block.compile_polynomial(dict(anchors=[0, 1], mean=torch.zeros(12), scale=torch.ones(12),
        coefficients={"2": torch.randn(12, 6) / 20}), [10, 11])
    for p in block.parameters():
        p.data = p.data.bfloat16()
    stack = CompactQwenStack([block], geometry_mode="bf16", pad_to_multiple=0, checkpoint_blocks=True)
    norm = Qwen2RMSNorm(8).bfloat16()
    rotary = Qwen2RotaryEmbedding(cfg).eval()
    model = runtime.response.ResponseModel(stack, norm, teacher.embedding, rotary,
        max_length=12, expected_layers=2, native_prefix=teacher.native_prefix)
    corpus = runtime.IndexedRecords(corpus_file(tmp_path), max_length=12, eos_token_ids=[15])
    spec = dict(seed=31, learning_rate=.02, accumulation_examples=3, rollout_max_tokens=4,
                rollout_temperature=.8, eos_token_ids=[15], initial_checkpoint_sha256="b" * 64)
    profile = dict(checkpoint_blocks=True, optimizer_chunk_bytes=64, max_cpu_state_bytes=1024**3,
        token_chunk_size=2, gradient_clip=1., host_reserve_bytes=0, native_prefix="streamed",
        mixture_policy="cumulative loss-bearing-token deficit; whole observed responses")
    profile.update(runtime.derive_token_window((row[1] for row in corpus.rows),
                                               nominal_examples=3, rollout_max_tokens=4))
    profile["retention"] = "zero, milestones and latest two"
    profile["maximum_record_json_bytes"] = max(row[5] for row in corpus.rows)
    identities = dict(plan="plan", corpus=runtime.sha(corpus.path), geometry="fixed", consumed_sha256={})
    session = runtime.RecoverySession(model, teacher, corpus, spec=spec, profile=profile,
                                      identities=identities, arm=arm, guard=lambda: None)
    return session


def close(session):
    session.close()
    session.corpus.close()


def capped_rollout(source=None, *, update=0, checkpoint="c" * 64):
    source = record() if source is None else source
    generator = torch.Generator().manual_seed(19)
    logits = torch.full((16,), -100000.)
    logits[2:5] = 0.
    return runtime.sample_rollout(source, lambda ids: logits, generator=generator,
        checkpoint=checkpoint, update=update, teacher_id="a" * 40, seed=19,
        max_new_tokens=4, max_length=12, temperature=.8, eos_token_ids=[15])


def test_reference_completion_and_partial_rollout_mask_are_distinct():
    incomplete = replace(record(), finish_reason="length", response_ids=(7, 8), eos_token_id=None)
    with pytest.raises(ValueError, match="Reference response is incomplete"):
        runtime.session_batch(incomplete, max_length=12, eos_token_ids=[15], update=0,
                              checkpoint="c" * 64, teacher_id="a" * 40)
    sampled = capped_rollout()
    assert sampled.finish_reason == "length" and sampled.eos_token_id is None
    assert 15 not in sampled.response_ids and len(sampled.response_ids) == 4
    batch = runtime.session_batch(sampled, max_length=12, eos_token_ids=[15], update=0,
                                   checkpoint="c" * 64, teacher_id="a" * 40)
    assert batch["loss_mask"].nonzero().tolist() == [[0, 1], [0, 2], [0, 3], [0, 4]]
    assert batch["targets"][batch["loss_mask"]].tolist() == list(sampled.response_ids)
    assert runtime.nal.validate_batch(batch) == 4
    assert not batch["loss_mask"][0, -1]
    with pytest.raises(ValueError, match="Stale rollout"):
        runtime.session_batch(sampled, max_length=12, eos_token_ids=[15], update=1,
                              checkpoint="c" * 64, teacher_id="a" * 40)


def test_categorical_sampling_replays_rng_and_preserves_natural_eos():
    sampled = capped_rollout()
    repeated = capped_rollout()
    assert sampled.response_ids == repeated.response_ids
    assert sampled.provenance["rng_before"] == repeated.provenance["rng_before"]
    assert sampled.provenance["rng_before"] != sampled.provenance["rng_after"]
    assert sampled.provenance["decoding"]["method"] == "categorical_full_vocabulary"
    eos_logits = torch.full((16,), -100000.)
    eos_logits[15] = 0.
    natural = runtime.sample_rollout(record(), lambda ids: eos_logits,
        generator=torch.Generator().manual_seed(1), checkpoint="c" * 64, update=0,
        teacher_id="a" * 40, seed=1, max_new_tokens=4, max_length=12, temperature=.8, eos_token_ids=[15])
    assert natural.response_ids == (15,) and natural.finish_reason == "eos"
    assert natural.eos_token_id == 15 and not natural.provenance["partial_history"]


def test_token_mixture_uses_loss_bearing_tokens_not_example_counts():
    reference = rollout = clean_cases = sampled_cases = 0
    for _ in range(200):
        if runtime.want_rollout("student_history", reference, rollout):
            rollout += 7
            sampled_cases += 1
        else:
            reference += 59
            clean_cases += 1
    assert abs(rollout - .25 * (rollout + reference)) <= 59
    assert sampled_cases > clean_cases
    assert not runtime.want_rollout("reference", reference, 0)


def test_family_sampler_restores_next_examples_and_replay_counts(tmp_path):
    corpus = runtime.IndexedRecords(corpus_file(tmp_path), max_length=12, eos_token_ids=[15])
    try:
        sampler = runtime.FamilySampler(corpus, 1)
        for _ in range(7):
            sampler.next()
        saved = deepcopy(sampler.state())
        expected = [sampler.next() for _ in range(20)]
        resumed = runtime.FamilySampler(corpus, 1)
        resumed.restore(saved)
        assert [resumed.next() for _ in range(20)] == expected
        assert resumed.state() == sampler.state()
    finally:
        corpus.close()


def test_exact_native_boundary_and_normalized_hidden_match_hf_and_changed_history():
    _, full, teacher = tiny_teacher()
    ids = torch.tensor([[1, 3, 7, 8, 15]])
    with torch.no_grad():
        expected = full(input_ids=ids, output_hidden_states=True, use_cache=False)
        boundary, hidden = teacher.forward(ids, full=True)
    torch.testing.assert_close(boundary, expected.hidden_states[1], atol=0., rtol=0.)
    torch.testing.assert_close(hidden, expected.last_hidden_state, atol=0., rtol=0.)
    changed_ids = ids.clone()
    changed_ids[0, 2] = 9
    changed_boundary, changed_hidden = teacher.forward(changed_ids, full=True)
    assert not torch.equal(boundary, changed_boundary)
    assert not torch.equal(hidden, changed_hidden)
    assert teacher.validate_unchanged()["unchanged"]
    with torch.no_grad():
        teacher.norm.weight[0] += 1
    with pytest.raises(ValueError, match="teacher weights"):
        teacher.validate_unchanged()


def test_teacher_rejects_changed_exact_history_after_capture():
    _, _, teacher = tiny_teacher()
    batch = runtime.session_batch(record(), max_length=12, eos_token_ids=[15], update=0,
                                  checkpoint="b" * 64, teacher_id=teacher.teacher_id)
    teacher.prepare(batch)
    positions = batch["loss_mask"].nonzero()
    logits = teacher(batch, positions)
    assert logits.shape == (3, 16)
    batch["input_ids"][0, 0] = 2
    with pytest.raises(ValueError, match="history mutated"):
        teacher(batch, positions)


def test_real_partial_kd_updates_without_positive_student_labels(tmp_path):
    session = make_session(tmp_path)
    try:
        sampled = capped_rollout(checkpoint=session.generating_checkpoint)
        batch = runtime.session_batch(sampled, max_length=12, eos_token_ids=[15], update=0,
            checkpoint=session.generating_checkpoint, teacher_id=session.teacher.teacher_id)
        session.teacher.prepare(batch)
        hidden = session.model.hidden_forward(batch)
        values = runtime.nal.chunked_backward(hidden, session.model.readout, session.teacher, batch, session.loss_config)
        assert values["supervised_tokens"] == 4
        assert values["loss"] == values["forward_kl"]
        assert all(session.optimizer.received[p] == 1 for _, p in session.named)
        before = runtime.offload.fingerprint(session.named)
        step = session.optimizer.step(max_grad_norm=1.)
        assert step["updated_scalars"] == sum(p.numel() for _, p in session.named)
        assert runtime.offload.fingerprint(session.named) != before
        assert session.unchanged_proof()["geometry"]["unchanged"]
    finally:
        close(session)


def test_atomic_full_resume_replays_real_mixed_windows_and_fp32_moments(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU runtime initialized CUDA"))
    session = make_session(tmp_path)
    try:
        zero = session.save(tmp_path / "baseline/update0", publish=tmp_path / "latest.json")
        zero_manifest = runtime.authenticate_checkpoint(zero["directory"], zero["manifest_sha256"])
        assert zero_manifest["optimizer_initialization"] == "new from model-only initializer"
        optimizer_id = id(session.optimizer)
        first = session.train_window()
        one = session.save(tmp_path / "baseline/update1", parent=zero, publish=tmp_path / "latest.json")
        session.train_window()
        expected = session.save(tmp_path / "baseline/update2", parent=one, publish=tmp_path / "latest.json")
        assert id(session.optimizer) == optimizer_id
        assert first["gradients_present_for_every_parameter"]
        assert session.reference_tokens > 0 and session.rollout_tokens > 0
        assert all(s["step"] == 2 for s in session.optimizer.state.values())
        assert all(s[k].dtype == torch.float32 for s in session.optimizer.state.values()
                   for k in ("master", "exp_avg", "exp_avg_sq"))
        assert any(torch.count_nonzero(s["exp_avg"]) for s in session.optimizer.state.values())
        fixed = session.fixed
    finally:
        close(session)
    resumed = make_session(tmp_path)
    try:
        resumed.restore(one)
        assert resumed.updates == 1 and resumed.fixed == fixed
        assert all(s["step"] == 1 for s in resumed.optimizer.state.values())
        resumed.train_window()
        actual = resumed.save(tmp_path / "resumed/update2", parent=one)
        proof = runtime.compare_checkpoints(expected, actual)
        assert proof["passed"] and proof["next_examples_rng_and_loss_masks_equal"]
        assert resumed.unchanged_proof()["teacher"]["unchanged"]
    finally:
        close(resumed)


def test_partial_checkpoint_failure_keeps_previous_commit_and_detects_tamper(tmp_path, monkeypatch):
    session = make_session(tmp_path)
    try:
        initial = session.save(tmp_path / "checkpoints/update0", publish=tmp_path / "latest.json")
        previous = (tmp_path / "latest.json").read_bytes()
        session.train_window()
        def fail_checkpoint(directory):
            Path(directory).mkdir()
            raise OSError("injected storage interruption")
        monkeypatch.setattr(session.optimizer, "save_checkpoint", fail_checkpoint)
        with pytest.raises(OSError, match="storage interruption"):
            session.save(tmp_path / "checkpoints/update1", parent=initial, publish=tmp_path / "latest.json")
        assert (tmp_path / "latest.json").read_bytes() == previous
        assert not (tmp_path / "checkpoints/update1").exists()
        runtime.authenticate_checkpoint(initial["directory"], initial["manifest_sha256"])
        control = Path(initial["directory"]) / "control.pt"
        with control.open("ab") as stream:
            stream.write(b"corruption")
        with pytest.raises(ValueError, match="payload changed"):
            runtime.authenticate_checkpoint(initial["directory"], initial["manifest_sha256"])
    finally:
        close(session)


def test_changed_live_parameter_membership_is_rejected(tmp_path):
    session = make_session(tmp_path)
    try:
        session.model.norm.weight = torch.nn.Parameter(session.model.norm.weight.detach().clone())
        with pytest.raises(ValueError, match="identities changed"):
            session.train_window()
    finally:
        close(session)


def test_generation_rows_are_consumable_by_pinned_evaluator():
    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens):
            return " ".join(map(str, ids))
    source = record(split="development")
    decoding = dict(max_new_tokens=4, temperature=0., do_sample=False, eos_token_ids=[15])
    generated = dict(output_ids=[2, 3, 4, 5], stop_reason="token_limit", max_new_tokens=4)
    row = runtime.generation_row(source, generated, role="student", checkpoint="b" * 64,
        tokenizer=Tokenizer(), decoding=decoding, grader_identity="grader", geometry="geometry")
    plan = dict(spec_snapshot=dict(native_model_revision="a" * 40), decoding=decoding, grader_identity="grader")
    evaluation.validate_response(row, source, plan)
    assert row["stop_reason"] == "length" and row["input_ids"] == list(source.prompt_ids)
    assert row["output_ids"][-1] != 15


def test_admitted_static_corpus_rejects_partial_references_and_nontrain(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(asdict(replace(record(), response_ids=(7,), finish_reason="length", eos_token_id=None))) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        runtime.IndexedRecords(path, max_length=12, eos_token_ids=[15])
    path.write_text(json.dumps(asdict(record(split="development"))) + "\n")
    with pytest.raises(ValueError, match="split differs"):
        runtime.IndexedRecords(path, max_length=12, eos_token_ids=[15])


def test_shared_token_window_derivation_and_complete_record_overshoot(tmp_path):
    contract = runtime.derive_token_window([3, 4, 3, 4], nominal_examples=3, rollout_max_tokens=4)
    assert contract["window_target_tokens"] == 11
    assert contract["maximum_window_tokens"] == 14
    for arm in ("reference", "student_history"):
        session = make_session(tmp_path, arm=arm)
        try:
            window = session.train_window()
            assert 11 <= window["supervised_tokens"] <= 14
            assert window["supervised_tokens"] == sum(len(r["response_ids"]) for r in window["records"])
        finally:
            close(session)


def test_committed_directory_before_latest_publication_is_adopted(tmp_path):
    session = make_session(tmp_path)
    output = tmp_path / "reference"
    try:
        initial = session.save(output / "checkpoints/update_00000000", publish=output / "latest.json")
        session.train_window()
        orphan = session.save(output / "checkpoints/update_00000001", parent=initial)
        assert runtime.read_json(output / "latest.json")["update"] == 0
        adopted = runtime.recover_commit_pointer(output, identities=session.identities)
        assert adopted["manifest_sha256"] == orphan["manifest_sha256"]
        assert runtime.read_json(output / "latest.json")["update"] == 1
    finally:
        close(session)


def test_retention_preserves_latest_two_zero_and_generation_references(tmp_path):
    session = make_session(tmp_path)
    output = tmp_path / "reference"
    try:
        committed = session.save(output / "checkpoints/update_00000000", publish=output / "latest.json")
        for update in range(1, 5):
            session.train_window()
            committed = session.save(output / "checkpoints" / f"update_{update:08d}",
                                     parent=committed, publish=output / "latest.json")
            if update == 1:
                runtime.write_json(output / "generation/tokens_11/result.json",
                                   dict(checkpoint=committed["manifest_sha256"]))
        receipt = runtime.retain_checkpoints(output, identities=session.identities)
        retained = {path.name for path in (output / "checkpoints").iterdir()}
        assert retained == {"update_00000000", "update_00000001", "update_00000003", "update_00000004"}
        assert receipt["pruned_checkpoint_bytes"] > 0
        history = runtime.authenticate_history(output / "history/update_00000002")
        assert history["status"] == "window_history_only_not_resumable"
        archived = runtime.read_json(output / "history/update_00000002/window.json")
        assert archived["records"] and archived["exact_teacher_bindings"]
        assert runtime.retain_checkpoints(output, identities=session.identities)["pruned_checkpoint_bytes"] == receipt["pruned_checkpoint_bytes"]
    finally:
        close(session)


def test_pilot_cost_projection_does_not_double_count_prefix():
    timings = dict(native_prefix_seconds=4., rollout_seconds=6., capture_seconds=5.,
        teacher_transfer_seconds=2., backward_seconds=3., optimizer_seconds=1.,
        residency_setup_seconds=1., residency_release_seconds=0.)
    result = runtime.project_pilot_cost([dict(supervised_tokens=100, timings=timings)],
        [dict(checkpoint_seconds=2.)], dict(retention="bounded"))
    assert result["measured_seconds"]["rollout_excluding_native_prefix_seconds"] == 2.
    assert result["projected_total_hours"]["100000"] == 18 * 1000 / 3600
    assert result["projected_seconds"]["1000000"]["native_prefix_seconds"] == 40000.


def test_interrupted_owned_pruning_resumes_without_touching_latest(tmp_path, monkeypatch):
    session = make_session(tmp_path)
    output = tmp_path / "reference"
    try:
        committed = session.save(output / "checkpoints/update_00000000", publish=output / "latest.json")
        for update in range(1, 4):
            session.train_window()
            committed = session.save(output / "checkpoints" / f"update_{update:08d}",
                                     parent=committed, publish=output / "latest.json")
        original_remove = runtime.shutil.rmtree
        def interrupted_remove(directory):
            next((Path(directory) / "optimizer").glob("*.pt")).unlink()
            raise OSError("interrupted owned prune")
        monkeypatch.setattr(runtime.shutil, "rmtree", interrupted_remove)
        with pytest.raises(OSError, match="owned prune"):
            runtime.retain_checkpoints(output, identities=session.identities)
        assert runtime.read_json(output / "retention.json")["status"] == "pruning"
        monkeypatch.setattr(runtime.shutil, "rmtree", original_remove)
        runtime.retain_checkpoints(output, identities=session.identities)
        assert not (output / "checkpoints/update_00000001").exists()
        assert runtime.read_json(output / "retention.json")["status"] == "completed"
        runtime.authenticate_checkpoint(committed["directory"], committed["manifest_sha256"])
    finally:
        close(session)


@pytest.mark.parametrize("resident_count", [0, 1, 2])
def test_partial_residency_preserves_prefix_and_releases_before_capture(resident_count):
    _, _, teacher = tiny_teacher(layers=3, cut=2)
    ids = torch.tensor([[1, 3, 7, 8, 15]])
    expected = teacher.forward(ids, full=False)
    original = [[p.data_ptr() for p in block.parameters()] for block in teacher.blocks]
    with teacher.prefix_residency(resident_count):
        actual = teacher.forward(ids, full=False)
        torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
        if resident_count:
            with pytest.raises(ValueError, match="Release rollout residency"):
                teacher.forward(ids, full=True)
    assert teacher.resident_count == 0
    assert original == [[p.data_ptr() for p in block.parameters()] for block in teacher.blocks]
    assert teacher.validate_unchanged()["unchanged"]
    assert torch.isfinite(teacher.forward(ids, full=True)[1]).all()


def test_partial_residency_rolls_back_when_admission_fails_mid_bank():
    _, _, teacher = tiny_teacher(layers=3, cut=2)
    pointers = [[p.data_ptr() for p in block.parameters()] for block in teacher.blocks]
    calls = 0
    def guard():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("reservation revoked during second block admission")
    teacher.guard = guard
    with pytest.raises(RuntimeError, match="reservation revoked"):
        with teacher.prefix_residency(2):
            pytest.fail("Incomplete resident bank was exposed")
    assert teacher.resident_count == 0
    assert pointers == [[p.data_ptr() for p in block.parameters()] for block in teacher.blocks]
    assert teacher.validate_unchanged()["unchanged"]


def test_residency_qualifies_context_boundary_when_complete_records_are_shorter(tmp_path):
    session = make_session(tmp_path)
    try:
        source = record(response_ids=(7, 8, 6, 5, 8, 7, 6, 8, 7, 15))
        provenance = dict(source.provenance, source_complete_tokens_sha256=runtime.data.digest(source.input_ids),
            source_complete_token_count=len(source.input_ids), source_row=0, historical_lineage={},
            derivation="exact_prefix_crop_for_geometry_only")
        calibration = tmp_path / "calibration.jsonl"
        calibration.write_text(json.dumps(dict(example_id=source.example_id, group_id=source.group_id,
            split="train", input_ids=source.input_ids, provenance=provenance)) + "\n")
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(dict(schema="jspace.recovery-reset.corpus.v1",
            calibration_records=str(calibration), train_records=str(session.corpus.path),
            sha256={str(calibration): runtime.sha(calibration)})))
        session.spec.update(corpus_manifest=str(manifest), target_context=12)
        session.profile.update(native_prefix="resident", resident_prefix_layers=1)
        before = runtime.offload.fingerprint(session.named)
        proof = runtime.qualify_prefix_residency(session)
        assert [row["tokens"] for row in proof] == [2, 5, 12]
        assert all(row["bit_exact"] for row in proof)
        assert proof[-1]["calibration"]["calibration_sha256"] == runtime.sha(calibration)
        assert before == runtime.offload.fingerprint(session.named)
        assert session.teacher.validate_unchanged()["unchanged"]
    finally:
        close(session)
