"""Persistent, exact-history Qwen recovery; GPU work requires an external lease.

Reference records must be complete. Capped sampled histories are explicitly
partial, but supervise *every observed token*, including a real EOS if present.
NAL's causal validator/chunked full-vocabulary loss accepts precisely that mask;
we do not fabricate EOS, disguise the record, patch its validator, or use its
single-use/offline-round trainer as an on-policy session.

Each optimizer window is a transaction. A crash discards its uncommitted cursor,
rollouts, gradients and RNG advancement; the previous authenticated checkpoint
is retained. Historical update46 is a model-only initializer, never an optimizer
resume. A scientific arm subsequently restores the named FP32 Adam chunks.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from experiments.neural_architecture_lab.distillation_recovery import module
from . import jspace_qwen3_train as offload
from . import jspace_qwen_parallel_model as native
from . import jspace_3b_response_train as response
from .jspace_3b_mmlu_redux import host_weights
from .jspace_3b_response_cache import reservation
from .jspace_3b_response_eval import generate_prompt
from .jspace_3b_staged_campaign import local_import_closure

nal = module("distillation_training")
data = module("distillation_data")
SCHEMA = "jspace.recovery-reset.session.v1"
CHECKPOINT_SCHEMA = "jspace.recovery-reset.session-checkpoint.v1"
ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)
    sync_directory(path.parent)


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU preparation/check requires CUDA_VISIBLE_DEVICES=''")


def validate_spec(spec):
    if (spec.get("schema") != "jspace.recovery-reset.spec.v1"
            or spec["temperature"] != 1. or spec["ce_weight"] != 0.
            or spec["rollout_fraction"] != .25 or spec["target_context"] != 2048
            or spec["accumulation_examples"] < 2 or spec["pilot_examples"] < 2
            or spec["learning_rate"] <= 0 or spec["rollout_temperature"] <= 0
            or spec["rollout_max_tokens"] < 1):
        raise ValueError("Frozen forward-KL/T1/CE0, 75/25 and 2048-context study required")


def derive_token_window(response_lengths, *, nominal_examples, rollout_max_tokens):
    lengths = list(response_lengths)
    if not lengths or min(lengths) < 1 or nominal_examples < 1:
        raise ValueError("Nonempty admitted loss-bearing responses required")
    target = (nominal_examples * sum(lengths) + len(lengths) - 1) // len(lengths)
    maximum = max(max(lengths), rollout_max_tokens)
    return dict(window_target_tokens=target, nominal_clean_examples=nominal_examples,
        maximum_response_tokens=maximum, maximum_window_tokens=target + maximum - 1,
        maximum_window_token_overshoot=target + maximum - 1, maximum_window_examples=target,
        token_window_derivation=dict(method="ceil(nominal_examples * admitted TRAIN mean supervised tokens)",
            train_records=len(lengths), train_supervised_tokens=sum(lengths)))


def prepare(root, spec_path, *, limit_mib=11008, host_reserve_bytes=8 * 1024**3,
            max_cpu_state_bytes=32 * 1024**3, token_chunk_size=8, prefix_mode="streamed", resident_layers=None):
    cpu_only()
    root, spec_path = Path(root).resolve(), Path(spec_path).resolve()
    spec = read_json(spec_path)
    validate_spec(spec)
    if min(limit_mib, max_cpu_state_bytes, token_chunk_size) <= 0 or host_reserve_bytes < 0:
        raise ValueError("Positive resource limits required")
    if prefix_mode not in {"streamed", "resident"}:
        raise ValueError("Declare streamed or resident rollout native-prefix weights")
    cut = read_json(spec["teacher_plan"])["cut"]
    resident_layers = (cut if prefix_mode == "resident" else 0) if resident_layers is None else resident_layers
    if (type(resident_layers) is not int or not 0 <= resident_layers <= cut
            or (prefix_mode == "streamed") != (resident_layers == 0)):
        raise ValueError("Resident layer count must match the declared prefix residency")
    if sha(spec["initial_checkpoint"]) != spec["initial_checkpoint_sha256"]:
        raise ValueError("Historical model-only initializer changed")
    corpus, _ = admit_corpus(dict(spec=spec), max_length=spec["target_context"])
    try:
        batching = derive_token_window((row[1] for row in corpus.rows),
            nominal_examples=spec["accumulation_examples"], rollout_max_tokens=spec["rollout_max_tokens"])
        batching["maximum_record_json_bytes"] = max(row[5] for row in corpus.rows)
        corpus_path = str(corpus.path)
    finally:
        corpus.close()
    # Context source exists only after implementation integration; preparation is
    # deliberately not run from a worker's unfinished shared checkout.
    from . import jspace_3b_context_qualification as geometry
    sources = local_import_closure([Path(__file__), Path(geometry.__file__)])
    sources.update(ROOT / "src/neural_architecture_lab" / (name + ".py")
                   for name in ("distillation_data", "distillation_training"))
    sources.add(ROOT / "experiments/neural_architecture_lab/distillation_recovery.py")
    model_root = Path(spec["native_model_path"])
    model_files = sorted(set(model_root.glob("*.safetensors")) |
                         set(model_root.glob("*.json")) | set(model_root.glob("*.jinja")))
    if not model_files:
        raise ValueError("Pinned local teacher snapshot missing")
    inputs = {str(spec_path): sha(spec_path), spec["teacher_plan"]: sha(spec["teacher_plan"]),
              spec["initial_checkpoint"]: spec["initial_checkpoint_sha256"]}
    inputs.update({str(p): sha(p) for p in model_files})
    inputs.update({spec["corpus_manifest"]: sha(spec["corpus_manifest"]), corpus_path: sha(corpus_path)})
    plan = dict(schema=SCHEMA, spec=spec, spec_path=str(spec_path), spec_sha256=sha(spec_path),
        input_sha256=inputs, source_sha256={str(p): sha(p) for p in sorted(sources)},
        profile=dict(dtype="torch.bfloat16", attention="eager", native_prefix=prefix_mode,
            resident_prefix_layers=resident_layers,
            student_backend="unpacked_differentiable_torch", timing_policy="synchronize timed CUDA prefix and transfer boundaries",
            limit_mib=limit_mib, host_reserve_bytes=host_reserve_bytes,
            max_cpu_state_bytes=max_cpu_state_bytes, token_chunk_size=token_chunk_size,
            optimizer_chunk_bytes=4 * 1024**2, gradient_clip=1., checkpoint_blocks=True,
            scheduler="constant", stale_updates=0, rollout_refresh="every optimizer window",
            mixture_policy="cumulative loss-bearing-token deficit; whole observed responses",
            **batching, retention="declared milestone/zero checkpoints and latest two durable commits; authenticated window history retained",
            resume_atol=0., resume_rtol=0.),
        receipt_paths=dict(pilot="pilot/result.json", train="arms/{arm}/phases/tokens_{target}/result.json",
            generate="arms/{arm}/generation/tokens_{target}/result.json",
            responses="arms/{arm}/generation/tokens_{target}/responses.jsonl"),
        missing_prerequisites=[key for key in ("corpus_manifest", "geometry_manifest")
                               if not Path(spec[key]).is_file()],
        initializer_optimizer="fresh CPU-gradient Adam from historical model-only checkpoint; no inherited moments",
        scientific_status="not_executed")
    plan["fingerprint"] = data.digest(plan)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", plan)
    for source, digest in plan["source_sha256"].items():
        target = root / "producer" / Path(source).relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(target) != digest:
            raise ValueError("Producer snapshot changed during preparation")
    return dict(status="prepared", plan=str(root / "plan.json"), fingerprint=plan["fingerprint"],
                missing_prerequisites=plan["missing_prerequisites"], **plan["receipt_paths"])


def load_plan(root, *, live=True):
    root = Path(root)
    plan = read_json(root / "plan.json")
    body = dict(plan)
    fingerprint = body.pop("fingerprint")
    if plan["schema"] != SCHEMA or data.digest(body) != fingerprint:
        raise ValueError("Session plan fingerprint differs")
    validate_spec(plan["spec"])
    for path, digest in plan["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Session input changed: " + path)
    for path, digest in plan["source_sha256"].items():
        if sha(root / "producer" / Path(path).relative_to(ROOT)) != digest or live and sha(path) != digest:
            raise ValueError("Session source changed: " + path)
    return plan


def validate_record(record, *, eos_token_ids, update=None, checkpoint=None, teacher_id=None):
    if record.split != "train":
        raise ValueError("Only admitted TRAIN records may update")
    rollout = record.provenance["response_provenance"] == "student_rollout"
    eos = set(eos_token_ids)
    if any(t in eos for t in record.response_ids[:-1]):
        raise ValueError("Record continues after an actual EOS")
    if record.finish_reason == "eos":
        if record.response_ids[-1] not in eos or record.eos_token_id != record.response_ids[-1]:
            raise ValueError("Observed EOS identity differs")
    elif record.response_ids[-1] in eos:
        raise ValueError("Observed EOS cannot be reported as a cap/stop")
    if not rollout:
        if record.finish_reason == "length":
            raise ValueError("Reference response is incomplete; no fabricated EOS")
        return len(record.response_ids)
    p = record.provenance
    if (p.get("generating_update") != update or p.get("student_checkpoint") != checkpoint
            or p.get("teacher_id") != teacher_id or p.get("max_stale_updates") != 0
            or p.get("observed_history_sha256") != data.digest(record.input_ids)):
        raise ValueError("Stale rollout or wrong exact-history/checkpoint identity")
    if (p.get("decoding", {}).get("method") != "categorical_full_vocabulary"
            or p.get("partial_history") != (record.finish_reason == "length")):
        raise ValueError("Explicit categorical partial-history policy required")
    if record.finish_reason == "length" and len(record.response_ids) != p["decoding"]["max_new_tokens"]:
        raise ValueError("Capped rollout must contain exactly its actual allowance")
    return len(record.response_ids)


def session_batch(record, *, max_length, eos_token_ids, update, checkpoint, teacher_id):
    count = validate_record(record, eos_token_ids=eos_token_ids, update=update,
                            checkpoint=checkpoint, teacher_id=teacher_id)
    batch = data.response_batch([record], pad_token_id=eos_token_ids[-1], max_length=max_length)
    # NAL validates the full *observed* record, not an invented completed history.
    if nal.validate_batch(batch) != count:
        raise ValueError("Every observed response token must carry loss exactly once")
    return batch


class IndexedRecords:
    """One open JSONL handle, compact offsets, no corpus hidden/logit bank."""
    def __init__(self, path, *, max_length, eos_token_ids, split="train"):
        self.path, self.max_length, self.split = Path(path), max_length, split
        self.rows, self.families, self.exclusions = [], {}, []
        self.tokenizer = self.template = None
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                record = data.ResponseRecord(**json.loads(line))
                if record.split != split:
                    raise ValueError("Corpus split differs from admitted shard")
                if split == "train":
                    validate_record(record, eos_token_ids=eos_token_ids)
                if record.provenance["response_provenance"] == "student_rollout":
                    raise ValueError("Static corpus must not contain stale student rollouts")
                identities = (record.provenance["tokenizer_sha256"], record.provenance["chat_template_sha256"])
                if self.tokenizer is None:
                    self.tokenizer, self.template = identities
                if identities != (self.tokenizer, self.template):
                    raise ValueError("Corpus contains mixed tokenizer/template identities")
                if len(record.input_ids) > max_length:
                    self.exclusions.append(dict(example_id=record.example_id, reason="unsupported_total_length"))
                    continue
                index = len(self.rows)
                self.rows.append((offset, len(record.response_ids), record.example_id, record.group_id, len(record.input_ids), len(line)))
                self.families.setdefault(record.task, []).append(index)
        if not self.rows:
            raise ValueError("No admitted records fit supported context")
        self.stream = self.path.open("rb")

    def __getitem__(self, index):
        self.stream.seek(self.rows[index][0])
        return data.ResponseRecord(**json.loads(self.stream.readline()))

    def close(self):
        self.stream.close()


class FamilySampler:
    """Same balanced TRAIN prompt-family policy in both arms, no DEV access."""
    def __init__(self, corpus, seed):
        self.random = random.Random(seed)
        self.order = {family: list(indices) for family, indices in sorted(corpus.families.items())}
        for indices in self.order.values():
            self.random.shuffle(indices)
        self.cursors = dict.fromkeys(self.order, 0)
        self.epochs = dict.fromkeys(self.order, 0)
        self.family_cursor = 0
        self.presentations = {}

    def next(self):
        family = list(self.order)[self.family_cursor % len(self.order)]
        self.family_cursor += 1
        if self.cursors[family] == len(self.order[family]):
            self.random.shuffle(self.order[family])
            self.cursors[family] = 0
            self.epochs[family] += 1
        index = self.order[family][self.cursors[family]]
        self.cursors[family] += 1
        self.presentations[index] = self.presentations.get(index, 0) + 1
        return index

    def state(self):
        return dict(order=self.order, cursors=self.cursors, epochs=self.epochs,
                    family_cursor=self.family_cursor, presentations=self.presentations, rng=self.random.getstate())

    def restore(self, state):
        if {k: sorted(v) for k, v in state["order"].items()} != {k: sorted(v) for k, v in self.order.items()}:
            raise ValueError("Sampler corpus membership changed")
        for name in ("order", "cursors", "epochs", "family_cursor", "presentations"):
            setattr(self, name, state[name])
        self.random.setstate(state["rng"])


def want_rollout(arm, reference_tokens, rollout_tokens, fraction=.25):
    if arm not in {"reference", "student_history"}:
        raise ValueError("Unknown scientific arm")
    return arm == "student_history" and rollout_tokens < fraction * (reference_tokens + rollout_tokens)


def rng_state(generator, *, cuda):
    array_state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(array_state[0], array_state[1].tolist(),
        array_state[2], array_state[3], array_state[4]), torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if cuda else [], rollout=generator.get_state())


def restore_rng(state, generator, *, cuda):
    random.setstate(state["python"])
    a = state["numpy"]
    np.random.set_state((a[0], np.array(a[1], dtype=np.uint32), a[2], a[3], a[4]))
    torch.set_rng_state(state["torch_cpu"])
    if cuda:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    elif state["torch_cuda"]:
        raise ValueError("CUDA checkpoint cannot resume under a CPU profile")
    generator.set_state(state["rollout"])


def seed_all(seed, *, cuda):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def encoded_rng(generator):
    return base64.b64encode(generator.get_state().numpy().tobytes()).decode("ascii")


@torch.no_grad()
def sample_rollout(record, next_logits, *, generator, checkpoint, update, teacher_id,
                   seed, max_new_tokens, max_length, temperature, eos_token_ids, device="cpu"):
    if record.split != "train" or len(record.prompt_ids) >= max_length:
        raise ValueError("Nonempty supported TRAIN generation prompt required")
    # Explicit per-prompt allowance is part of provenance, not position clamping.
    allowance = min(max_new_tokens, max_length - len(record.prompt_ids))
    before = encoded_rng(generator)
    prefix = torch.tensor([record.prompt_ids], dtype=torch.long, device=device)
    output = []
    for _ in range(allowance):
        logits = next_logits(prefix)
        if logits.ndim != 1 or not torch.isfinite(logits).all():
            raise ValueError("Finite original full-vocabulary student logits required")
        probabilities = torch.softmax(logits.detach().float().cpu() / temperature, dim=-1)
        token = int(torch.multinomial(probabilities, 1, generator=generator))
        output.append(token)
        if token in eos_token_ids:
            break
        prefix = torch.cat((prefix, prefix.new_tensor([[token]])), dim=1)
    reason = "eos" if output[-1] in eos_token_ids else "length"
    round_id = f"update_{update:08d}"
    provenance = dict(record.provenance, response_provenance="student_rollout",
        source_example_id=record.example_id, generator_model=checkpoint, student_checkpoint=checkpoint,
        generating_update=update, teacher_id=teacher_id, round_id=round_id, max_stale_updates=0,
        seed=seed, rng_before=before, rng_after=encoded_rng(generator),
        partial_history=reason == "length", observed_history_sha256=data.digest(record.prompt_ids + tuple(output)),
        decoding=dict(method="categorical_full_vocabulary", temperature=temperature,
                      max_new_tokens=allowance, declared_max_new_tokens=max_new_tokens,
                      total_context=max_length, eos_token_ids=list(eos_token_ids), use_cache=False))
    return replace(record, example_id=record.example_id + ":" + round_id + ":" + data.digest(before)[:12],
        response_ids=tuple(output), finish_reason=reason,
        eos_token_id=output[-1] if reason == "eos" else None, provenance=provenance)


class ExactHistoryTeacher:
    """Original frozen blocks streamed with the established host_weights primitive.

    Capture native cut residual and final-normalized hidden in the *same pass*,
    on the exact provided history. Generation stops native execution at the cut;
    teacher scoring executes all blocks. No reference-boundary cache is consulted.
    """
    def __init__(self, blocks, embedding, norm, rotary, *, cut, max_length, teacher_id, guard):
        self.blocks, self.embedding = blocks, embedding
        self.norm, self.rotary, self.cut = norm, rotary, cut
        self.max_length, self.teacher_id, self.guard = max_length, teacher_id, guard
        self.provider = response.CachedHiddenTeacher(embedding, teacher_id=teacher_id, guard=guard)
        self.calls = self.prefix_calls = self.transfer_bytes = 0
        self.transfer_seconds = 0.
        if not 0 <= cut < len(blocks):
            raise ValueError("Native suffix boundary outside teacher")
        for block in blocks:
            block.eval().requires_grad_(False)
        norm.eval().requires_grad_(False)
        rotary.eval().requires_grad_(False)
        self.original = self.fingerprint()
        self.resident_count = 0

    @contextmanager
    def prefix_residency(self, count):
        """Retain a bounded prefix bank; the same forward streams the remainder."""
        if type(count) is not int or not 0 <= count <= self.cut or self.resident_count:
            raise ValueError("Invalid or nested native-prefix residency")
        try:
            with ExitStack() as residency:
                for block in self.blocks[:count]:
                    residency.enter_context(host_weights(block, self.embedding.device, self.guard))
                self.resident_count = count
                yield
        finally:
            self.resident_count = 0
            if self.embedding.device.type == "cuda":
                torch.cuda.empty_cache()

    def fingerprint(self):
        named = [(f"block.{i}.{n}", t) for i, block in enumerate(self.blocks)
                 for n, t in list(block.named_parameters()) + list(block.named_buffers())]
        named += [("norm." + n, t) for n, t in self.norm.state_dict().items()]
        named += [("rotary." + n, t) for n, t in self.rotary.named_buffers()]
        named.append(("readout", self.embedding))
        return offload.fingerprint(named)

    def validate_unchanged(self):
        actual = self.fingerprint()
        if actual != self.original:
            raise ValueError("Frozen original teacher weights/rotary/readout changed")
        return dict(before=self.original, after=actual, unchanged=True)

    @torch.no_grad()
    def forward(self, ids, *, full, last_only=False):
        self.guard()
        if full and self.resident_count:
            raise ValueError("Release rollout residency before teacher target capture")
        if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= self.max_length:
            raise ValueError("One exact unpadded supported history required")
        ids = ids.to(self.embedding.device)
        hidden = F.embedding(ids, self.embedding)
        positions = torch.arange(ids.shape[1], device=ids.device)[None]
        allowed = torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool, device=ids.device).tril()[None, None]
        mask = torch.zeros(allowed.shape, dtype=hidden.dtype, device=ids.device)
        mask.masked_fill_(~allowed, torch.finfo(hidden.dtype).min)
        rope = self.rotary(hidden, positions)
        boundary = hidden.detach().cpu() if self.cut == 0 and full and not last_only else None
        for index, block in enumerate(self.blocks[:len(self.blocks) if full else self.cut]):
            if full and not last_only and index == self.cut:
                boundary = hidden.detach().cpu()
            if ids.device.type == "cuda":
                torch.cuda.synchronize(ids.device)
            started = time.monotonic()
            resident = index < self.resident_count
            with nullcontext() if resident else host_weights(block, ids.device, self.guard):
                self.transfer_seconds += time.monotonic() - started
                hidden = block(hidden, attention_mask=mask, position_ids=positions,
                    position_embeddings=rope, cache_position=positions[0], use_cache=False)[0]
                if ids.device.type == "cuda":
                    if not resident:
                        self.transfer_bytes += sum(v.numel() * v.element_size()
                            for v in (*block.parameters(), *block.buffers()))
                    torch.cuda.synchronize(ids.device)
                started = time.monotonic()
            self.transfer_seconds += time.monotonic() - started
        self.guard()
        if full:
            self.calls += 1
            return boundary, self.norm(hidden[:, -1:] if last_only else hidden).detach().cpu()
        self.prefix_calls += 1
        return hidden

    def native_prefix(self, ids, valid):
        if not bool(valid.all()):
            raise ValueError("Streamed generation accepts an unpadded exact history")
        return self.forward(ids, full=False)

    def prepare(self, batch):
        key = nal.batch_fingerprint(batch)
        boundary, hidden = self.forward(batch["input_ids"], full=True)
        batch.update(boundary_hidden=boundary, teacher_hidden=hidden,
            teacher_input_ids=batch["input_ids"].clone(),
            teacher_record_ids=[r.example_id for r in batch["records"]],
            teacher_history_binding=dict(batch_sha256=key, teacher_id=self.teacher_id,
                prediction_positions=batch["loss_mask"].nonzero().tolist(), cut=self.cut,
                boundary_sha256=offload.fingerprint([("boundary", boundary)])["sha256"],
                normalized_hidden_sha256=offload.fingerprint([("hidden", hidden)])["sha256"]))
        self.provider.prepare(batch)

    def __call__(self, batch, positions):
        if batch["teacher_history_binding"]["batch_sha256"] != nal.batch_fingerprint(batch):
            raise ValueError("Teacher history mutated after exact capture")
        return self.provider(batch, positions)

    def clear_cache(self):
        self.provider.clear_cache()

    @torch.no_grad()
    def next_logits(self, ids):
        _, hidden = self.forward(ids, full=True, last_only=True)
        return F.linear(hidden[:, -1].to(self.embedding), self.embedding)[0]


@contextmanager
def timed_prefix(model):
    original = model.native_prefix
    if not hasattr(model, "rollout_prefix_seconds"):
        model.rollout_prefix_seconds = 0.
    def forward(ids, valid):
        started = time.monotonic()
        try:
            return original(ids, valid)
        finally:
            if model.embedding.device.type == "cuda":
                torch.cuda.synchronize(model.embedding.device)
            model.rollout_prefix_seconds += time.monotonic() - started
    model.native_prefix = forward
    try:
        yield
    finally:
        model.native_prefix = original


@contextmanager
def rollout_residency(model, teacher, profile):
    """Residency changes storage placement, never the native prefix arithmetic."""
    count = 0 if profile["native_prefix"] == "streamed" else profile["resident_prefix_layers"]
    with teacher.prefix_residency(count), timed_prefix(model):
        yield


class RecoverySession:
    def __init__(self, model, teacher, corpus, *, spec, profile, identities, arm, guard):
        self.model, self.teacher, self.corpus = model, teacher, corpus
        self.spec, self.profile, self.identities, self.arm = spec, profile, identities, arm
        self.guard = guard
        self.model.stack.requires_grad_(True)
        self.model.norm.requires_grad_(True)
        self.model.stack.checkpoint_blocks = profile["checkpoint_blocks"]
        self.model.stack.enabled = False  # No BF16 Triton path, including Turing rollout.
        self.named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        self.parameter_ids = tuple(id(p) for _, p in self.named)
        if any(p.dtype != torch.bfloat16 for _, p in self.named) or model.embedding.dtype != torch.bfloat16:
            raise ValueError("Only the declared BF16 CPU-gradient profile is admitted")
        count = sum(p.numel() for _, p in self.named)
        model_bytes = sum(v.numel() * v.element_size() for v in model.state_dict().values() if isinstance(v, torch.Tensor))
        window_bytes = profile["maximum_window_examples"] * (
            profile["maximum_record_json_bytes"] * 40 + model.max_length * 40 + 32768)
        self.memory = dict(cpu_master_moments_bytes=12 * count, cpu_gradient_bytes=2 * count,
            checkpoint_model_bytes=model_bytes, bounded_window_record_bytes=window_bytes,
            additional_bound=14 * count + model_bytes + window_bytes + 6 * profile["optimizer_chunk_bytes"])
        self.host_admission = offload.require_host_memory(self.memory["additional_bound"], reserve_bytes=profile["host_reserve_bytes"])
        self.optimizer = offload.CPUGradientAdamW([p for _, p in self.named], lr=spec["learning_rate"],
            weight_decay=0., chunk_bytes=profile["optimizer_chunk_bytes"], max_cpu_state_bytes=profile["max_cpu_state_bytes"])
        self.optimizer_id = id(self.optimizer)
        self.loss_config = nal.DistillationConfig(objective="forward_kl", temperature=1., kd_weight=1., ce_weight=0.,
            token_chunk_size=profile["token_chunk_size"], grad_clip_norm=profile["gradient_clip"])
        self.sampler = FamilySampler(corpus, spec["seed"])
        self.generator = torch.Generator(device="cpu").manual_seed(spec["seed"])
        self.fixed = offload.fixed_state(model.stack, model.embedding, model.rotary)
        self.initial_parameters = offload.fingerprint(self.named)
        self.updates = self.reference_tokens = self.rollout_tokens = self.input_tokens = self.generated_tokens = 0
        self.clean_presentations = self.rollout_presentations = 0
        self.generating_checkpoint = spec["initial_checkpoint_sha256"]
        self.last_window = None

    @property
    def supervised_tokens(self):
        return self.reference_tokens + self.rollout_tokens

    def state(self):
        return dict(updates=self.updates, reference_tokens=self.reference_tokens, rollout_tokens=self.rollout_tokens,
            input_tokens=self.input_tokens, generated_tokens=self.generated_tokens,
            clean_presentations=self.clean_presentations, rollout_presentations=self.rollout_presentations,
            sampler=self.sampler.state(), rng=rng_state(self.generator, cuda=self.model.embedding.device.type == "cuda"),
            scheduler=dict(kind="constant", learning_rate=self.spec["learning_rate"], completed_updates=self.updates),
            mixture=dict(policy=self.profile["mixture_policy"], fraction=0. if self.arm == "reference" else .25,
                         pending_rollouts=[], max_stale_updates=0),
            initial_parameters=self.initial_parameters, fixed=self.fixed, last_window=self.last_window)

    def train_window(self, *, examples=None, pilot_indices=None):
        if (tuple(id(p) for p in self.model.parameters() if p.requires_grad) != self.parameter_ids
                or id(self.optimizer) != self.optimizer_id):
            raise ValueError("Live optimizer/parameter identities changed across refresh")
        self.guard()
        if examples is not None and not 1 <= examples <= self.spec["accumulation_examples"]:
            raise ValueError("Disposable pilot exceeds its declared example count")
        maximum_examples = self.profile["maximum_window_examples"] if examples is None else examples
        reference, rollout = self.reference_tokens, self.rollout_tokens
        records, indices = [], []
        started = time.monotonic()
        rollout_seconds = 0.
        prefix_seconds_before = getattr(self.model, "rollout_prefix_seconds", 0.)
        if pilot_indices is not None and (len(pilot_indices) != examples or
                any(not 0 <= i < len(self.corpus.rows) for i in pilot_indices)):
            raise ValueError("Disposable pilot indices must be admitted TRAIN records")
        residency = rollout_residency(self.model, self.teacher, self.profile) if self.arm == "student_history" else nullcontext()
        residency_started = time.monotonic()
        with residency:
            residency_setup_seconds = time.monotonic() - residency_started
            for position in range(maximum_examples):
                index = self.sampler.next() if pilot_indices is None else pilot_indices[position]
                if pilot_indices is not None:
                    self.sampler.presentations[index] = self.sampler.presentations.get(index, 0) + 1
                record = self.corpus[index]
                indices.append(index)
                if want_rollout(self.arm, reference, rollout):
                    tick = time.monotonic()
                    record = sample_rollout(record, self.model.next_logits, generator=self.generator,
                        checkpoint=self.generating_checkpoint, update=self.updates, teacher_id=self.teacher.teacher_id,
                        seed=self.spec["seed"], max_new_tokens=self.spec["rollout_max_tokens"],
                        max_length=self.model.max_length, temperature=self.spec["rollout_temperature"],
                        eos_token_ids=self.spec["eos_token_ids"], device=self.model.embedding.device)
                    rollout_seconds += time.monotonic() - tick
                    rollout += len(record.response_ids)
                else:
                    reference += len(record.response_ids)
                records.append(record)
                if examples is None and reference + rollout - self.supervised_tokens >= self.profile["window_target_tokens"]:
                    break
            residency_started = time.monotonic()
        residency_release_seconds = time.monotonic() - residency_started
        total = reference + rollout - self.supervised_tokens
        if examples is None and not self.profile["window_target_tokens"] <= total <= self.profile["maximum_window_tokens"]:
            raise ValueError("Token window missed the shared target or exceeded one complete response overshoot")
        self.model.train()
        self.optimizer.zero_grad()
        values, bindings = [], []
        capture_seconds = backward_seconds = 0.
        transfer_before, transfer_bytes = self.teacher.transfer_seconds, self.teacher.transfer_bytes
        gradient_before = self.optimizer.offload_bytes
        try:
            for record in records:
                batch = session_batch(record, max_length=self.model.max_length,
                    eos_token_ids=self.spec["eos_token_ids"], update=self.updates,
                    checkpoint=self.generating_checkpoint, teacher_id=self.teacher.teacher_id)
                tick = time.monotonic()
                self.teacher.prepare(batch)
                capture_seconds += time.monotonic() - tick
                bindings.append(batch["teacher_history_binding"])
                tick = time.monotonic()
                hidden = self.model.hidden_forward(batch)
                values.append(nal.chunked_backward(hidden, self.model.readout, self.teacher,
                                                  batch, self.loss_config, normalizer=total))
                del hidden, batch
                self.teacher.clear_cache()
                backward_seconds += time.monotonic() - tick
            if any(self.optimizer.received.get(p, 0) != len(records) or p.grad is not None for _, p in self.named):
                raise ValueError("Every trainable block/norm parameter must contribute every observed history")
            tick = time.monotonic()
            self.guard()
            optimizer_receipt = self.optimizer.step(max_grad_norm=self.profile["gradient_clip"])
            if any(not bool(torch.isfinite(p).all()) for _, p in self.named):
                raise ValueError("Nonfinite model after update; transaction will not commit")
            optimizer_seconds = time.monotonic() - tick
            self.updates += 1
            self.reference_tokens, self.rollout_tokens = reference, rollout
            self.input_tokens += sum(len(r.input_ids) for r in records)
            sampled = [r for r in records if r.provenance["response_provenance"] == "student_rollout"]
            self.generated_tokens += sum(len(r.response_ids) for r in sampled)
            self.clean_presentations += len(records) - len(sampled)
            self.rollout_presentations += len(sampled)
            self.last_window = dict(update=self.updates, indices=indices, records=[asdict(r) for r in records],
                exact_teacher_bindings=bindings, supervised_tokens=total,
                cumulative_supervised_tokens=self.supervised_tokens, reference_tokens=reference, rollout_tokens=rollout,
                gradients_present_for_every_parameter=True, optimizer=optimizer_receipt,
                **{name: sum(v[name] for v in values) / total for name in
                   ("loss", "forward_kl", "ce", "correct", "teacher_agreement")},
                timings=dict(rollout_seconds=rollout_seconds, capture_seconds=capture_seconds,
                    native_prefix_seconds=getattr(self.model, "rollout_prefix_seconds", 0.) - prefix_seconds_before,
                    residency_setup_seconds=residency_setup_seconds, residency_release_seconds=residency_release_seconds,
                    teacher_transfer_seconds=self.teacher.transfer_seconds - transfer_before,
                    backward_seconds=backward_seconds, optimizer_seconds=optimizer_seconds,
                    window_seconds=time.monotonic() - started),
                teacher_transfer_bytes=self.teacher.transfer_bytes - transfer_bytes,
                gradient_transfer_bytes=self.optimizer.offload_bytes - gradient_before)
            return self.last_window
        finally:
            self.teacher.clear_cache()
            self.optimizer.zero_grad()

    def unchanged_proof(self):
        after = offload.fixed_state(self.model.stack, self.model.embedding, self.model.rotary)
        if after != self.fixed:
            raise ValueError("Frozen student geometry/readout/rotary changed")
        return dict(geometry=dict(before=self.fixed, after=after, unchanged=True),
                    teacher=self.teacher.validate_unchanged())

    def save(self, directory, *, parent=None, publish=None):
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError(directory)
        if self.optimizer.cpu_gradients or self.optimizer.received:
            raise ValueError("Only completed zero-gradient optimizer windows may commit")
        tick = time.monotonic()
        temporary = directory.with_name("." + directory.name + "." + uuid.uuid4().hex)
        temporary.mkdir(parents=True)
        required_disk = self.memory["checkpoint_model_bytes"] + self.memory["cpu_master_moments_bytes"] + 64 * 1024**2
        if shutil.disk_usage(temporary).free < required_disk:
            raise OSError(f"Atomic checkpoint requires {required_disk} free bytes while retaining prior commits")
        state = {name: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                 for name, v in self.model.state_dict().items()}
        torch.save(state, temporary / "model.pt")
        del state
        torch.save(self.state(), temporary / "control.pt")
        self.optimizer.save_checkpoint(temporary / "optimizer")
        files = {str(p.relative_to(temporary)): sha(p) for p in sorted(temporary.rglob("*")) if p.is_file()}
        manifest = dict(schema=CHECKPOINT_SCHEMA, status="committed", arm=self.arm, identities=self.identities,
            parent=parent, update=self.updates, supervised_tokens=self.supervised_tokens,
            reference_tokens=self.reference_tokens, rollout_tokens=self.rollout_tokens,
            last_window_sha256=data.digest(self.last_window),
            parameter_names=[name for name, _ in self.named], files=files,
            profile=self.profile, model_fingerprint=offload.fingerprint(self.named), fixed=self.fixed,
            optimizer_initialization="new from model-only initializer" if self.updates == 0 else "continuous named CPU Adam masters/moments",
            checkpoint_seconds=time.monotonic() - tick)
        manifest["fingerprint"] = data.digest(manifest)
        write_json(temporary / "manifest.json", manifest)
        authenticate_checkpoint(temporary, identities=self.identities)
        for path in temporary.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        sync_directory(temporary / "optimizer")
        sync_directory(temporary)
        os.rename(temporary, directory)
        sync_directory(directory.parent)
        receipt = dict(directory=str(directory.resolve()), manifest_sha256=sha(directory / "manifest.json"),
                       update=self.updates, supervised_tokens=self.supervised_tokens,
                       checkpoint_seconds=time.monotonic() - tick)
        if publish is not None:
            atomic_json(publish, receipt)
        self.generating_checkpoint = receipt["manifest_sha256"]
        return receipt

    def restore(self, receipt):
        directory = Path(receipt["directory"])
        manifest = authenticate_checkpoint(directory, receipt["manifest_sha256"], identities=self.identities)
        if (manifest["arm"] != self.arm or manifest["profile"] != self.profile
                or manifest["parameter_names"] != [name for name, _ in self.named] or manifest["fixed"] != self.fixed):
            raise ValueError("Resume model/optimizer names, profile or geometry changed")
        model_state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        self.model.load_state_dict(model_state, strict=True)
        del model_state
        self.optimizer.load_checkpoint(directory / "optimizer")
        if offload.fingerprint(self.named) != manifest["model_fingerprint"]:
            raise ValueError("CPU masters do not reproduce committed model weights")
        state = torch.load(directory / "control.pt", map_location="cpu", weights_only=True)
        if state["scheduler"] != dict(kind="constant", learning_rate=self.spec["learning_rate"], completed_updates=manifest["update"]):
            raise ValueError("Scheduler trajectory changed")
        if state["mixture"]["pending_rollouts"] or state["mixture"]["max_stale_updates"] != 0:
            raise ValueError("Committed checkpoints cannot carry stale pending rollouts")
        for name in ("updates", "reference_tokens", "rollout_tokens", "input_tokens", "generated_tokens",
                     "clean_presentations", "rollout_presentations", "initial_parameters", "last_window"):
            setattr(self, name, state[name])
        self.sampler.restore(state["sampler"])
        self.generating_checkpoint = receipt["manifest_sha256"]
        self.optimizer.zero_grad()
        if (self.updates != manifest["update"] or self.supervised_tokens != manifest["supervised_tokens"]
                or any(s["step"] != self.updates for s in self.optimizer.state.values())):
            raise ValueError("Committed optimizer/global update/token cursor differs")
        restore_rng(state["rng"], self.generator, cuda=self.model.embedding.device.type == "cuda")

    def close(self):
        self.optimizer.close()
        self.teacher.clear_cache()


def authenticate_checkpoint(directory, expected_sha256=None, *, identities=None):
    directory = Path(directory).resolve()
    if expected_sha256 is not None and sha(directory / "manifest.json") != expected_sha256:
        raise ValueError("Checkpoint manifest changed")
    manifest = read_json(directory / "manifest.json")
    if (manifest.get("schema") != CHECKPOINT_SCHEMA or manifest.get("status") != "committed"
            or identities is not None and manifest["identities"] != identities):
        raise ValueError("Checkpoint is incomplete or belongs to another source/data/geometry")
    if data.digest({k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest.get("fingerprint"):
        raise ValueError("Checkpoint manifest fingerprint changed")
    for name, digest in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha(path) != digest:
            raise ValueError("Checkpoint payload changed: " + name)
    if not {"model.pt", "control.pt", "optimizer/manifest.json"}.issubset(manifest["files"]):
        raise ValueError("Whole-window model/optimizer/RNG/cursor checkpoint required")
    optimizer = read_json(directory / "optimizer/manifest.json")
    if len(optimizer["parameters"]) != len(manifest["parameter_names"]):
        raise ValueError("Named optimizer membership is incomplete")
    for parameter in optimizer["parameters"]:
        if parameter["step"] != manifest["update"]:
            raise ValueError("Optimizer and committed update disagree")
        for chunk in parameter["chunks"]:
            if manifest["files"].get("optimizer/" + chunk["file"]) != chunk["sha256"]:
                raise ValueError("Optimizer chunk is not authenticated by the whole-window transaction")
    return manifest


def recover_commit_pointer(output, *, identities):
    """Recover a durable directory renamed just before a crash publishing latest."""
    output = Path(output)
    latest = output / "latest.json"
    current = read_json(latest) if latest.exists() else None
    if current is not None:
        authenticate_checkpoint(current["directory"], current["manifest_sha256"], identities=identities)
    while True:
        update = 0 if current is None else current["update"] + 1
        directory = output / "checkpoints" / f"update_{update:08d}"
        if not directory.exists():
            return current
        manifest = authenticate_checkpoint(directory, identities=identities)
        parent = manifest["parent"]
        if (manifest["update"] != update or (parent is None) != (current is None)
                or parent is not None and any(parent[key] != current[key]
                    for key in ("directory", "manifest_sha256", "update", "supervised_tokens"))):
            raise ValueError("Unpublished checkpoint does not descend from the last committed window")
        current = dict(directory=str(directory.resolve()), manifest_sha256=sha(directory / "manifest.json"),
            update=manifest["update"], supervised_tokens=manifest["supervised_tokens"],
            checkpoint_seconds=manifest["checkpoint_seconds"])
        atomic_json(latest, current)


def admit_corpus(plan, *, max_length, original_range=False):
    spec = plan["spec"]
    manifest = read_json(spec["corpus_manifest"])
    if (manifest.get("schema") != "jspace.recovery-reset.corpus.v1"
            or manifest.get("training_admitted") is not True
            or data.digest({k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest.get("fingerprint")):
        raise ValueError("Authenticated admitted recovery corpus manifest required")
    path = Path(manifest["train_records"])
    if not path.is_absolute():
        raise ValueError("Corpus records must have absolute pinned paths")
    digests = manifest["sha256"]
    expected = digests.get(str(path), digests.get("train_records"))
    if expected is None or sha(path) != expected:
        raise ValueError("TRAIN shard hash differs from admitted corpus")
    corpus = IndexedRecords(path, max_length=max_length, eos_token_ids=spec["eos_token_ids"])
    if corpus.exclusions and not original_range:
        corpus.close()
        raise ValueError("Scientific corpus contains unsupported lengths; no silent subsetting")
    return corpus, manifest


def build_runtime(plan, *, arm, guard, original_range=False, device="cuda:0"):
    from . import jspace_3b_context_qualification as geometry
    spec, profile = plan["spec"], plan["profile"]
    guard()
    seed_all(spec["seed"], cuda=torch.device(device).type == "cuda")
    geometry_sha = None
    if original_range:
        max_length = read_json(spec["teacher_plan"])["max_length"]
        parts = geometry.load_initial(spec, device="cpu", guard=None)
    else:
        manifest = read_json(spec["geometry_manifest"])
        if (manifest.get("schema") != "jspace.recovery-reset.geometry.v1" or manifest.get("qualified") is not True
                or manifest["max_length"] != spec["target_context"]
                or manifest["original_checkpoint_sha256"] != spec["initial_checkpoint_sha256"]):
            raise ValueError("Scientific training requires qualified identical 2048-position geometry")
        geometry_sha = sha(spec["geometry_manifest"])
        max_length = manifest["max_length"]
        parts = geometry.load_extended(spec["geometry_manifest"], device="cpu", guard=None)
    stack, norm, embedding, rotary = parts
    corpus, corpus_manifest = admit_corpus(plan, max_length=max_length, original_range=original_range)
    cfg = native._config(spec["native_model_path"])
    teacher_bytes = sum(Path(path).stat().st_size for path in plan["input_sha256"] if path.endswith(".safetensors"))
    offload.require_host_memory(teacher_bytes, reserve_bytes=profile["host_reserve_bytes"])
    original_embedding, teacher_norm, teacher_rotary = native.load_readout_norm(spec["native_model_path"], device="cpu", config=cfg)
    if not torch.equal(embedding, original_embedding):
        raise ValueError("Student readout differs from the original native teacher")
    del original_embedding
    blocks = [native.load_native_block(spec["native_model_path"], i, device="cpu", config=cfg)
              for i in range(cfg.num_hidden_layers)]
    guard()
    model = response.ResponseModel(stack.to(device), norm.to(device), embedding.to(device), rotary.to(device),
        max_length=max_length, expected_layers=cfg.num_hidden_layers, guard=guard)
    teacher = ExactHistoryTeacher(blocks, model.embedding, teacher_norm.to(device), teacher_rotary.to(device),
        cut=model.cut, max_length=max_length, teacher_id=spec["native_model_revision"], guard=guard)
    model.native_prefix = teacher.native_prefix
    identities = dict(plan=plan["fingerprint"], source_sha256=plan["source_sha256"],
        spec_sha256=plan["spec_sha256"],
        model_sha256={p: h for p, h in plan["input_sha256"].items() if p.startswith(spec["native_model_path"])},
        teacher=spec["native_model_revision"], teacher_plan_sha256=sha(spec["teacher_plan"]),
        initializer=spec["initial_checkpoint_sha256"], geometry=geometry_sha,
        corpus=sha(spec["corpus_manifest"]), train_records=sha(corpus.path),
        consumed_sha256={spec["corpus_manifest"]: sha(spec["corpus_manifest"]), str(corpus.path): sha(corpus.path),
            **({spec["geometry_manifest"]: geometry_sha, **manifest["artifacts"]} if not original_range else {})},
        tokenizer=corpus.tokenizer, chat_template=corpus.template,
        max_length=max_length, original_range_disposable=original_range,
        hardware=dict(name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)))
                 if torch.device(device).type == "cuda" else dict(name="cpu", capability=[]),
        torch_version=str(torch.__version__))
    session = RecoverySession(model, teacher, corpus, spec=spec, profile=profile, identities=identities, arm=arm, guard=guard)
    return session, corpus_manifest


def failure(directory, error, *, session=None):
    directory = Path(directory)
    write_json(directory / ("failure_" + uuid.uuid4().hex + ".json"),
        dict(status="execution_failure", error=repr(error),
             committed_state_policy="Resume only the last authenticated whole-window checkpoint",
             in_memory_updates=None if session is None else session.updates))


def summary(session):
    return dict(updates=session.updates, supervised_tokens=session.supervised_tokens,
        reference_tokens=session.reference_tokens, rollout_tokens=session.rollout_tokens,
        realized_rollout_fraction=session.rollout_tokens / session.supervised_tokens if session.supervised_tokens else 0.,
        input_tokens=session.input_tokens, generated_tokens=session.generated_tokens,
        clean_presentations=session.clean_presentations, rollout_presentations=session.rollout_presentations,
        unique_prompts=len(session.sampler.presentations),
        repeated_prompt_presentations=sum(max(0, n - 1) for n in session.sampler.presentations.values()),
        memory=session.memory, host_admission=session.host_admission,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if session.model.embedding.device.type == "cuda" else 0,
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if session.model.embedding.device.type == "cuda" else 0)


def archive_window(output, directory):
    """Retain exact histories and authenticated old manifest, not fake resumability."""
    output, directory = Path(output), Path(directory)
    destination = output / "history" / directory.name
    if destination.exists():
        return authenticate_history(destination)
    manifest = read_json(directory / "manifest.json")
    state = torch.load(directory / "control.pt", map_location="cpu", weights_only=True)
    window = state["last_window"]
    if data.digest(window) != manifest["last_window_sha256"]:
        raise ValueError("Window history differs from committed checkpoint")
    temporary = destination.with_name("." + destination.name + "." + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    shutil.copy2(directory / "manifest.json", temporary / "checkpoint_manifest.json")
    write_json(temporary / "window.json", window)
    receipt = dict(status="window_history_only_not_resumable", source_checkpoint=str(directory.resolve()),
        checkpoint_manifest_sha256=sha(temporary / "checkpoint_manifest.json"),
        window_sha256=sha(temporary / "window.json"), update=manifest["update"])
    write_json(temporary / "receipt.json", receipt)
    for path in temporary.iterdir():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    sync_directory(temporary)
    os.rename(temporary, destination)
    sync_directory(destination.parent)
    return receipt


def authenticate_history(directory):
    directory = Path(directory)
    receipt = read_json(directory / "receipt.json")
    if (receipt["status"] != "window_history_only_not_resumable"
            or sha(directory / "checkpoint_manifest.json") != receipt["checkpoint_manifest_sha256"]
            or sha(directory / "window.json") != receipt["window_sha256"]):
        raise ValueError("Retained window history changed")
    manifest = read_json(directory / "checkpoint_manifest.json")
    if (data.digest({k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest["fingerprint"]
            or data.digest(read_json(directory / "window.json")) != manifest["last_window_sha256"]):
        raise ValueError("Archived checkpoint/observed history binding changed")
    return receipt


def retain_checkpoints(output, *, identities):
    """Only owned unreferenced old payloads; zero, latest two and milestones survive."""
    output = Path(output).resolve()
    latest = read_json(output / "latest.json")
    authenticate_checkpoint(latest["directory"], latest["manifest_sha256"], identities=identities)
    protected_hashes = {latest["manifest_sha256"]}
    for path in [*output.glob("phases/*/result.json"), *output.glob("generation/*/result.json")]:
        reference = read_json(path).get("checkpoint")
        if isinstance(reference, dict):
            protected_hashes.add(reference["manifest_sha256"])
        elif isinstance(reference, str):
            protected_hashes.add(reference)
    ledger_path = output / "retention.json"
    previous = read_json(ledger_path) if ledger_path.exists() else dict(pruned=[])
    if previous.get("status") == "pruning":
        for deletion in previous["pruned"]:
            directory = Path(deletion["directory"])
            if not directory.exists():
                continue
            if (directory.is_symlink() or directory.resolve().parent != output / "checkpoints"
                    or deletion["manifest_sha256"] in protected_hashes):
                raise ValueError("Interrupted pruning intent is not an owned unreferenced checkpoint")
            history = authenticate_history(deletion["history"])
            archived = read_json(Path(deletion["history"]) / "checkpoint_manifest.json")
            if (history != deletion["history_receipt"] or archived["identities"] != identities
                    or history["checkpoint_manifest_sha256"] != deletion["manifest_sha256"]):
                raise ValueError("Interrupted deletion lacks authenticated owned history")
            shutil.rmtree(directory)
            sync_directory(directory.parent)
    directories = sorted(output.glob("checkpoints/update_[0-9]*"))
    rows = []
    for directory in directories:
        if directory.is_symlink() or directory.resolve().parent != output / "checkpoints":
            raise ValueError("Refusing retention outside owned checkpoint directories")
        manifest = read_json(directory / "manifest.json")
        if manifest["identities"] != identities:
            raise ValueError("Refusing retention of another study's checkpoint")
        rows.append((directory, manifest, sha(directory / "manifest.json")))
    latest_updates = sorted((manifest["update"] for _, manifest, _ in rows), reverse=True)[:2]
    pruned = list(previous["pruned"])
    retained_bytes = 0
    retained = []
    for directory, manifest, digest in rows:
        archive_window(output, directory)
        size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        if manifest["update"] == 0 or manifest["update"] in latest_updates or digest in protected_hashes:
            retained_bytes += size
            retained.append(dict(directory=str(directory), update=manifest["update"], bytes=size))
            continue
        authenticate_checkpoint(directory, digest, identities=identities)
        history = authenticate_history(output / "history" / directory.name)
        deletion = dict(directory=str(directory), manifest_sha256=digest, bytes=size,
                        history=str(output / "history" / directory.name), history_receipt=history)
        # Publish pruning intent only after the new latest checkpoint is durable;
        # an interrupted deletion is recoverable from this explicit owned list.
        pruned.append(deletion)
        atomic_json(ledger_path, dict(pruned=pruned, retained=retained, status="pruning"))
        shutil.rmtree(directory)
        sync_directory(directory.parent)
    ledger = dict(status="completed", policy="zero and declared referenced milestones plus latest two full commits",
        retained=retained, pruned=pruned, retained_checkpoint_bytes=retained_bytes,
        pruned_checkpoint_bytes=sum(row["bytes"] for row in pruned))
    atomic_json(ledger_path, ledger)
    return {key: value for key, value in ledger.items() if key not in {"retained", "pruned"}}


def train(root, *, arm, target_tokens):
    root = Path(root).resolve()
    plan = load_plan(root)
    if target_tokens not in plan["spec"]["milestones"]:
        raise ValueError("Target must be a frozen cumulative scientific milestone")
    pilot_receipt = read_json(root / "pilot/result.json")
    if (pilot_receipt.get("status") != "systems_passed" or pilot_receipt.get("original_range_disposable")
            or pilot_receipt.get("plan_fingerprint") != plan["fingerprint"]):
        raise ValueError("Qualified-context paired-resume systems pilot required before scientific updates")
    output = root / "arms" / arm
    phase = output / "phases" / f"tokens_{target_tokens}"
    if (phase / "result.json").exists():
        raise FileExistsError("Milestone receipt already exists")
    phase.mkdir(parents=True, exist_ok=True)
    session = None
    start = time.monotonic()
    try:
        lease, guard = reservation(plan["profile"]["limit_mib"])
        session, _ = build_runtime(plan, arm=arm, guard=guard)
        if session.identities != pilot_receipt["identities"]:
            raise ValueError("Pilot and scientific model/data/geometry/hardware identities differ")
        latest = output / "latest.json"
        committed = recover_commit_pointer(output, identities=session.identities)
        if committed is not None:
            session.restore(committed)
        else:
            committed = session.save(output / "checkpoints/update_00000000", publish=latest)
        retention = retain_checkpoints(output, identities=session.identities)
        # Milestones cannot be bypassed; a resumed interrupted phase may already
        # exceed its target by the single whole window which committed last.
        earlier = [n for n in plan["spec"]["milestones"] if n < target_tokens]
        if any(not (output / "phases" / f"tokens_{n}" / "result.json").is_file() for n in earlier):
            raise ValueError("Earlier cumulative milestones need committed receipts")
        starting_tokens = session.supervised_tokens
        while session.supervised_tokens < target_tokens:
            row = session.train_window()
            committed = session.save(output / "checkpoints" / f"update_{session.updates:08d}",
                                      parent=committed, publish=latest)
            retention = retain_checkpoints(output, identities=session.identities)
            print(json.dumps(dict(phase="committed_window", arm=arm, update=session.updates,
                supervised_tokens=session.supervised_tokens, window_tokens=row["supervised_tokens"])), flush=True)
        proof = session.unchanged_proof()
        overshoot = session.supervised_tokens - target_tokens
        if overshoot < 0 or overshoot > plan["profile"]["maximum_window_token_overshoot"]:
            raise ValueError("Milestone exceeded the declared one-window overshoot")
        result = dict(schema=SCHEMA, status="completed", scientific_status="ungraded", confirmed=False,
            arm=arm, target_tokens=target_tokens, starting_tokens=starting_tokens, overshoot_tokens=overshoot,
            plan_fingerprint=plan["fingerprint"], identities=session.identities, checkpoint=committed,
            spec_sha256=plan["spec_sha256"],
            retention=retention,
            parent_model_only=plan["spec"]["initial_checkpoint_sha256"], lease=lease,
            frozen_proof=proof, objective=dict(name="full_vocabulary_forward_kl", temperature=1., ce_weight=0.),
            elapsed_seconds=time.monotonic() - start, **summary(session))
        write_json(phase / "result.json", result)
        return result
    except BaseException as error:
        failure(phase, error, session=session)
        raise
    finally:
        if session is not None:
            session.close()
            session.corpus.close()


def _tensor_comparison(left, right, *, atol, rtol):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.shape == right.shape and left.dtype == right.dtype
                and torch.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _tensor_comparison(left[k], right[k], atol=atol, rtol=rtol) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _tensor_comparison(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right))
    return left == right


def compare_checkpoints(left, right, *, atol=0., rtol=0.):
    a = authenticate_checkpoint(left["directory"], left["manifest_sha256"])
    b = authenticate_checkpoint(right["directory"], right["manifest_sha256"], identities=a["identities"])
    paths = ["model.pt"] + sorted(name for name in a["files"] if name.startswith("optimizer/") and name.endswith(".pt"))
    for name in paths:
        x = torch.load(Path(left["directory"]) / name, map_location="cpu", weights_only=True, mmap=True)
        y = torch.load(Path(right["directory"]) / name, map_location="cpu", weights_only=True, mmap=True)
        if not _tensor_comparison(x, y, atol=atol, rtol=rtol):
            raise ValueError("Interrupted/resumed numerical trajectory differs: " + name)
        del x, y
    x = torch.load(Path(left["directory"]) / "control.pt", map_location="cpu", weights_only=True)
    y = torch.load(Path(right["directory"]) / "control.pt", map_location="cpu", weights_only=True)
    # Wall-clock/transfer counters differ across restart; mathematical state,
    # sampled IDs, next sampler cursor, all RNGs and target bindings must not.
    windows = [x.pop("last_window"), y.pop("last_window")]
    for window in windows:
        for key in ("timings", "teacher_transfer_bytes", "gradient_transfer_bytes"):
            window.pop(key, None)
    if not _tensor_comparison(x, y, atol=0., rtol=0.) or not _tensor_comparison(windows[0], windows[1], atol=atol, rtol=rtol):
        raise ValueError("Resume RNG/cursor/masks/rollout or numerical loss trajectory differs")
    if a["parameter_names"] != b["parameter_names"] or a["update"] != b["update"]:
        raise ValueError("Resume optimizer membership/step differs")
    return dict(passed=True, compared_model_and_optimizer_chunks=len(paths), atol=atol, rtol=rtol,
                next_examples_rng_and_loss_masks_equal=True, exact_teacher_history_bindings_equal=True)


def project_pilot_cost(windows, checkpoints, profile):
    tokens = sum(row["supervised_tokens"] for row in windows)
    measured = {key: sum(row["timings"][key] for row in windows) for key in
                ("native_prefix_seconds", "rollout_seconds", "capture_seconds", "teacher_transfer_seconds",
                 "backward_seconds", "optimizer_seconds", "residency_setup_seconds", "residency_release_seconds")}
    measured["checkpoint_seconds"] = sum(cp["checkpoint_seconds"] for cp in checkpoints)
    measured["rollout_excluding_native_prefix_seconds"] = max(0., measured["rollout_seconds"] - measured["native_prefix_seconds"])
    total = sum(measured[key] for key in ("rollout_seconds", "capture_seconds", "backward_seconds",
        "optimizer_seconds", "residency_setup_seconds", "residency_release_seconds", "checkpoint_seconds"))
    return dict(measured_loss_bearing_tokens=tokens, measured_seconds=measured,
        projected_seconds={str(target): {key: value * target / tokens for key, value in measured.items()}
                           for target in (100000, 1000000)},
        projected_total_hours={str(target): total * target / tokens / 3600 for target in (100000, 1000000)},
        selected_profile=profile, checkpoint_retention=profile["retention"],
        interpretation="Linear two-window systems extrapolation, not scientific budget execution or accuracy evidence; native-prefix and teacher-transfer are subsets of rollout and capture")


def qualify_prefix_residency(session, *, original_range=False):
    """Exercise complete TRAIN histories and the admitted context-boundary prefix."""
    longest = max(range(len(session.corpus.rows)), key=lambda i: session.corpus.rows[i][4])
    cases = [("prompt", session.corpus[0].prompt_ids),
             ("longest_observed_history", session.corpus[longest].input_ids[:-1])]
    calibration = None
    if not original_range:
        from .jspace_3b_context_qualification import calibration_records
        rows, calibration = calibration_records(session.spec["corpus_manifest"],
            target=session.spec["target_context"], maximum=1, minimum=1)
        cases.append(("full_context_calibration", rows[0]["input_ids"]))
    results = []
    for name, tokens in cases:
        ids = torch.tensor([tokens], device=session.model.embedding.device)
        expected = session.model.next_logits(ids).detach().cpu()
        with rollout_residency(session.model, session.teacher, session.profile):
            actual = session.model.next_logits(ids).detach().cpu()
        if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
            raise ValueError(f"Native-prefix residency changed {name} logits")
        results.append(dict(case=name, tokens=len(tokens), bit_exact=True,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if ids.device.type == "cuda" else 0))
    if calibration is not None:
        results[-1]["calibration"] = calibration
    return results


def pilot(root, *, original_range=False, proof_stage="all"):
    root = Path(root).resolve()
    plan = load_plan(root)
    directory = root / "pilot"
    if (directory / "result.json").exists():
        raise FileExistsError("Disposable pilot already has an immutable result")
    directory.mkdir(parents=True, exist_ok=True)
    session = None
    try:
        lease, guard = reservation(plan["profile"]["limit_mib"])
        session, _ = build_runtime(plan, arm="student_history", guard=guard, original_range=original_range)
        count = min(plan["spec"]["pilot_examples"], plan["spec"]["accumulation_examples"])
        if proof_stage in {"all", "baseline"}:
            prefix_parity = qualify_prefix_residency(session, original_range=original_range)
            write_json(directory / "prefix_parity.json",
                       dict(plan_fingerprint=plan["fingerprint"], cases=prefix_parity))
            initial = session.save(directory / "baseline/update_00000000")
            initial_logits = session.model.next_logits(torch.tensor([session.corpus[0].prompt_ids],
                                      device=session.model.embedding.device)).detach().cpu()
            torch.save(initial_logits, directory / "initial_logits.pt")
            lengths = sorted(range(len(session.corpus.rows)), key=lambda i: session.corpus.rows[i][4])
            # Both short and longest admitted histories really execute teacher
            # capture and backward. No fabricated long filler or cap extension.
            pilot_indices = [lengths[-1], lengths[0]]
            pilot_indices += [lengths[i % len(lengths)] for i in range(count - 2)]
            if not original_range and session.corpus.rows[lengths[-1]][4] <= read_json(plan["spec"]["teacher_plan"])["max_length"]:
                raise ValueError("Extended runtime pilot requires an actual admitted long TRAIN history")
            initial_blocks = [offload.fingerprint(block.named_parameters()) for block in session.model.stack.blocks]
            session.train_window(examples=count, pilot_indices=pilot_indices)
            interruption = session.save(directory / "baseline/update_00000001", parent=initial)
            first = session.last_window
            session.train_window(examples=count)
            expected = session.save(directory / "baseline/update_00000002", parent=interruption)
            learned = offload.fingerprint(session.named)
            if learned == session.initial_parameters or session.reference_tokens == 0 or session.rollout_tokens == 0:
                raise ValueError("Pilot must change actual BF16 weights and exercise clean plus sampled histories")
            changed_blocks = [before != offload.fingerprint(block.named_parameters())
                              for before, block in zip(initial_blocks, session.model.stack.blocks)]
            if not all(changed_blocks):
                raise ValueError("Pilot did not change actual BF16 weights in every compressed block")
            proof = session.unchanged_proof()
            final_logits = session.model.next_logits(torch.tensor([session.corpus[0].prompt_ids],
                                      device=session.model.embedding.device)).detach().cpu()
            torch.save(final_logits, directory / "final_logits.pt")
            baseline = dict(schema=SCHEMA, status="awaiting_fresh_process_resume", pid=os.getpid(),
                plan_fingerprint=plan["fingerprint"], identities=session.identities, lease=lease,
                original_range_disposable=original_range, initial=initial, interruption=interruption, expected=expected,
                exercised_lengths=[len(r["prompt_ids"]) + len(r["response_ids"]) for r in first["records"]],
                prefix_parity=prefix_parity,
                changed_bf16_blocks=changed_blocks, corpus_exclusions=session.corpus.exclusions,
                frozen_proof=proof, changed_bf16_weights=True, first_window=first,
                initial_logits_sha256=sha(directory / "initial_logits.pt"), final_logits_sha256=sha(directory / "final_logits.pt"),
                cost_projection=project_pilot_cost([first, session.last_window], [interruption, expected], plan["profile"]),
                logits_changed=not torch.equal(initial_logits, final_logits), **summary(session))
            write_json(directory / "baseline.json", baseline)
            session.close()
            session.corpus.close()
            del session
            session = None
            gc.collect()
            torch.cuda.empty_cache()
            if proof_stage == "baseline":
                return dict(status="awaiting_fresh_process_resume", baseline=str(directory / "baseline.json"))
            command = [sys.executable, "-m", "experiments.structure_net.jspace_3b_recovery_session",
                       "pilot", "--root", str(root), "--proof-stage", "resume"]
            if original_range:
                command.append("--original-range")
            subprocess.run(command, check=True, env=os.environ.copy())
            return read_json(directory / "result.json")
        baseline = read_json(directory / "baseline.json")
        if baseline["pid"] == os.getpid() or baseline["original_range_disposable"] != original_range:
            raise ValueError("Resume proof requires a fresh process with identical context policy")
        session.restore(baseline["interruption"])
        session.train_window(examples=count)
        resumed = session.save(directory / "resumed/update_00000002", parent=baseline["interruption"])
        comparison = compare_checkpoints(baseline["expected"], resumed,
            atol=plan["profile"]["resume_atol"], rtol=plan["profile"]["resume_rtol"])
        proof = session.unchanged_proof()
        result = dict(schema=SCHEMA, status="systems_passed", passed=True, discarded=True,
            quality_status="not_evaluated", confirmed=False, spec_sha256=plan["spec_sha256"],
            plan_fingerprint=plan["fingerprint"], identities=session.identities, original_range_disposable=original_range,
            baseline_pid=baseline["pid"], resumed_pid=os.getpid(), fresh_process=True,
            paired_resume=comparison, frozen_proof=proof, checkpoint=resumed,
            baseline_checkpoint=baseline["expected"], changed_bf16_weights=baseline["changed_bf16_weights"],
            cost_projection=baseline["cost_projection"], exercised_lengths=baseline["exercised_lengths"],
            prefix_parity=baseline["prefix_parity"],
            lease=lease, **summary(session))
        write_json(directory / "result.json", result)
        return result
    except BaseException as error:
        failure(directory, error, session=session)
        raise
    finally:
        if session is not None:
            session.close()
            session.corpus.close()


def generation_row(record, generated, *, role, checkpoint, tokenizer, decoding, grader_identity, geometry):
    return dict(example_id=record.example_id, group_id=record.group_id, task=record.task,
        role=role, checkpoint=checkpoint, split="development", input_ids=list(record.prompt_ids),
        decoding=decoding, decoding_identity=data.digest(decoding), grader_identity=grader_identity,
        checkpoint_geometry=geometry, response=tokenizer.decode(generated["output_ids"], skip_special_tokens=True),
        output_ids=generated["output_ids"], stop_reason="length" if generated["stop_reason"] == "token_limit"
        else generated["stop_reason"], status="generated", actual_max_new_tokens=generated["max_new_tokens"])


def native_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["role"] == "native":
                yield row


def generate(root, *, arm, target_tokens):
    from transformers import AutoTokenizer
    root = Path(root).resolve()
    plan = load_plan(root)
    if target_tokens not in [0, *plan["spec"]["milestones"]]:
        raise ValueError("Generation requires initializer or a frozen milestone")
    evaluation_path = Path(plan["spec"]["artifact_root"]) / "evaluation/plan.json"
    evaluation = read_json(evaluation_path)
    if (data.digest({k: v for k, v in evaluation.items() if k != "fingerprint"}) != evaluation.get("fingerprint")
            or evaluation["spec_sha256"] != plan["spec_sha256"]
            or evaluation["corpus_manifest_sha256"] != sha(plan["spec"]["corpus_manifest"])):
        raise ValueError("Pinned evaluator/spec/corpus required before candidate generation")
    for path, digest in evaluation["implementation_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Evaluator implementation changed")
    directory = root / "arms" / arm / "generation" / f"tokens_{target_tokens}"
    directory.mkdir(parents=True, exist_ok=False)
    session = None
    reuse = None
    try:
        lease, guard = reservation(plan["profile"]["limit_mib"])
        session, corpus = build_runtime(plan, arm=arm, guard=guard)
        if target_tokens:
            phase = read_json(root / "arms" / arm / "phases" / f"tokens_{target_tokens}" / "result.json")
            committed = phase["checkpoint"]
            session.restore(committed)
            checkpoint = committed["manifest_sha256"]
        else:
            checkpoint = plan["spec"]["initial_checkpoint_sha256"]
        tokenizer = AutoTokenizer.from_pretrained(plan["spec"]["native_model_path"], local_files_only=True, trust_remote_code=False)
        path = Path(corpus["development_records"])
        expected = corpus["sha256"].get(str(path), corpus["sha256"].get("development_records"))
        if expected is None or sha(path) != expected:
            raise ValueError("Development shard changed")
        decoding = evaluation["decoding"]
        if decoding["do_sample"] or decoding["temperature"] != 0.:
            raise ValueError("Deployment evaluation is separately pinned greedy, never TRAIN sampling")
        reused_from = None
        for previous_path in sorted(root.glob("arms/*/generation/*/result.json")):
            previous = read_json(previous_path)
            if (previous.get("status") == "completed" and previous.get("evaluation_plan_sha256") == sha(evaluation_path)
                    and previous.get("development_sha256") == expected
                    and previous.get("identities") == session.identities):
                if sha(previous["responses"]) != previous["responses_sha256"]:
                    raise ValueError("Reusable native responses changed")
                reuse = native_rows(previous["responses"])
                reused_from = dict(receipt=str(previous_path), receipt_sha256=sha(previous_path),
                                   responses=previous["responses"], responses_sha256=previous["responses_sha256"])
                break
        rows = reused_rows = 0
        start = time.monotonic()
        session.model.eval()
        with (directory / "responses.jsonl").open("x") as output:
            for record in data.stream_records(path, allowed_splits=("development",)):
                allowance = min(decoding["max_new_tokens"], session.model.max_length - len(record.prompt_ids))
                if allowance < 1:
                    raise ValueError("Unsupported DEV prompt; no position clamping or prompt truncation")
                for role, runner in (("native", session.teacher.next_logits), ("student", session.model.next_logits)):
                    if role == "native" and reuse is not None:
                        row = next(reuse)
                        if (row["example_id"] != record.example_id or row["group_id"] != record.group_id
                                or row["task"] != record.task or row["input_ids"] != list(record.prompt_ids)
                                or row["checkpoint"] != plan["spec"]["native_model_revision"]
                                or row["decoding"] != decoding or row["grader_identity"] != evaluation["grader_identity"]):
                            raise ValueError("Native response reuse requires identical prompts/teacher/decoding/grader")
                        reused_rows += 1
                    else:
                        residency = rollout_residency(session.model, session.teacher, plan["profile"]) if role == "student" else nullcontext()
                        with residency:
                            generated = generate_prompt(runner, record.prompt_ids, eos_token_ids=decoding["eos_token_ids"],
                                max_new_tokens=allowance, max_length=session.model.max_length,
                                vocabulary_size=session.model.embedding.shape[0], device=session.model.embedding.device)
                        row = generation_row(record, generated, role=role,
                            checkpoint=plan["spec"]["native_model_revision"] if role == "native" else checkpoint,
                            tokenizer=tokenizer, decoding=decoding, grader_identity=evaluation["grader_identity"],
                            geometry=session.identities["geometry"])
                    output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    output.flush()
                    rows += 1
            if reuse is not None and next(reuse, None) is not None:
                raise ValueError("Reused native panel has extra unadmitted records")
            os.fsync(output.fileno())
        proof = session.unchanged_proof()
        result = dict(schema=SCHEMA, status="completed", quality_status="awaiting_cpu_grading", arm=arm,
            target_tokens=target_tokens, checkpoint=checkpoint, plan_fingerprint=plan["fingerprint"], identities=session.identities,
            spec_sha256=plan["spec_sha256"], evaluation_plan=str(evaluation_path), evaluation_plan_sha256=sha(evaluation_path),
            development_records=str(path), development_sha256=expected,
            responses=str(directory / "responses.jsonl"), responses_sha256=sha(directory / "responses.jsonl"),
            rows=rows, native_reused_rows=reused_rows, native_reused_from=reused_from,
            decoding=decoding, decoding_identity=data.digest(decoding),
            frozen_proof=proof, lease=lease, elapsed_seconds=time.monotonic() - start)
        write_json(directory / "result.json", result)
        return result
    except BaseException as error:
        failure(directory, error, session=session)
        raise
    finally:
        if reuse is not None:
            reuse.close()
        if session is not None:
            session.close()
            session.corpus.close()


def check(root):
    cpu_only()
    root = Path(root).resolve()
    plan = load_plan(root, live=True)
    checked, assets = [], {}
    manifests = sorted(root.glob("arms/*/checkpoints/*/manifest.json")) + sorted(root.glob("pilot/*/update_*/manifest.json"))
    for path in manifests:
        manifest = authenticate_checkpoint(path.parent)
        if manifest["identities"]["plan"] != plan["fingerprint"]:
            raise ValueError("Checkpoint belongs to another plan")
        assets.update(manifest["identities"]["consumed_sha256"])
        checked.append(str(path))
    for latest in root.glob("arms/*/latest.json"):
        cp = read_json(latest)
        authenticate_checkpoint(cp["directory"], cp["manifest_sha256"])
    for history in root.glob("arms/*/history/update_*"):
        authenticate_history(history)
    for path in root.glob("arms/*/retention.json"):
        ledger = read_json(path)
        if ledger["status"] != "completed":
            raise ValueError("Retention was interrupted; resume the owned cleanup before phase adoption")
        for row in ledger["pruned"]:
            authenticate_history(row["history"])
    receipts = []
    for path in [root / "pilot/result.json", *sorted(root.glob("arms/*/phases/*/result.json")),
                 *sorted(root.glob("arms/*/generation/*/result.json"))]:
        if not path.is_file():
            continue
        receipt = read_json(path)
        if receipt["plan_fingerprint"] != plan["fingerprint"] or receipt["spec_sha256"] != plan["spec_sha256"]:
            raise ValueError("Phase receipt belongs to another session")
        assets.update(receipt["identities"]["consumed_sha256"])
        for key in ("checkpoint", "baseline_checkpoint"):
            if isinstance(receipt.get(key), dict):
                cp = receipt[key]
                authenticate_checkpoint(cp["directory"], cp["manifest_sha256"], identities=receipt["identities"])
        for key in ("responses", "evaluation_plan", "development_records"):
            if key in receipt:
                digest_key = "development_sha256" if key == "development_records" else key + "_sha256"
                assets[receipt[key]] = receipt[digest_key]
        if receipt.get("native_reused_from"):
            reused = receipt["native_reused_from"]
            assets[reused["receipt"]] = reused["receipt_sha256"]
            assets[reused["responses"]] = reused["responses_sha256"]
        receipts.append(str(path))
    baseline_path = root / "pilot/baseline.json"
    if baseline_path.exists():
        baseline = read_json(baseline_path)
        for key in ("initial", "interruption", "expected"):
            cp = baseline[key]
            authenticate_checkpoint(cp["directory"], cp["manifest_sha256"], identities=baseline["identities"])
        for name in ("initial_logits", "final_logits"):
            assets[str(root / "pilot" / (name + ".pt"))] = baseline[name + "_sha256"]
    for path, digest in assets.items():
        if sha(path) != digest:
            raise ValueError("Consumed runtime artifact changed: " + path)
    return dict(status="authenticated" if receipts else "prepared_only", checkpoints=checked, receipts=receipts,
                plan_fingerprint=plan["fingerprint"], scientific_success=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "pilot", "train", "run", "generate", "check"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--spec")
    parser.add_argument("--arm", choices=("reference", "student_history"))
    parser.add_argument("--target-tokens", type=int)
    parser.add_argument("--original-range", action="store_true")
    parser.add_argument("--proof-stage", choices=("all", "baseline", "resume"), default="all")
    parser.add_argument("--limit-mib", type=int, default=11008)
    parser.add_argument("--host-reserve-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--max-cpu-state-bytes", type=int, default=32 * 1024**3)
    parser.add_argument("--token-chunk-size", type=int, default=8)
    parser.add_argument("--prefix-mode", choices=("streamed", "resident"), default="streamed")
    parser.add_argument("--resident-layers", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        if not args.spec:
            parser.error("prepare requires --spec")
        result = prepare(args.root, args.spec, limit_mib=args.limit_mib, host_reserve_bytes=args.host_reserve_bytes,
                         max_cpu_state_bytes=args.max_cpu_state_bytes, token_chunk_size=args.token_chunk_size,
                         prefix_mode=args.prefix_mode, resident_layers=args.resident_layers)
    elif args.phase == "pilot":
        result = pilot(args.root, original_range=args.original_range, proof_stage=args.proof_stage)
    elif args.phase == "check":
        result = check(args.root)
    else:
        if args.arm is None or args.target_tokens is None or args.original_range:
            parser.error("train/run/generate require --arm and --target-tokens; scientific execution requires extended geometry")
        function = generate if args.phase == "generate" else train
        result = function(args.root, arm=args.arm, target_tokens=args.target_tokens)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
