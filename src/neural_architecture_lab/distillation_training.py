"""Dense response-token and short student-rollout distillation primitives.

No scheduler, implicit GPU admission, corpus selection, or quality certification
is provided here. Teacher providers may read authenticated disk targets, so a
resident full teacher is not required during student backpropagation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DistillationConfig:
    objective: str = "forward_kl"
    temperature: float = 1.
    kd_weight: float = 1.
    ce_weight: float = 0.
    jsd_beta: float = .5
    token_chunk_size: int = 32
    accumulation_batches: int = 1
    max_supervised_tokens: int = 100_000
    max_updates: int = 1_000_000
    grad_clip_norm: float = 1.
    require_all_gradients: bool = True
    budget_policy: str = "complete_batches_only"
    rollout_policy: str = "teacher_forced"
    rollout_student_checkpoint: str | None = None
    rollout_round_id: str | None = None
    max_stale_updates: int = 0

    def __post_init__(self):
        if self.objective not in ("forward_kl", "generalized_jsd"):
            raise ValueError("Forward KL and generalized JSD are separate objectives")
        if (not math.isfinite(self.temperature) or self.temperature <= 0
                or any(not math.isfinite(v) or v < 0 for v in (self.kd_weight, self.ce_weight))
                or self.kd_weight + self.ce_weight <= 0 or not 0 < self.jsd_beta < 1
                or not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm <= 0):
            raise ValueError("Invalid explicit objective coefficients, temperature, or clipping")
        if any(type(v) is not int or v < 1 for v in (self.token_chunk_size,
                self.accumulation_batches, self.max_supervised_tokens, self.max_updates)):
            raise ValueError("Positive integer chunk/update/token budgets required")
        if self.budget_policy != "complete_batches_only":
            raise ValueError("Token budgets must preserve complete response masks, including observed EOS")
        if (self.rollout_policy not in ("teacher_forced", "offline_round")
                or type(self.max_stale_updates) is not int or self.max_stale_updates < 0
                or self.rollout_policy == "offline_round" and not all(isinstance(v, str) and v for v in
                    (self.rollout_student_checkpoint, self.rollout_round_id))
                or self.rollout_policy == "teacher_forced" and any(v is not None for v in
                    (self.rollout_student_checkpoint, self.rollout_round_id))):
            raise ValueError("Captured student rounds require explicit offline policy, identities and update-age bound")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""): digest.update(chunk)
    return digest.hexdigest()


def _tensor_digest(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(_json([str(value.dtype), list(value.shape)]).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def module_fingerprint(module):
    values = {name: _tensor_digest(value) if isinstance(value, torch.Tensor) else value
              for name, value in module.state_dict().items()}
    # Nonpersistent buffers, such as RoPE frequencies, still affect execution.
    values["__all_buffers__"] = {n: _tensor_digest(v) for n, v in module.named_buffers()}
    return hashlib.sha256(_json(values).encode()).hexdigest()


def fixed_fingerprint(module):
    values = {"buffer:" + n: _tensor_digest(v) for n, v in module.named_buffers()}
    values.update({"parameter:" + n: _tensor_digest(v) for n, v in module.named_parameters()
                   if not v.requires_grad})
    return hashlib.sha256(_json(values).encode()).hexdigest()


def _record(record):
    return asdict(record) if is_dataclass(record) else dict(record)


def batch_fingerprint(batch):
    return hashlib.sha256(_json(dict(records=[_record(r) for r in batch["records"]],
        input_ids=_tensor_digest(batch["input_ids"]), attention_mask=_tensor_digest(batch["attention_mask"]),
        loss_mask=_tensor_digest(batch["loss_mask"]))).encode()).hexdigest()


def validate_batch(batch, *, training=True):
    ids, targets, mask, response = (batch[k] for k in ("input_ids", "targets", "loss_mask", "response_mask"))
    attention = batch["attention_mask"].bool()
    if (ids.ndim != 2 or ids.dtype != torch.long or targets.dtype != torch.long
            or any(v.shape != ids.shape for v in (targets, mask, response, attention))
            or mask.dtype != torch.bool or response.dtype != torch.bool or ids.shape[1] < 2
            or len(batch["records"]) != ids.shape[0]):
        raise ValueError("Full-L causal batch, boolean masks, and original records required")
    expected = torch.zeros_like(mask)
    expected[:, :-1] = response[:, 1:] & attention[:, 1:] & attention[:, :-1]
    if not torch.equal(mask, expected) or not torch.equal(targets[:, :-1][mask[:, :-1]], ids[:, 1:][mask[:, :-1]]):
        raise ValueError("Loss mask/targets must predict every eligible response token at its preceding position")
    for i, original in enumerate(batch["records"]):
        record = _record(original); full = list(record["prompt_ids"]) + list(record["response_ids"])
        if (training and record["split"] != "train") or not record["prompt_ids"] or not record["response_ids"]:
            raise ValueError("Only complete, nonempty TRAIN responses may update weights")
        if (int(attention[i].sum()) != len(full) or not bool(attention[i, :len(full)].all())
                or ids[i, :len(full)].tolist() != full
                or int(response[i].sum()) != len(record["response_ids"])
                or not bool(response[i, len(record["prompt_ids"]):len(full)].all())):
            raise ValueError("Batch tokens or complete assistant boundary differ from record")
        eos = record.get("eos_token_id")
        if record["finish_reason"] == "eos" and (eos is None or full[-1] != eos):
            raise ValueError("Observed EOS must be present in the supervised response")
    return int(mask.sum())


class FrozenTeacherProvider:
    """Small-model/provider reference; production can substitute TeacherTargetStore.

    prepare() runs before student activations exist. One frozen hidden batch is
    retained on CPU; readout is evaluated only for the requested token chunk.
    Callbacks own device placement, allowing externally managed sequential use.
    """
    def __init__(self, teacher, hidden_forward, readout, *, teacher_id, temperature=1., guard=None):
        self.teacher = teacher.eval().requires_grad_(False)
        self.hidden_forward, self.readout = hidden_forward, readout
        self.teacher_id, self.temperature, self.guard = teacher_id, temperature, guard
        self.initial_fingerprint = module_fingerprint(teacher)
        self._key = self._hidden = None

    def prepare(self, batch):
        if self.guard is not None: self.guard()
        if self.teacher.training or any(p.requires_grad for p in self.teacher.parameters()):
            raise ValueError("Original teacher must remain frozen and in evaluation mode")
        key = batch_fingerprint(batch)
        if key != self._key:
            with torch.no_grad(): hidden = self.hidden_forward(batch)
            if hidden.shape[:2] != batch["input_ids"].shape or hidden.ndim != 3:
                raise ValueError("Teacher backbone shape mismatch")
            self._hidden, self._key = hidden.detach().cpu(), key

    def __call__(self, batch, positions):
        if batch_fingerprint(batch) != self._key:
            raise ValueError("Teacher.prepare(batch) must precede the student forward")
        positions = positions.cpu()
        with torch.no_grad():
            rows = self._hidden[positions[:, 0], positions[:, 1]]
            logits = self.readout(rows)
        return logits.detach().cpu()

    def validate_unchanged(self):
        if module_fingerprint(self.teacher) != self.initial_fingerprint:
            raise ValueError("Original teacher state changed")
        return True

    def clear_cache(self):
        self._key = self._hidden = None


def loss_vectors(student_logits, teacher_logits, targets, config):
    """Exact full-vocabulary per-token loss; CE uses untempered logits."""
    if (student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape
            or targets.shape != student_logits.shape[:1] or teacher_logits.requires_grad
            or not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all()):
        raise ValueError("Aligned finite full-vocabulary logits and detached teacher targets required")
    dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    s, t = student_logits.to(dtype), teacher_logits.to(device=student_logits.device, dtype=dtype)
    log_s, log_t = F.log_softmax(s / config.temperature, -1), F.log_softmax(t / config.temperature, -1)
    p, q = log_t.exp(), log_s.exp()
    kl = (p * (log_t - log_s)).sum(-1) * config.temperature**2
    distillation = kl
    if config.objective == "generalized_jsd":
        beta = config.jsd_beta
        log_mix = torch.logaddexp(log_t + math.log(beta), log_s + math.log1p(-beta))
        distillation = (beta * (p * (log_t - log_mix)).sum(-1)
            + (1 - beta) * (q * (log_s - log_mix)).sum(-1)) * config.temperature**2
    ce = F.cross_entropy(s, targets.to(s.device), reduction="none")
    return dict(loss=config.kd_weight * distillation + config.ce_weight * ce,
                distillation=distillation, forward_kl=kl, ce=ce)


def topk_tail_kl(student_logits, top_ids, probabilities, tail_mass, *, temperature=1.):
    """APPROXIMATE coarsened KL: selected vocabulary entries plus one tail bucket.

    Full student normalization is retained. This standalone optional objective
    is not silently substituted by DistillationTrainer's exact provider path.
    """
    if (student_logits.ndim != 2 or top_ids.ndim != 2 or top_ids.dtype != torch.long
            or probabilities.shape != top_ids.shape or tail_mass.shape != top_ids.shape[:1]
            or top_ids.shape[0] != student_logits.shape[0] or not 0 < top_ids.shape[1] <= student_logits.shape[1]
            or not math.isfinite(temperature) or temperature <= 0
            or not torch.isfinite(student_logits).all() or not torch.isfinite(probabilities).all()
            or not torch.isfinite(tail_mass).all() or (probabilities < 0).any() or (tail_mass < 0).any()
            or (top_ids < 0).any() or (top_ids >= student_logits.shape[1]).any()
            or any(len(row.unique()) != len(row) for row in top_ids)
            or not torch.allclose(probabilities.sum(-1) + tail_mass, torch.ones_like(tail_mass), atol=1e-6, rtol=1e-6)):
        raise ValueError("Normalized, distinct top-k IDs plus finite tail mass required")
    dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    log_q = F.log_softmax(student_logits.to(dtype) / temperature, -1)
    ids, p, tail = top_ids.to(log_q.device), probabilities.detach().to(log_q), tail_mass.detach().to(log_q)
    result = (p * (p.clamp_min(torch.finfo(dtype).tiny).log() - log_q.gather(1, ids))).sum(-1)
    if ids.shape[1] < log_q.shape[1]:
        omitted = torch.ones_like(log_q, dtype=torch.bool).scatter_(1, ids, False)
        log_tail_q = torch.logsumexp(log_q.masked_fill(~omitted, -torch.inf), -1)
        result += tail * (tail.clamp_min(torch.finfo(dtype).tiny).log() - log_tail_q)
    elif bool((tail != 0).any()):
        raise ValueError("Full-vocabulary top-k has no tail")
    return result * temperature**2


def chunked_backward(hidden, readout, teacher_provider, batch, config, *, normalizer=None):
    """Exact token chunking; backpropagate the shared student backbone once.

    Head gradients accumulate per chunk. The detached selected hidden rows
    accumulate their adjoints, which are then propagated through the original
    backbone. This also handles a tied embedding/readout parameter correctly.
    ``readout`` must be deterministic and independent across rows (for example
    the model's linear vocabulary head), without batch statistics or dropout.
    """
    count = validate_batch(batch)
    if hidden.ndim != 3 or hidden.shape[:2] != batch["input_ids"].shape or count == 0:
        raise ValueError("Student hidden states and nonempty complete response mask required")
    denominator = count if normalizer is None else normalizer
    if type(denominator) is not int or denominator < count: raise ValueError("Invalid token normalization")
    positions = batch["loss_mask"].nonzero().cpu()
    selected = hidden[positions[:, 0].to(hidden.device), positions[:, 1].to(hidden.device)]
    bridge = selected.detach().requires_grad_(True)
    totals = dict(loss=0., distillation=0., forward_kl=0., ce=0., correct=0, teacher_agreement=0)
    for start in range(0, count, config.token_chunk_size):
        pos = positions[start:start + config.token_chunk_size]
        teacher = teacher_provider(batch, pos)
        logits = readout(bridge[start:start + len(pos)])
        target = batch["targets"][pos[:, 0].to(batch["targets"].device), pos[:, 1].to(batch["targets"].device)]
        values = loss_vectors(logits, teacher, target, config)
        (values["loss"].sum() / denominator).backward()
        for name, value in values.items(): totals[name] += float(value.detach().sum())
        totals["correct"] += int((logits.argmax(-1) == target.to(logits.device)).sum())
        totals["teacher_agreement"] += int((logits.argmax(-1) == teacher.to(logits.device).argmax(-1)).sum())
        del logits, teacher, values
    if bridge.grad is None or not torch.isfinite(bridge.grad).all():
        raise ValueError("Readout did not produce finite hidden-state gradients")
    if selected.requires_grad: selected.backward(bridge.grad)
    return dict(totals, supervised_tokens=count)


def _cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return type(value)(_cpu(v) for v in value)
    return value


def save_checkpoint(path, student, optimizer, *, config, identities, updates, supervised_tokens):
    path = Path(path)
    with path.open("xb") as stream:
        torch.save(dict(schema="nal.distillation-checkpoint.v1", student=_cpu(student.state_dict()),
            optimizer=_cpu(optimizer.state_dict()), config=asdict(config), identities=dict(identities),
            updates=updates, supervised_tokens=supervised_tokens,
            model_fingerprint=module_fingerprint(student)), stream)
    return dict(path=str(path.resolve()), sha256=file_sha256(path))


def load_checkpoint(path, student, *, expected_sha256, identities, optimizer=None):
    if file_sha256(path) != expected_sha256: raise ValueError("Checkpoint digest changed")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != "nal.distillation-checkpoint.v1" or state.get("identities") != dict(identities):
        raise ValueError("Checkpoint provenance differs")
    actual = student.state_dict()
    if actual.keys() != state["student"].keys() or any(isinstance(v, torch.Tensor) and
            (v.shape != actual[n].shape or v.dtype != actual[n].dtype) for n, v in state["student"].items()):
        raise ValueError("Checkpoint registration, shapes or precision differ")
    student.load_state_dict(state["student"], strict=True)
    if module_fingerprint(student) != state["model_fingerprint"]:
        raise ValueError("Strict checkpoint reload changed state, including nonpersistent buffers")
    if optimizer is not None: optimizer.load_state_dict(state["optimizer"])
    return state


class DistillationTrainer:
    def __init__(self, student, optimizer, hidden_forward, readout, teacher_provider, *, config, identities, guard=None):
        initialized = time.monotonic()
        self.student, self.optimizer = student, optimizer
        self.hidden_forward, self.readout, self.teacher_provider = hidden_forward, readout, teacher_provider
        self.config, self.identities, self.guard = config, dict(identities), guard
        if set(self.identities) != {"teacher", "student_initializer", "tokenizer", "chat_template", "data"} or any(
                not isinstance(v, str) or not v for v in self.identities.values()):
            raise ValueError("Original teacher, initializer, tokenizer/template and admitted corpus identities required")
        self.parameters = [p for p in student.parameters() if p.requires_grad]
        if not self.parameters: raise ValueError("No intended trainable student parameters")
        if any(p.device.type == "cuda" for p in student.parameters()) and guard is None:
            raise ValueError("CUDA runtime requires caller-owned active reservation guard")
        self._guard()
        if (getattr(teacher_provider, "teacher_id", None) != self.identities["teacher"]
                or getattr(teacher_provider, "temperature", None) != config.temperature):
            raise ValueError("Teacher provider identity or temperature differs")
        if module_fingerprint(student) != self.identities["student_initializer"]:
            raise ValueError("Student differs from authenticated scientific initializer")
        if optimizer.state_dict()["state"]: raise ValueError("Fresh optimizer required; pilot state is not a scientific initializer")
        optimized = [p for group in optimizer.param_groups for p in group["params"]]
        if len(optimized) != len(set(map(id, optimized))) or set(map(id, optimized)) != set(map(id, self.parameters)):
            raise ValueError("Optimizer must own exactly the intended trainable student parameters")
        self.fixed = fixed_fingerprint(student)
        self.initial_parameters = {n: _tensor_digest(p) for n, p in student.named_parameters() if p.requires_grad}
        self.initialization_seconds = time.monotonic() - initialized
        self.finished = False

    def _guard(self):
        if self.guard is not None: self.guard()

    def fit(self, batches: Iterable[Mapping], output_dir, *, progress=None):
        if self.finished: raise RuntimeError("This trainer is single-use; construct a fresh run explicitly")
        self.finished = True
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=False)
        initial = save_checkpoint(directory / "initial.pt", self.student, self.optimizer,
            config=self.config, identities=self.identities, updates=0, supervised_tokens=0)
        if progress is not None:
            progress(dict(phase="initialized", initialization_seconds=self.initialization_seconds,
                devices=sorted({str(p.device) for p in self.student.parameters()}), supervised_tokens=0, update=0))
        started = time.monotonic(); history = []; tokens = inputs = responses = 0
        iterator = iter(batches); done = False; stop_reason = "data_exhausted"
        try:
            while not done and len(history) < self.config.max_updates:
                group = []; counts = []
                for _ in range(self.config.accumulation_batches):
                    try: batch = next(iterator)
                    except StopIteration: done = True; break
                    count = validate_batch(batch)
                    if count < 1: raise ValueError("Empty supervised response batch")
                    for item in batch["records"]:
                        provenance = _record(item)["provenance"]
                        if (provenance["tokenizer_sha256"] != self.identities["tokenizer"] or
                                provenance["chat_template_sha256"] != self.identities["chat_template"]):
                            raise ValueError("Batch tokenizer/template differs from admitted study")
                        rollout = provenance.get("response_provenance") == "student_rollout"
                        if rollout != (self.config.rollout_policy == "offline_round"):
                            raise ValueError("Student rollout versus teacher-forced stage must be declared explicitly")
                        if rollout and (provenance.get("student_checkpoint") != self.config.rollout_student_checkpoint
                                or provenance.get("teacher_id") != self.identities["teacher"]
                                or provenance.get("round_id") != self.config.rollout_round_id
                                or len(history) > self.config.max_stale_updates):
                            raise ValueError("Student rollout provenance or declared stale-update bound differs")
                    if tokens + sum(counts) + count > self.config.max_supervised_tokens:
                        done = True; stop_reason = "token_budget"; break
                    group.append(batch); counts.append(count)
                if not group: break
                self._guard(); self.student.train(); self.optimizer.zero_grad(set_to_none=True)
                step_started = time.monotonic(); total = sum(counts); values = []
                for batch in group:
                    self._guard()
                    prepare = getattr(self.teacher_provider, "prepare", None)
                    if prepare is not None: prepare(batch)  # Teacher can offload BEFORE student graph exists.
                    hidden = self.hidden_forward(batch)
                    values.append(chunked_backward(hidden, self.readout, self.teacher_provider,
                        batch, self.config, normalizer=total))
                    del hidden
                    inputs += int(batch["attention_mask"].sum()); responses += int(batch["response_mask"].sum())
                present = [name for name, p in self.student.named_parameters() if p.requires_grad and p.grad is not None]
                if self.config.require_all_gradients and len(present) != len(self.parameters):
                    raise ValueError("An intended student parameter received no gradient")
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters):
                    raise ValueError("Nonfinite student gradient; optimizer update refused")
                norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.grad_clip_norm, error_if_nonfinite=True)
                self._guard(); self.optimizer.step(); tokens += total
                if any(not torch.isfinite(p).all() for p in self.parameters): raise ValueError("Nonfinite updated parameter")
                row = dict(update=len(history) + 1, supervised_tokens=total, cumulative_supervised_tokens=tokens,
                    input_tokens=inputs, response_tokens=responses, gradient_norm=float(norm),
                    gradient_parameters=present, step_seconds=time.monotonic()-step_started,
                    **{name: sum(v[name] for v in values) / total for name in
                       ("loss", "distillation", "forward_kl", "ce", "correct", "teacher_agreement")})
                history.append(row)
                if progress is not None: progress(dict(phase="training", **row, elapsed_seconds=time.monotonic()-started))
                if tokens == self.config.max_supervised_tokens: stop_reason = "token_budget"; break
            if len(history) == self.config.max_updates: stop_reason = "max_updates"
            if not history: raise ValueError("No complete response batch fits the declared training budget")
            if fixed_fingerprint(self.student) != self.fixed: raise ValueError("Fixed student state changed")
            teacher_check = getattr(self.teacher_provider, "validate_unchanged", None)
            teacher_unchanged = None if teacher_check is None else bool(teacher_check())
            if teacher_unchanged is False: raise ValueError("Teacher provider reported changed original state")
            self.student.eval(); final = save_checkpoint(directory / "final.pt", self.student, self.optimizer,
                config=self.config, identities=self.identities, updates=len(history), supervised_tokens=tokens)
            before = module_fingerprint(self.student)
            load_checkpoint(final["path"], self.student, expected_sha256=final["sha256"], identities=self.identities)
            result = dict(status="systems_passed", quality_status="systems_only", confirmed=False,
                config=asdict(self.config), identities=self.identities, updates=len(history), supervised_tokens=tokens,
                input_tokens=inputs, response_tokens=responses, stop_reason=stop_reason, history=history,
                initial_checkpoint=initial, final_checkpoint=final, fixed_state_unchanged=True,
                teacher_unchanged=teacher_unchanged, teacher_provider=type(self.teacher_provider).__name__,
                in_process_reload_exact=module_fingerprint(self.student) == before,
                changed_parameters=[n for n, p in self.student.named_parameters()
                    if n in self.initial_parameters and _tensor_digest(p) != self.initial_parameters[n]],
                initialization_seconds=self.initialization_seconds,
                devices=sorted({str(p.device) for p in self.student.parameters()}),
                elapsed_seconds=time.monotonic()-started,
                scope="Runtime validation only; no checkpoint quality selection, fresh-process proof, or recovery claim")
            with (directory / "result.json").open("x") as stream: stream.write(_json(result) + "\n")
            return result
        except BaseException as error:
            with (directory / "failure.json").open("x") as stream:
                stream.write(_json(dict(status="execution_failure", error=repr(error),
                    completed_updates=len(history), supervised_tokens=tokens)) + "\n")
            raise
        finally:
            clear = getattr(self.teacher_provider, "clear_cache", None)
            if clear is not None: clear()


def repetition_report(tokens, *, ngram_size=4, minimum_repeats=3):
    """Flag long adjacent repeated token blocks, not isolated repeated words.

    Heuristic evidence only: no tokens are removed and no rollout cap changes.
    """
    if type(ngram_size) is not int or ngram_size < 1 or type(minimum_repeats) is not int or minimum_repeats < 2:
        raise ValueError("Positive n-gram width and at least two adjacent repeats required")
    longest = 0; first = None
    for start in range(max(0, len(tokens) - ngram_size + 1)):
        block = tokens[start:start + ngram_size]; count = 1
        while tokens[start + count*ngram_size:start + (count+1)*ngram_size] == block:
            count += 1
        if count > longest: longest, first = count, start
    return dict(flagged=longest >= minimum_repeats, ngram_size=ngram_size,
        minimum_repeats=minimum_repeats, longest_adjacent_repetitions=longest,
        first_longest_start=first, suffix_removed=False, adaptive_cap=False,
        scope="Conservative token-block heuristic, not a semantic collapse verdict")


@torch.inference_mode()
def collect_rollouts(records, next_logits: Callable, *, student_checkpoint, teacher_id, round_id,
                     eos_token_id, max_new_tokens=32, max_length=416, device="cpu", seed=0, guard=None):
    """Greedy short TRAIN continuations; teacher providers later score exact prefixes.

    Each call declares a new round. Replaying its records after updates is
    offline/stale-policy training, not a freshly sampled on-policy round.
    """
    if (type(max_new_tokens) is not int or not 0 < max_new_tokens <= 128
            or type(max_length) is not int or max_length < 2 or type(eos_token_id) is not int
            or eos_token_id < 0 or type(seed) is not int
            or not all(isinstance(v, str) and v for v in (student_checkpoint, teacher_id, round_id))):
        raise ValueError("Bounded short rollout and explicit model/round identities required")
    device = torch.device(device)
    if device.type == "cuda" and guard is None: raise ValueError("CUDA rollout requires a reservation guard")
    if guard is not None: guard()
    generated = []; receipts = []
    for record in records:
        item = _record(record); prompt = list(item["prompt_ids"])
        if item["split"] != "train" or not prompt or len(prompt) + max_new_tokens > max_length:
            raise ValueError("Only untruncated TRAIN prompts within validated capacity may roll out")
        response = []; finish = "length"
        for _ in range(max_new_tokens):
            if guard is not None: guard()
            logits = next_logits(torch.tensor([prompt + response], dtype=torch.long, device=device))
            if logits.ndim == 2 and logits.shape[0] == 1: logits = logits[0]
            if logits.ndim != 1 or logits.numel() < 2 or not torch.isfinite(logits).all():
                raise ValueError("Invalid singleton full-vocabulary rollout logits")
            token = int(logits.argmax()); response.append(token)
            if token == eos_token_id: finish = "eos"; break
        settings = dict(do_sample=False, max_new_tokens=max_new_tokens, max_length=max_length,
                        eos_token_id=eos_token_id, temperature=1.)
        provenance = dict(item["provenance"], response_provenance="student_rollout",
            generator_model=student_checkpoint, student_checkpoint=student_checkpoint,
            teacher_id=teacher_id, round_id=round_id, decoding=settings, seed=seed)
        value = replace(record, example_id=item["example_id"] + "/rollout/" + round_id,
            response_ids=tuple(response), provenance=provenance, finish_reason=finish, eos_token_id=eos_token_id)
        generated.append(value)
        receipts.append(dict(example_id=value.example_id, prompt_ids=prompt, response_ids=response,
            finish_reason=finish, provenance=provenance, repetition=repetition_report(response)))
    return generated, dict(schema="nal.student-rollout-round.v1", round_id=round_id,
        student_checkpoint=student_checkpoint, teacher_id=teacher_id, decoding="greedy",
        seed=seed, records=receipts, supervised_tokens=sum(len(r.response_ids) for r in generated),
        prefixes_sha256=hashlib.sha256(_json(receipts).encode()).hexdigest(),
        scope="Captured student contexts; reuse after updates is offline/stale-policy and requires an explicit refresh bound")
