"""Response-only causal supervision and authenticated streamed teacher targets.

This module never tokenizes, truncates, generates an EOS, or opens a test corpus
implicitly. The caller supplies tokens from its pinned template and keeps test
files unavailable to the training worker. Bucketing pads separate documents;
it does not concatenate them into an incorrectly unmasked packed sequence.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Iterable, Mapping

import numpy as np
import torch


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""): h.update(chunk)
    return h.hexdigest()


def _identity(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


@dataclass(frozen=True)
class ResponseRecord:
    example_id: str
    group_id: str
    split: str
    task: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    provenance: Mapping
    finish_reason: str
    eos_token_id: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "prompt_ids", tuple(self.prompt_ids))
        object.__setattr__(self, "response_ids", tuple(self.response_ids))
        # Detach from the caller's mutable metadata, retaining JSON serialization.
        object.__setattr__(self, "provenance", json.loads(json.dumps(self.provenance, allow_nan=False)))
        if any(not isinstance(x, str) or not x for x in (self.example_id, self.group_id, self.task)):
            raise ValueError("Stable example/group/task identities required")
        if self.split not in {"train", "development", "test"}:
            raise ValueError("Explicit train/development/test split required")
        if not self.prompt_ids or not self.response_ids or any(type(t) is not int or t < 0 for t in self.input_ids):
            raise ValueError("Nonempty prompt and response with nonnegative integer token IDs required")
        required = {"source_uri", "source_revision", "tokenizer_sha256", "chat_template_sha256",
                    "prompt_provenance", "response_provenance"}
        if any(not self.provenance.get(k) for k in required) or self.provenance.get("assistant_boundary_verified") is not True:
            raise ValueError("Source, tokenizer/template, prompt/response and verified assistant boundary provenance required")
        prefix = self.provenance.get("tokenized_prefix_sha256")
        if prefix is not None and prefix != digest(self.prompt_ids):
            raise ValueError("Verified assistant prefix differs from actual prompt IDs")
        if self.finish_reason not in {"eos", "stop", "length"}:
            raise ValueError("Explicit eos/stop/length finish reason required")
        if self.eos_token_id is not None and (type(self.eos_token_id) is not int or self.eos_token_id < 0):
            raise ValueError("EOS must be an actual nonnegative tokenizer ID")
        if self.finish_reason == "eos" and (self.eos_token_id is None or self.response_ids[-1] != self.eos_token_id):
            raise ValueError("EOS completion requires the observed terminal token; never append one")
        if self.eos_token_id is not None and self.eos_token_id in self.response_ids[:-1]:
            raise ValueError("Response continues after an observed EOS")
        if self.finish_reason == "length" and self.response_ids[-1] == self.eos_token_id:
            raise ValueError("Token-cap termination conflicts with an observed EOS")
        if self.provenance["response_provenance"] in {"generated", "student_rollout"}:
            if (not self.provenance.get("generator_model") or not isinstance(self.provenance.get("decoding"), dict)
                    or type(self.provenance.get("seed")) is not int):
                raise ValueError("Generated response requires model/version, decoding settings and seed")
        if self.provenance["response_provenance"] == "student_rollout":
            if self.split != "train" or any(not self.provenance.get(k) for k in ("student_checkpoint", "teacher_id", "round_id")):
                raise ValueError("Student rollout requires TRAIN, student/teacher checkpoints and round identity")

    @property
    def input_ids(self):
        return self.prompt_ids + self.response_ids

    @property
    def prediction_positions(self):
        return tuple(range(len(self.prompt_ids) - 1, len(self.input_ids) - 1))


def record_from_tokenized(*, full_ids, assistant_boundary, expected_prompt_ids, provenance, **fields):
    """Verify the boundary against the actual pinned-template token sequence.

    ``expected_prompt_ids`` must be the independently rendered generation prefix,
    not a character boundary or an assumed delimiter token.
    """
    full_ids, expected_prompt_ids = tuple(full_ids), tuple(expected_prompt_ids)
    if (type(assistant_boundary) is not int or assistant_boundary != len(expected_prompt_ids)
            or not 0 < assistant_boundary < len(full_ids) or full_ids[:assistant_boundary] != expected_prompt_ids):
        raise ValueError("Actual tokenized assistant boundary does not match the generation prefix")
    proof = dict(provenance, assistant_boundary_verified=True, tokenized_prefix_sha256=digest(expected_prompt_ids))
    return ResponseRecord(prompt_ids=expected_prompt_ids, response_ids=full_ids[assistant_boundary:], provenance=proof, **fields)


def record_fingerprint(record: ResponseRecord) -> str:
    return digest(asdict(record))


class SplitGuard:
    """One-pass group and exact-token duplicate guard; not a paraphrase detector."""
    def __init__(self):
        self.groups, self.prompts, self.contents, self.ids = {}, {}, {}, set()
        self.tokenization = None
        self.counts = Counter()

    def add(self, record: ResponseRecord):
        tokenization = (record.provenance["tokenizer_sha256"], record.provenance["chat_template_sha256"])
        if self.tokenization is not None and tokenization != self.tokenization:
            raise ValueError("Corpus tokenizer/template identity changed")
        self.tokenization = tokenization
        if record.example_id in self.ids: raise ValueError("Repeated example ID")
        for table, key, kind in ((self.groups, record.group_id, "source group"),
                                 (self.prompts, digest(record.prompt_ids), "exact prompt"),
                                 (self.contents, digest(record.input_ids), "exact content")):
            if key in table and table[key] != record.split:
                raise ValueError(f"Cross-split {kind} leakage")
            table[key] = record.split
        self.ids.add(record.example_id)
        self.counts[record.split] += 1


def dataset_fingerprint(records: Iterable[ResponseRecord]) -> str:
    guard, h = SplitGuard(), hashlib.sha256()
    for record in records:
        guard.add(record)
        h.update((record_fingerprint(record) + "\n").encode())
    if not guard.ids: raise ValueError("Empty response corpus")
    return h.hexdigest()


def stream_records(path, *, allowed_splits=("train", "development")):
    """Stream an explicitly admitted JSONL file; never discover/open other splits."""
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split") not in allowed_splits: raise ValueError("Unadmitted split in supplied file")
            yield ResponseRecord(**row)


def length_buckets(records, *, batch_size, max_length, buffer_size=256, seed=0):
    """Bounded, deterministic length bucketing; each record appears exactly once."""
    if batch_size < 1 or buffer_size < batch_size or max_length < 2:
        raise ValueError("Invalid bucket/batch/context limits")
    rng, pending, split = random.Random(seed), [], None
    def batches(values):
        values.sort(key=lambda r: (len(r.input_ids), r.example_id))
        groups = [values[i:i+batch_size] for i in range(0, len(values), batch_size)]
        rng.shuffle(groups)
        return groups
    for record in records:
        if len(record.input_ids) > max_length: raise ValueError("Oversize example rejected; no silent truncation")
        if split is not None and record.split != split: raise ValueError("Bucket one declared split at a time")
        split = record.split; pending.append(record)
        if len(pending) == buffer_size:
            yield from batches(pending); pending = []
    if pending: yield from batches(pending)


def response_batch(records, *, pad_token_id, max_length, device="cpu", ignore_index=-100):
    """FULL-L tensors with both target-token and predicting-logit masks.

    ``response_mask[b,t]`` selects a token at t. ``loss_mask[b,t-1]`` selects
    its predicting logit. ``targets[b,t-1]`` contains that same token. Thus a
    consumer uses logits[loss_mask] and targets[loss_mask] with NO second shift.
    """
    records = list(records)
    if not records or len({r.split for r in records}) != 1: raise ValueError("Nonempty single-split batch required")
    if type(pad_token_id) is not int or pad_token_id < 0: raise ValueError("Pinned padding token required")
    lengths = [len(r.input_ids) for r in records]; length = max(lengths)
    if length > max_length: raise ValueError("Oversize example rejected; no silent truncation")
    shape = (len(records), length)
    ids = torch.full(shape, pad_token_id, dtype=torch.long)
    attention = torch.zeros(shape, dtype=torch.bool); response = attention.clone()
    targets = torch.full(shape, ignore_index, dtype=torch.long); loss = attention.clone()
    for i, record in enumerate(records):
        n, start = lengths[i], len(record.prompt_ids)
        ids[i, :n] = torch.tensor(record.input_ids)
        attention[i, :n] = True; response[i, start:n] = True
        targets[i, :n-1] = ids[i, 1:n]; loss[i, start-1:n-1] = True
    counts = dict(input_tokens=sum(lengths), prompt_tokens=sum(len(r.prompt_ids) for r in records),
                  response_tokens=sum(len(r.response_ids) for r in records), supervised_tokens=int(loss.sum()))
    return dict(input_ids=ids.to(device), attention_mask=attention.to(device),
                position_ids=(attention.long().cumsum(1)-1).clamp_min(0).to(device),
                response_mask=response.to(device), targets=targets.to(device), loss_mask=loss.to(device),
                example_ids=[r.example_id for r in records], records=records, lengths=lengths, counts=counts)


def target_contract(record, *, teacher_id, temperature, vocabulary_size, positions=None):
    positions = record.prediction_positions if positions is None else tuple(positions)
    if not teacher_id or not math.isfinite(temperature) or temperature <= 0 or type(vocabulary_size) is not int or vocabulary_size < 2:
        raise ValueError("Frozen teacher, positive temperature and vocabulary required")
    if (not positions or len(set(positions)) != len(positions)
            or any(type(p) is not int or p not in record.prediction_positions for p in positions)):
        raise ValueError("Unique assistant predicting-logit positions required")
    if any(t >= vocabulary_size for t in record.input_ids): raise ValueError("Token outside teacher vocabulary")
    if record.provenance["response_provenance"] == "student_rollout" and record.provenance["teacher_id"] != teacher_id:
        raise ValueError("Rollout scoring teacher differs from the declared round teacher")
    return dict(schema="nal.response-teacher-targets.v1", record_sha256=record_fingerprint(record),
                example_id=record.example_id, teacher_id=teacher_id, temperature=float(temperature),
                vocabulary_size=vocabulary_size, positions=list(positions),
                tokenizer_sha256=record.provenance["tokenizer_sha256"],
                chat_template_sha256=record.provenance["chat_template_sha256"],
                response_token_count=len(record.response_ids), supervised_position_count=len(positions),
                coverage="complete_response" if set(positions) == set(record.prediction_positions) else "declared_subset",
                full_input_sha256=digest(record.input_ids),
                prefix_sha256=[digest(record.input_ids[:p+1]) for p in positions])


def write_target_cache(path, record, *, teacher_id, temperature, vocabulary_size, logits=None,
                       positions=None, topk_ids=None, probabilities=None, tail_mass=None):
    """Write one immutable record shard. Top-k is an explicitly coarse target.

    Full logits are FP32. Top-k probabilities/tail are already evaluated at the
    recorded temperature. The caller must compare its approximate objective
    against the exact reference; this serializer does not claim equivalence.
    """
    path = Path(path)
    contract = target_contract(record, teacher_id=teacher_id, temperature=temperature,
                               vocabulary_size=vocabulary_size, positions=positions)
    n = len(contract["positions"])
    def array(value, dtype):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if dtype == np.float32: value = value.float()
            value = value.numpy()
        return np.asarray(value, dtype=dtype)
    if logits is not None:
        if any(x is not None for x in (topk_ids, probabilities, tail_mass)): raise ValueError("Choose full OR top-k target format")
        values = {"logits": array(logits, np.float32)}
        if values["logits"].shape != (n, vocabulary_size) or not np.isfinite(values["logits"]).all():
            raise ValueError("Finite full-vocabulary logits at every declared position required")
        form = "full_logits_fp32"
    else:
        if any(x is None for x in (topk_ids, probabilities, tail_mass)): raise ValueError("Top-k IDs, probabilities and tail mass required")
        raw_ids = topk_ids.detach().cpu().numpy() if isinstance(topk_ids, torch.Tensor) else np.asarray(topk_ids)
        if raw_ids.dtype.kind not in "iu": raise ValueError("Top-k token IDs must be integers")
        ids, probs, tail = array(raw_ids, np.int64), array(probabilities, np.float32), array(tail_mass, np.float32)
        if (ids.ndim != 2 or ids.shape[0] != n or not 0 < ids.shape[1] < vocabulary_size
                or probs.shape != ids.shape or tail.shape != (n,) or (ids < 0).any() or (ids >= vocabulary_size).any()
                or any(len(set(row.tolist())) != len(row) for row in ids)
                or not np.isfinite(probs).all() or not np.isfinite(tail).all() or (probs < 0).any() or (tail < 0).any()
                or not np.allclose(probs.sum(1) + tail, 1., atol=2e-6, rtol=2e-6)):
            raise ValueError("Valid unique top-k support and normalized nonnegative tail mass required")
        values, form = {"topk_ids": ids, "probabilities": probs, "tail_mass": tail}, "topk_plus_tail_approximation"
    path.mkdir(parents=True, exist_ok=False)
    for name, value in values.items():
        with (path/(name+".npy")).open("xb") as stream: np.save(stream, value, allow_pickle=False)
    manifest = dict(status="completed", contract=contract, format=form,
                    artifacts={name+".npy": _sha(path/(name+".npy")) for name in values})
    manifest["fingerprint"] = digest(manifest)
    _json(path/"manifest.json", manifest)
    return manifest


class TeacherTargetStore:
    """Callable full-logit provider; hash each shard once, check identity thereafter.

    ``positions[N,2]`` contains (batch row, predicting-logit position). Rows are
    returned in exactly that order, without shifting or silently dropping EOS.
    Full shards are mmap-read only for requested rows, then promptly closed.
    """
    def __init__(self, index, *, teacher_id, temperature, vocabulary_size):
        self.index = dict(index)
        self.teacher_id, self.temperature, self.vocabulary_size = teacher_id, temperature, vocabulary_size
        self._admitted = {}

    def _admit(self, record):
        key = record_fingerprint(record)
        if key not in self.index: raise ValueError("No targets for this exact record/student prefix provenance")
        path = Path(self.index[key])
        if key not in self._admitted:
            manifest = json.loads((path/"manifest.json").read_text())
            if (manifest.get("status") != "completed" or manifest.get("format") != "full_logits_fp32"
                    or manifest.get("fingerprint") != digest({k:v for k,v in manifest.items() if k != "fingerprint"})
                    or set(manifest.get("artifacts", {})) != {"logits.npy"}):
                raise ValueError("Complete exact full-logit cache required; top-k is a separate approximation")
            expected = target_contract(record, teacher_id=self.teacher_id, temperature=self.temperature,
                vocabulary_size=self.vocabulary_size, positions=manifest["contract"]["positions"])
            if manifest["contract"] != expected: raise ValueError("Teacher/prefix/tokenizer/temperature target provenance mismatch")
            if _sha(path/"logits.npy") != manifest["artifacts"]["logits.npy"]: raise ValueError("Teacher cache payload changed")
            self._admitted[key] = (manifest, {name: _identity(path/name) for name in ("manifest.json", "logits.npy")})
        manifest, identities = self._admitted[key]
        if any(_identity(path/name) != identity for name, identity in identities.items()):
            raise ValueError("Previously admitted teacher cache changed")
        return path, manifest

    def __call__(self, batch, positions):
        if isinstance(positions, torch.Tensor): positions = positions.detach().cpu().tolist()
        positions = list(positions)
        result = torch.empty((len(positions), self.vocabulary_size), dtype=torch.float32)
        groups = {}
        for out, pair in enumerate(positions):
            if len(pair) != 2 or any(type(p) is not int for p in pair): raise ValueError("Expected [batch, predicting-position] integer pairs")
            row, position = pair
            if not 0 <= row < len(batch["records"]) or position not in batch["records"][row].prediction_positions:
                raise ValueError("Requested target is not an assistant prediction")
            groups.setdefault(row, []).append((out, position))
        for row, requested in groups.items():
            record = batch["records"][row]
            actual = batch["input_ids"][row, :len(record.input_ids)].detach().cpu().tolist()
            if tuple(actual) != record.input_ids: raise ValueError("Batch context tokens differ from cache request")
            if "attention_mask" in batch:
                mask = batch["attention_mask"][row].detach().cpu().tolist()
                if mask != [True]*len(record.input_ids)+[False]*(len(mask)-len(record.input_ids)):
                    raise ValueError("Batch attention context differs from cache request")
            if "position_ids" in batch and batch["position_ids"][row,:len(record.input_ids)].detach().cpu().tolist() != list(range(len(record.input_ids))):
                raise ValueError("Batch position IDs differ from cache request")
            path, manifest = self._admit(record); lookup = {p:i for i,p in enumerate(manifest["contract"]["positions"])}
            if any(p not in lookup for _,p in requested): raise ValueError("Cache omits requested response positions")
            bank = np.load(path/"logits.npy", mmap_mode="r", allow_pickle=False)
            try:
                if bank.dtype != np.float32 or bank.shape != (len(lookup), self.vocabulary_size): raise ValueError("Teacher bank shape/dtype changed")
                values = np.array(bank[[lookup[p] for _,p in requested]], copy=True)
                if not np.isfinite(values).all(): raise ValueError("Nonfinite teacher target")
                result[[out for out,_ in requested]] = torch.from_numpy(values)
            finally: bank._mmap.close()
        return result
