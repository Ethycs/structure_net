"""Fresh, bounded progressive response recovery with packed-only qualification.

One native block is compiled before this worker starts. Only that new block is
locally aligned to native residual targets; joint recovery then trains the
compact suffix against the original teacher. Existing campaigns are untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch

from experiments.neural_architecture_lab.distillation_recovery import module
from . import jspace_3b_rebuild_data as data_source
from . import jspace_3b_rebuild_fit_campaign as fit_campaign
from . import jspace_3b_response_cache as cache
from . import jspace_3b_response_train as training
from . import jspace_3b_response_eval as evaluation
from . import jspace_3b_response_judge as judge
from . import jspace_3b_response_task_metrics as task_metrics
from . import jspace_3b_response_model as model_source
from . import jspace_3b_packed_response_model as packed_source
from . import jspace_3b_packed_response_evaluate as packed_evaluation
from . import jspace_qwen3b_progressive_model as builder
from . import jspace_qwen3_train as offload
from . import jspace_3b_final_evaluate as execution
from .jspace_qwen_affine_stack import forward_reduced
from .jspace_qwen_parallel_train import cpu_module_clone

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/"docs/07 - Status Reports/jspace/2026-11-16_qwen3b-progressive-response-rebuild.md"
STANDARD = ROOT/"docs/05 - Standards/NAL-STD-DISTILLATION-v1.md"
SCHEMA = "jspace.3b-progressive-response-recovery.v1"
read, write, sha, identity = cache.read, cache.write, cache.sha, cache.identity
nal = module("distillation_recovery")


@dataclass(frozen=True)
class LocalConfig:
    learning_rate: float = 2e-5
    accumulation_examples: int = 16
    gradient_clip: float = 1.
    optimizer_chunk_bytes: int = 4*1024**2
    max_cpu_state_bytes: int = 4*1024**3

    def __post_init__(self):
        if (not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or self.accumulation_examples < 1 or self.gradient_clip <= 0):
            raise ValueError("Positive local optimization settings required")


def _event(directory, phase, **values):
    row = dict(phase=phase, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **values)
    with (Path(directory)/"progress.jsonl").open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n"); stream.flush()
    print(json.dumps(row, allow_nan=False), flush=True)


def _sources():
    files = [Path(__file__), PROTOCOL, STANDARD,
        ROOT/"tests/structure_net/test_jspace_3b_rebuild_recovery.py",
        ROOT/"experiments/structure_net/jspace_3b_rebuild_fit.py",
        ROOT/"experiments/neural_architecture_lab/distillation_recovery.py"]
    files.extend(ROOT/"src/neural_architecture_lab"/(name+".py") for name in
        ("distillation_data", "distillation_training", "distillation_recovery", "core", "local_scheduler"))
    return cache.prior.source.inherited.local_import_closure(files)


def _parent(checkpoint, cut, data_fingerprint):
    if checkpoint is None:
        if cut != 35: raise ValueError("Only cut35 can start from the original native model")
        return None
    checkpoint = Path(checkpoint).resolve()
    root = checkpoint.parents[2]
    p = read(root/"plan.json"); result = read(root/"run/result.json")
    body = dict(p); fingerprint = body.pop("fingerprint")
    selected = result.get("selected") or {}
    if (p.get("schema") != SCHEMA or cache.data.digest(body) != fingerprint or p["cut"] != cut+1
            or p["data_fingerprint"] != data_fingerprint or not result.get("relative_retention_pass")
            or result.get("fingerprint") != fingerprint
            or selected.get("checkpoint", {}).get("path") != str(checkpoint)
            or selected["checkpoint"]["sha256"] != sha(checkpoint)):
        raise ValueError("An adjacent, actually qualified parent checkpoint from this rebuild is required")
    manifest = read(root/"run/manifest.json")
    if (manifest.get("status") != "completed" or manifest.get("fingerprint") != fingerprint
            or manifest["artifacts"].get("result.json") != sha(root/"run/result.json")):
        raise ValueError("Parent completion receipt is missing or inconsistent")
    return dict(root=str(root), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                fingerprint=fingerprint, fits=p["fits"], result_sha256=sha(root/"run/result.json"))


def prepare(root, data_root, fit_root, cut, parent_checkpoint=None):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Preparation requires CUDA hidden")
    if cut not in (35, 34): raise ValueError("This prospective pilot admits cuts35 and34 only")
    root, data_root, fit_root = map(lambda p:Path(p).resolve(), (root, data_root, fit_root))
    if root.exists(): raise FileExistsError(root)
    dp, records = data_source.verify(data_root, live=False)
    if cut not in dp["cuts"] or dp["mc"] is None:
        raise ValueError("Both response and original MC boundary controls are required")
    inputs = data_source._manifest_files(data_root/"native", dp["fingerprint"])
    inputs[str(data_root/"plan.json")] = sha(data_root/"plan.json")
    parent = _parent(parent_checkpoint, cut, dp["fingerprint"])
    # Accept the fit wrapper root or its explicit work/ directory, but require
    # its completed admission record, not merely plausible same-cut fit files.
    fit_campaign_root = fit_root.parent if (fit_root/"fit_receipt.json").is_file() else fit_root
    fit_result = fit_campaign.check(fit_campaign_root)
    fp = fit_campaign.verify(fit_campaign_root, live=False)
    if (fit_result.get("status") != "completed" or fp["data_fingerprint"] != dp["fingerprint"]
            or Path(fp["data_root"]).resolve() != data_root or fp["cut"] != cut or fp["model"] != dp["model"]
            or (None if fp["parent"] is None else fp["parent"]["checkpoint_sha256"])
                != (None if parent is None else parent["checkpoint_sha256"])):
        raise ValueError("Fresh fit campaign is not bound to this data, cut, native model and accepted parent")
    fit_root = Path(fit_result["fit_receipt"]).parent.resolve()
    fit = read(fit_root/"fit_receipt.json")
    if (fit.get("status") != "completed" or fit.get("cut") != cut
            or not fit.get("native_seam_bit_exact") or not fit.get("saved_fit_compilation_passed")
            or fit.get("teacher_fingerprint") != fp["teacher_fingerprint"]
            or fit.get("parent_checkpoint_sha256") != (parent["checkpoint_sha256"] if parent else None)):
        raise ValueError("Fresh complete fit and original-native seam proof required")
    for name, digest in fit["artifacts"].items():
        path = fit_root/name
        if not path.is_relative_to(fit_root) or sha(path) != digest:
            raise ValueError("Fresh fit artifact differs from its completion receipt")
        inputs[str(path)] = digest
    inputs[str(fit_root/"fit_receipt.json")] = sha(fit_root/"fit_receipt.json")
    for name in ("plan.json", "result.json"):
        path = fit_campaign_root/name; inputs[str(path)] = sha(path)
    fits = dict(parent["fits"] if parent else {})
    fits[str(cut)] = fit["fit_directory"]
    if parent:
        for name in ("plan.json", "run/result.json", "run/manifest.json"):
            path = Path(parent["root"])/name; inputs[str(path)] = sha(path)
        inputs[parent["checkpoint"]] = parent["checkpoint_sha256"]
        # Inherited geometry is needed for strict builder reconstruction too.
        for directory in parent["fits"].values():
            for name in ("fit.pt", "fit.json", "relation.pt", "relations.json", "interface.json"):
                path = Path(directory)/name; inputs[str(path)] = sha(path)
    local_indices = dp["local_alignment_indices"]
    if (len(local_indices) != 256 or len(set(local_indices)) != 256
            or any(records[i].split != "train" for i in local_indices)):
        raise ValueError("Exactly256 unique local TRAIN examples required")
    p = dict(schema=SCHEMA, cut=cut, data_root=str(data_root), data_fingerprint=dp["fingerprint"],
        fit_root=str(fit_root), fit_campaign_root=str(fit_campaign_root), fits=fits, parent=parent, model=dp["model"],
        corpus=dp["corpus"], corpus_fingerprint=dp["records_sha256"],
        local_configuration=asdict(LocalConfig()), local_alignment_indices=local_indices,
        local_alignment_windows=dp["local_alignment_windows"],
        local_normalization="mean native next-boundary squared energy over local TRAIN response prediction positions, FP32",
        training=dp["training"], checkpoints=dp["recovery_checkpoints"],
        configuration=asdict(training.ResponseConfig(max_supervised_tokens=dp["recovery_response_tokens"])),
        generation_indices=dp["generation_indices"], generation_eos_token_ids=dp["generation_eos_token_ids"],
        max_new_tokens=128, prefix_mode="resident", seed=dp["training"]["seed"],
        judge=asdict(judge.JudgeConfig(workers=1, expected_returned_model="codex_sdk/gpt-5.6-sol")),
        gates=dict(maximum_drop_pp=5., maximum_family_drop_pp=10., minimum_generation_rate=0.,
                   maximum_native_nll_ratio=1.10),
        gating_order="pack/export/reload -> DEV NLL and MC -> generation/semantic/executable if cheap gates pass",
        selection="Eligible highest paired semantic task success, lowest NLL, earliest response token checkpoint",
        confirmation="Exploratory reused DEV retention, not independent application-quality certification",
        input_sha256=inputs, input_identity={path:identity(path) for path in inputs},
        source_sha256={str(path):sha(path) for path in sorted(_sources())},
        allocation_mib=11264, limit_mib=11008, required_discarded_pilot=True, confirmed=False)
    p["fingerprint"] = cache.data.digest(p)
    root.mkdir(parents=True); write(root/"plan.json", p)
    for path, digest in p["source_sha256"].items():
        target = root/"producer"/Path(path).relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        if sha(target) != digest: raise ValueError("Producer archive differs")
    return dict(status="prepared", fingerprint=p["fingerprint"], cut=cut,
                updates=len(p["training"]["windows"]), response_tokens=dp["recovery_response_tokens"])


def verify(root, *, live=True):
    root = Path(root); p = read(root/"plan.json"); body = dict(p); fingerprint = body.pop("fingerprint")
    if p.get("schema") != SCHEMA or cache.data.digest(body) != fingerprint:
        raise ValueError("Rebuild recovery plan changed")
    for path, old in p["input_identity"].items():
        if identity(path) != old: raise ValueError("Admitted recovery input changed: "+path)
    for path, digest in p["source_sha256"].items():
        if sha(root/"producer"/Path(path).relative_to(ROOT)) != digest or live and sha(path) != digest:
            raise ValueError("Recovery producer changed: "+path)
    dp, records = data_source.verify(p["data_root"], live=False)
    if (dp["fingerprint"] != p["data_fingerprint"] or p["training"] != dp["training"]
            or p["checkpoints"] != dp["recovery_checkpoints"]
            or p["local_alignment_indices"] != dp["local_alignment_indices"]):
        raise ValueError("Frozen recovery data/order differs")
    return p, dp, records


def _first_output(stack, rotary, batch, device):
    """Same reduced entrance computation, reconstructed for its auxiliary loss."""
    block = stack.blocks[0]
    if stack.packed or stack.stripped: raise ValueError("Local training requires unpacked geometry")
    hidden = batch["boundary_hidden"].detach().to(device=device, dtype=torch.bfloat16)
    valid = batch["attention_mask"].to(device).bool()
    positions = torch.arange(hidden.shape[1], device=device)[None].expand(hidden.shape[:2])
    allowed = torch.ones(hidden.shape[1], hidden.shape[1], dtype=torch.bool, device=device).tril()[None,None]
    allowed = allowed & valid[:,None,None,:]
    mask = torch.zeros(allowed.shape, device=device, dtype=hidden.dtype)
    mask.masked_fill_(~allowed, torch.finfo(hidden.dtype).min)
    locations = block._positions(hidden, positions[0], positions, None)
    z = stack._geometry(hidden, block.basis.T)
    z, _ = forward_reduced(block, z, locations, attention_mask=mask,
        cache_position=positions[0], position_embeddings=rotary(hidden, positions))
    anchor = block._rows(block.anchor, locations, *hidden.shape[:2])
    return stack._geometry(z, block.basis, anchor)


class LocalTrainer:
    """One pass, new-block-only, response-position normalized residual MSE."""
    def __init__(self, stack, norm, embedding, rotary, dataset, indices, *,
                 config=LocalConfig(), guard=None):
        self.stack, self.norm, self.embedding, self.rotary = stack, norm, embedding, rotary
        self.dataset, self.indices, self.config, self.guard = dataset, list(indices), config, guard
        self.device = embedding.device
        if self.device.type == "cuda" and guard is None: raise ValueError("Reservation guard required")
        if (not indices or len(set(indices)) != len(indices)
                or any(i not in dataset.training for i in indices)):
            raise ValueError("Local normalization and optimization require distinct TRAIN examples")
        stack.eval().requires_grad_(False); norm.eval().requires_grad_(False)
        embedding.requires_grad_(False); rotary.eval().requires_grad_(False)
        self.block = stack.blocks[0]; self.block.requires_grad_(True)
        self.parameters = list(self.block.parameters())
        if any(p.dtype != torch.bfloat16 for p in self.parameters):
            raise ValueError("Local BF16 weights must be established before optimizer registration")
        self.fixed = self.fixed_signature()
        self.parameter_ids = tuple(map(id, stack.parameters()))
        self.optimizer = offload.CPUGradientAdamW(self.parameters, lr=config.learning_rate, weight_decay=0.,
            chunk_bytes=config.optimizer_chunk_bytes, max_cpu_state_bytes=config.max_cpu_state_bytes)
        self.visited, self.updates = set(), 0
        self.before = offload.fingerprint(self.block.named_parameters())
        self.normalization = self.fit_normalization()

    def _guard(self):
        if self.guard is not None: self.guard()

    def fixed_signature(self):
        values = [("buffer."+n,v) for n,v in self.stack.named_buffers()]
        values += [(f"parent.{i}."+n,v) for i,b in enumerate(self.stack.blocks[1:]) for n,v in b.named_parameters()]
        values += [("norm."+n,v) for n,v in self.norm.named_parameters()]
        values += [("norm_buffer."+n,v) for n,v in self.norm.named_buffers()]
        values += [("rotary."+n,v) for n,v in self.rotary.named_buffers()]
        values.append(("embedding", self.embedding))
        return offload.fingerprint(values)

    @torch.no_grad()
    def fit_normalization(self):
        energy, tokens = 0., 0
        for index in self.indices:
            self._guard(); batch = self.dataset.batch([index])
            target = batch["next_boundary_hidden"][batch["loss_mask"].bool()].float()
            energy += float(target.square().mean(-1).sum()); tokens += len(target)
        energy /= tokens
        if not math.isfinite(energy) or energy <= 0: raise ValueError("Degenerate native TRAIN target energy")
        return dict(energy=energy, response_tokens=tokens, examples=len(self.indices),
                    formula="sum_{TRAIN response positions} mean_d(native_next[d]^2) / response_position_count",
                    dtype="float32", frozen=True)

    def train_window(self, indices):
        indices = list(indices)
        if (not indices or len(indices) > self.config.accumulation_examples or len(set(indices)) != len(indices)
                or any(i not in self.indices or i in self.visited for i in indices)):
            raise ValueError("One-pass selected TRAIN windows required")
        if tuple(map(id, self.stack.parameters())) != self.parameter_ids:
            raise RuntimeError("Live parameter objects changed")
        count = sum(len(self.dataset.records[i].response_ids) for i in indices)
        self._guard(); self.block.train(); self.optimizer.zero_grad(); total = 0.; started=time.monotonic()
        try:
            for index in indices:
                self._guard(); batch = self.dataset.batch([index])
                output = _first_output(self.stack, self.rotary, batch, self.device)
                mask = batch["loss_mask"].to(self.device).bool()
                target = batch["next_boundary_hidden"].to(output).detach()
                loss = (output.float()-target.float()).square().mean(-1)[mask].sum()
                loss = loss/(count*self.normalization["energy"])
                if not torch.isfinite(loss): raise FloatingPointError("Nonfinite local loss")
                loss.backward(); total += float(loss.detach())
            if any(self.optimizer.received.get(p, 0) != len(indices) for p in self.parameters):
                raise RuntimeError("Every new-block parameter must receive each local microbatch gradient")
            if any(p.grad is not None for b in self.stack.blocks[1:] for p in b.parameters()):
                raise RuntimeError("Inherited parent received local gradients")
            receipt = self.optimizer.step(max_grad_norm=self.config.gradient_clip)
            self.updates += 1; self.visited.update(indices)
            return dict(update=self.updates, indices=indices, response_tokens=count,
                        normalized_feature_mse=total, optimizer=receipt, only_new_block_gradients=True,
                        train_window_seconds=time.monotonic()-started)
        finally: self.optimizer.zero_grad()

    def finish(self):
        if set(self.indices) != self.visited or self.fixed_signature() != self.fixed:
            raise RuntimeError("Local pass membership or frozen parent/geometry changed")
        changed = offload.fingerprint(self.block.named_parameters()) != self.before
        if not changed: raise RuntimeError("Local fitting did not change actual BF16 new-block weights")
        return dict(status="completed", updates=self.updates, normalization=self.normalization,
                    fixed_state_unchanged=True, actual_bf16_new_block_changed=changed,
                    examples=len(self.visited), configuration=asdict(self.config), task_quality_claim=False)

    def close(self): self.optimizer.close()


def build_student(p, *, device, guard):
    parent = None if p["parent"] is None else p["parent"]["checkpoint"]
    stack, norm, embedding, old_rotary, inheritance = builder.build_stack(p["model"]["path"],
        {int(k):v for k,v in p["fits"].items()}, p["cut"], previous=parent, device="cpu", trainable=True)
    del old_rotary
    # Set storage once, before either optimizer registers hooks.
    for parameter in (*stack.parameters(), *norm.parameters()): parameter.data = parameter.data.bfloat16()
    config = builder.native._config(p["model"]["path"])
    rotary = model_source.partial.training_rotary(config, device, guard=guard)
    return stack.to(device), norm.to(device), embedding.to(device), rotary, inheritance


def _identities(p):
    return dict(study=p["fingerprint"], teacher=p["model"]["revision"], data=p["data_fingerprint"],
                corpus=p["corpus_fingerprint"], fit_receipt=sha(Path(p["fit_root"])/"fit_receipt.json"))


def save_raw(path, stack, norm, p):
    with Path(path).open("xb") as stream:
        torch.save(dict(schema="jspace.rebuild-prefit-checkpoint.v1", identities=_identities(p),
            stack=offload.recursive_cpu(stack.state_dict()), norm=offload.recursive_cpu(norm.state_dict()),
            updates=0, supervised_tokens=0, local_alignment_applied=False), stream)
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


@torch.inference_mode()
def packed_mc(model, dataset):
    if not model.stack.packed or not model.stack.stripped or dataset.training:
        raise ValueError("Packed model and exposed DEV-only MC dataset required")
    rows = []
    for index in dataset.development:
        hidden = model.hidden_forward(dataset.batch([index]))[:, -1]
        rows.append(model_source.joint.cache.sanity._record(dataset.items[index], hidden,
                                                           model.embedding, dataset.label_ids))
    correct = sum(r["answer"] == r["prediction"] for r in rows)
    families = {f:dict(count=sum(r["family"] == f for r in rows),
        correct=sum(r["family"] == f and r["answer"] == r["prediction"] for r in rows))
        for f in sorted({r["family"] for r in rows})}
    return dict(records=rows, metrics=dict(count=len(rows), correct=correct, accuracy=correct/len(rows), by_family=families),
        native_correct=dataset.native_correct, native_count=len(rows), packed=True,
        eligibility=model_source.joint.broad.eligibility(rows, dataset.native_records, dataset.items))


def cheap_gate(response, native_response, mc, *, ratio=1.10):
    nll = response["nll"]
    checks = dict(response_nll=math.isfinite(nll) and nll <= ratio*native_response["nll"],
                  multiple_choice=mc["eligibility"]["passed"])
    return dict(passed=all(checks.values()), checks=checks, nll=nll, native_nll=native_response["nll"],
                maximum_native_nll_ratio=ratio)


@torch.inference_mode()
def evaluate_packed(stack, norm, embedding, rotary, dataset, mc_dataset, p, dp, directory,
                    checkpoint, *, update, response_tokens, label, guard):
    """Only the freshly reloaded packed artifact supplies quality measurements."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    device = embedding.device; guard(); started = time.monotonic()
    config = builder.native._config(p["model"]["path"])
    packed = packed_source.PackedResponseModel(builder.packed_clone(stack, device=device),
        cpu_module_clone(norm).to(device), embedding.detach(), cpu_module_clone(rotary).to(device),
        config=config, guard=guard)
    exported = packed_source.export_packed_snapshot(packed, directory/"export",
        identities=dict(_identities(p), checkpoint_sha256=checkpoint["sha256"], label=label))
    # Replay actual cached prefixes against the same exported model, not against
    # unpacked arithmetic. No task loss is inspected before the fresh reload.
    probes = []
    for index in dataset.development[:3]:
        batch = dataset.batch([index])
        probes.append(packed.readout(packed.hidden_forward(batch)[:, -1]).cpu())
    del packed; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    model, reloaded = packed_source.load_packed_snapshot(directory/"export", device=device, guard=guard)
    for index, reference in zip(dataset.development[:3], probes):
        actual = model.readout(model.hidden_forward(dataset.batch([index]))[:, -1]).cpu()
        if not torch.equal(actual, reference): raise ValueError("Fresh packed reload changed full-vocabulary inference")
    if reloaded["packed_state_sha256"] != exported["packed_state_sha256"]:
        raise ValueError("Packed state identity differs after fresh reload")
    write(directory/"export_receipt.json", exported); write(directory/"reload_receipt.json", reloaded)
    write(directory/"reload_parity.json", dict(passed=True, full_vocabulary_exact=True, prefixes=len(probes),
        scope="Same packed exported state versus independent reload; no packed/unpacked equality gate"))
    by_id = {r.example_id:i for i,r in enumerate(dataset.records)}
    teacher = training.CachedHiddenTeacher(model.embedding, teacher_id=p["model"]["revision"], guard=guard)
    response = evaluation.response_losses(model, (dataset.records[i] for i in dataset.development),
        pad_token_id=dataset.pad_token_id, max_length=416, teacher_provider=teacher, token_chunk_size=8,
        prepare_batch=lambda b:dataset.batch([by_id[name] for name in b["example_ids"]]))
    write(directory/"development.json", response)
    mc = packed_mc(model, mc_dataset); write(directory/"multiple_choice.json", mc)
    native_response = read(Path(dp["response_cache_root"])/"native/development.json")
    cheap = cheap_gate(response, native_response, mc, ratio=p["gates"]["maximum_native_nll_ratio"])
    candidate = dict(label=label, update=update, response_tokens=response_tokens, checkpoint=checkpoint,
        directory=str(directory.resolve()), nll=response["nll"], teacher_kl=response["kl"],
        cheap_gate=cheap, mc_accuracy=mc["metrics"]["accuracy"], packed_artifact_path=str(directory/"export"),
        packed_export=exported, packed_reload=reloaded, packed_reload_passed=True,
        semantic_task_success=None, eligibility=dict(passed=False), confirmed=False)
    if cheap["passed"]:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(p["model"]["path"], local_files_only=True, trust_remote_code=False)
        texts = {row["example_id"]:row for row in map(json.loads, (Path(dp["corpus"])/"text.jsonl").read_text().splitlines())}
        generation = []
        with packed_source.attach_prefix(model, p["model"]["path"], mode=p["prefix_mode"]) as prefix:
            begin = time.monotonic()
            canaries = evaluation.evaluate_canaries(model, tokenizer, device=device,
                eos_token_ids=p["generation_eos_token_ids"], max_new_tokens=p["max_new_tokens"])
            for index in p["generation_indices"]:
                record = dataset.records[index]
                generation.append(evaluation.generate_record(model, record, tokenizer, device=device,
                    source_text=texts[record.example_id]["context"], eos_token_ids=p["generation_eos_token_ids"],
                    max_new_tokens=p["max_new_tokens"]))
                _event(directory, "generation", completed=len(generation), total=len(p["generation_indices"]))
            duration = time.monotonic()-begin
            timing = packed_evaluation.warm_timing(model, dataset.records[p["generation_indices"][0]].prompt_ids,
                                                  warmup_calls=1, repeats=3)
        tokens = sum(len(r["output_ids"]) for r in [*canaries["records"], *generation])
        write(directory/"generation.json", dict(canaries=canaries, corpus=generation, use_cache=False,
            native_prefix=prefix, seconds=duration, output_tokens_including_eos=tokens,
            cachefree_generation_tokens_per_second=tokens/duration,
            timing_scope="Packed cache-free full-prefix recomputation, not KV-cached deployment speed"))
        write(directory/"warm_timing.json", timing)
        candidate["grading_status"] = "pending"
    else:
        candidate["grading_status"] = "not_run_failed_cheap_retention"
        candidate["eligibility"] = dict(passed=False, checks=cheap["checks"],
            missing_semantic_evidence=True, semantic_failure_inferred=False)
    candidate["gpu_evaluation_seconds"] = time.monotonic()-started
    if device.type == "cuda": candidate["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    write(directory/"candidate.json", candidate)
    del model, teacher, probes; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return candidate


def finish_quality(candidate, p, dp):
    """CPU/network-only grading of immutable exported checkpoint responses."""
    directory = Path(candidate["directory"])
    if not candidate["cheap_gate"]["passed"]:
        write(directory/"result.json", candidate); return candidate
    actual = read(directory/"generation.json")
    native = read(Path(dp["response_cache_root"])/"native/generation.json")
    texts = {row["example_id"]:row for row in map(json.loads, (Path(dp["corpus"])/"text.jsonl").read_text().splitlines())}
    canary = evaluation.generation_gate(native["canaries"], actual["canaries"],
        maximum_drop_pp=p["gates"]["maximum_drop_pp"], minimum_student_rate=0.)
    executable = task_metrics.grade_generated_tasks(native["corpus"], actual["corpus"], texts,
        maximum_drop_pp=p["gates"]["maximum_drop_pp"], minimum_student_rate=0.)
    write(directory/"generative_task_checks.json", executable)
    contexts = {r["example_id"]:dict(prompt=texts[r["example_id"]]["prompt_text"],
        source=texts[r["example_id"]]["context"], reference=texts[r["example_id"]]["response"],
        task=texts[r["example_id"]]["task"]) for r in actual["corpus"]}
    semantic = judge.judge_pairs(native["corpus"], actual["corpus"], directory/"semantic_judge",
                                  config=judge.JudgeConfig(**p["judge"]), contexts=contexts)
    # The shared judge also emits its historical absolute-floor diagnostic.
    # Recompute the NEW prospective relative-retention gate from raw verdicts.
    gate = evaluation.semantic_gate(native["corpus"], actual["corpus"], semantic["judgments"],
        judge_provenance=semantic["judge_provenance"], maximum_drop_pp=p["gates"]["maximum_drop_pp"],
        minimum_student_rate=0.)
    response = read(directory/"development.json"); mc = read(directory/"multiple_choice.json")
    native_response = read(Path(dp["response_cache_root"])/"native/development.json")
    quality = packed_evaluation.combine_quality(response, native_response, canary, gate, mc, executable, p["gates"])
    quality["scope"] = "New exploratory native-relative DEV retention; no absolute application quality certification"
    result = dict(candidate, grading_status=semantic["status"], canary_gate=canary,
        semantic_gate=gate, executable_gates=executable["gates"], eligibility=quality,
        semantic_task_success=gate.get("rates",{}).get("task_success",{}).get("student"))
    write(directory/"result.json", result)
    return result


def select(candidates):
    if not candidates or not any(c["label"] == "prefit" for c in candidates):
        raise ValueError("Retained raw compression baseline and recovery candidates required")
    def key(c): return (-c["semantic_task_success"], c["nll"], c["response_tokens"], c["update"])
    eligible = [c for c in candidates if c["eligibility"]["passed"]]
    if any(c.get("grading_status") != "completed" or c.get("semantic_task_success") is None for c in eligible):
        raise ValueError("Missing generation evidence cannot qualify a checkpoint")
    diagnostic = min(candidates, key=lambda c:(c["nll"], c["response_tokens"], c["update"]))
    return dict(selected=min(eligible,key=key) if eligible else None, diagnostic_best=diagnostic,
                relative_retention_pass=bool(eligible), confirmed=False)


def _ledger(root, p, result):
    chosen = result["selected"] or result["diagnostic_best"]
    experiment = nal.Experiment(id=f"rebuild.cut{p['cut']}.{p['fingerprint'][:12]}",
        hypothesis_id="JSPACE-QWEN3B-PROGRESSIVE-RESPONSE-REBUILD-01",
        name="Fresh broad-response progressive suffix recovery", parameters=dict(fingerprint=p["fingerprint"], cut=p["cut"]),
        seed=p["seed"])
    record = nal.ExperimentResult(experiment_id=experiment.id, hypothesis_id=experiment.hypothesis_id,
        metrics=dict(relative_retention_pass=float(result["relative_retention_pass"]), nll=chosen["nll"],
                     mc_accuracy=chosen["mc_accuracy"]), primary_metric=chosen["mc_accuracy"],
        model_architecture=[p["cut"],36-p["cut"],1024], model_parameters=result['compact_trainable_parameters'],
        training_time=result["training_seconds"], model_checkpoint=chosen["checkpoint"]["path"],
        observations=["Exploratory DEV selection, not independently confirmed", "Reported quality uses actual packed export reload",
            "Parameter count covers trainable compact suffix and norm; full native-prefix storage is recorded separately"],
        status=nal.ExperimentStatus.COMPLETED)
    return str(nal.ExperimentResultLedger(root).save(experiment,record,attempts=1,device_id=0))


def initializer_fingerprint(stack, norm, embedding, rotary):
    return dict(learned=offload.fingerprint([*(('stack.'+n,v) for n,v in stack.named_parameters()),
                *(('norm.'+n,v) for n,v in norm.named_parameters())]),
                fixed=offload.fixed_state(stack,embedding,rotary))


def systems_pilot(stack,norm,embedding,rotary,dataset,*,local_indices,joint_indices,
                  local_config,response_config,teacher_id,model_config,directory,guard=None,
                  expected_layers=36,host_reserve_bytes=8*1024**3):
    """Disposable real-gradient and packed-reload smoke; no quality selection."""
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    initial=initializer_fingerprint(stack,norm,embedding,rotary)
    local=trainer=None
    try:
        local=LocalTrainer(stack,norm,embedding,rotary,dataset,local_indices,
                           config=local_config,guard=guard)
        fixed=local.fixed_signature()
        local_update=local.train_window(local_indices[:local_config.accumulation_examples])
        local_fixed=local.fixed_signature()==fixed
        local_changed=offload.fingerprint(local.block.named_parameters())!=local.before
        if not local_fixed or not local_changed or not any(local_update['optimizer']['master_changed_by_parameter']):
            raise RuntimeError("Discarded local pilot failed actual new-block update or fixed-parent proof")
        local.close();local=None
        trainer=training.ResponseTrainer(stack,norm,embedding,rotary,dataset,
            teacher_id=teacher_id,config=response_config,guard=guard,expected_layers=expected_layers,
            host_reserve_bytes=host_reserve_bytes)
        before=trainer.parameter_fingerprints()
        joint_update=trainer.train_window(joint_indices)
        changed=trainer.change_report(before)
        if not all(changed['bf16_changed_by_block']):
            raise RuntimeError("Discarded joint pilot did not update every actual BF16 compact block")
        checkpoint=trainer.save_checkpoint(directory/'discarded.pt',identities={'teacher':teacher_id,'discarded':True})
        saved=torch.load(checkpoint['path'],map_location='cpu',weights_only=True,mmap=True)
        # Structural trainable reload: same keys/geometry, independent objects.
        copied=cpu_module_clone(stack); copied_norm=cpu_module_clone(norm)
        copied.load_state_dict(saved['stack'],strict=True);copied_norm.load_state_dict(saved['norm'],strict=True)
        if (not builder.state_equal(copied.state_dict(),saved['stack'])
                or not builder.state_equal(copied_norm.state_dict(),saved['norm'])):
            raise RuntimeError("Discarded checkpoint strict reconstruction differs")
        del saved,copied,copied_norm
        trainer.close();trainer=None
        packed=packed_source.PackedResponseModel(builder.packed_clone(stack),cpu_module_clone(norm).to(embedding.device),
            embedding.detach(),cpu_module_clone(rotary).to(embedding.device),config=model_config,
            max_length=response_config.max_length,guard=guard)
        exported=packed_source.export_packed_snapshot(packed,directory/'export',identities={'discarded':True})
        batch=dataset.batch([joint_indices[0]])
        with torch.inference_mode():
            logits=packed.readout(packed.hidden_forward(batch)[:,-1]).cpu()
        del packed;gc.collect()
        loaded,reloaded=packed_source.load_packed_snapshot(directory/'export',device=embedding.device,guard=guard)
        with torch.inference_mode():
            replay=loaded.readout(loaded.hidden_forward(batch)[:,-1]).cpu()
        if not torch.equal(logits,replay) or exported['packed_state_sha256']!=reloaded['packed_state_sha256']:
            raise RuntimeError("Discarded packed export/reload differs")
        result=dict(status='completed',passed=True,discarded=True,systems_only=True,
            initializer=initial,local_update=local_update,local_fixed_parent=True,local_actual_bf16_changed=True,
            joint_update=joint_update,joint_change=changed,checkpoint=checkpoint,
            strict_trainable_reload=True,packed_export=exported,packed_reload=reloaded,
            full_vocabulary_packed_replay_exact=True,vocabulary=logits.shape[-1],
            scientific_restart='Fresh native-initialized stack and fresh optimizer required')
        write(directory/'result.json',result)
        return result
    finally:
        if local is not None:local.close()
        if trainer is not None:trainer.close()


def pilot(root,*,limit_mib=11008):
    root=Path(root).resolve();p,_,records=verify(root)
    directory=root/'pilot'
    if directory.exists():raise FileExistsError(directory)
    if limit_mib!=p['limit_mib']:raise ValueError('Allocator differs from the frozen recovery plan')
    dataset=None;lease=None;started=time.monotonic()
    try:
        lease,guard=cache.reservation(limit_mib)
        if next(iter(lease['allocation'].values()))!=p['allocation_mib']:
            raise ValueError('Pilot requires its declared reservation allocation')
        torch.set_num_threads(2);torch.manual_seed(p['seed'])
        dataset=data_source.RebuildDataset(p['data_root'],cut=p['cut'])
        with execution.arithmetic_policy():
            stack,norm,E,rotary,_=build_student(p,device='cuda:0',guard=guard)
            # Joint pilot covers the longest input/response and every family;
            # it is deliberately discarded before the scientific order begins.
            from .jspace_3b_response_campaign import pilot_windows
            joint_indices=pilot_windows(records,p['training']['order'],updates=1)[0]
            config=training.ResponseConfig(**{**p['configuration'],
                'max_supervised_tokens':sum(len(records[i].response_ids) for i in joint_indices)})
            result=systems_pilot(stack,norm,E,rotary,dataset,
                local_indices=p['local_alignment_indices'],joint_indices=joint_indices,
                local_config=LocalConfig(**p['local_configuration']),response_config=config,
                teacher_id=p['model']['revision'],model_config=builder.native._config(p['model']['path']),
                directory=directory/'systems',guard=guard)
        receipt=dict(status='completed',passed=result['passed'],discarded=True,fingerprint=p['fingerprint'],
            initializer=result['initializer'],systems_result_sha256=sha(directory/'systems/result.json'),
            lease_id=lease['id'],allocation=lease['allocation'],allocator_mib=limit_mib,
            elapsed_seconds=time.monotonic()-started)
        write(directory/'result.json',receipt);return receipt
    except BaseException as error:
        directory.mkdir(parents=True,exist_ok=True)
        write(directory/'failure.json',dict(status='failed',fingerprint=p['fingerprint'],
            error_type=type(error).__name__,error=str(error),lease=lease))
        raise
    finally:
        if dataset is not None:dataset.close()


def pilot_receipt(root,p):
    directory=Path(root)/'pilot';result=read(directory/'result.json')
    if (result.get('status')!='completed' or result.get('passed') is not True
            or result.get('discarded') is not True or result.get('fingerprint')!=p['fingerprint']
            or sha(directory/'systems/result.json')!=result.get('systems_result_sha256')):
        raise ValueError('Matching completed discarded systems pilot required before scientific training')
    return result


def run_stage(root, *, limit_mib=11008):
    root = Path(root).resolve(); p, dp, records = verify(root)
    if limit_mib!=p['limit_mib']:raise ValueError('Allocator differs from the frozen recovery plan')
    pilot_result=pilot_receipt(root,p)
    directory = root/"run"
    if directory.exists(): raise FileExistsError(directory)
    directory.mkdir(); (directory/"checkpoints").mkdir()
    started = time.monotonic(); phase = "reservation"; lease = None; local = trainer = dataset = mc_dataset = None
    try:
        lease, guard = cache.reservation(limit_mib); torch.set_num_threads(2); torch.manual_seed(p["seed"])
        if next(iter(lease['allocation'].values()))!=p['allocation_mib']:
            raise ValueError('Scientific worker requires its declared reservation allocation')
        _event(directory, phase, cut=p["cut"], lease_id=lease["id"])
        dataset = data_source.RebuildDataset(p["data_root"], cut=p["cut"])
        mc_dataset = data_source.MCDataset(p["data_root"], cut=p["cut"])
        with execution.arithmetic_policy(), ThreadPoolExecutor(max_workers=1) as pool:
            phase = "native_initialized_stack"
            stack,norm,embedding,rotary,inheritance = build_student(p,device="cuda:0",guard=guard)
            if initializer_fingerprint(stack,norm,embedding,rotary)!=pilot_result['initializer']:
                raise RuntimeError('Scientific initializer differs from the pre-update pilot initializer')
            storage=builder.inventory(p['model']['path'],stack,norm,rotary)
            write(directory/'storage.json',storage)
            trainable_count=sum(v.numel() for v in (*stack.parameters(),*norm.parameters()))
            training_seconds=0.
            write(directory/"inheritance.json", inheritance)
            raw = save_raw(directory/"checkpoints/prefit.pt",stack,norm,p)
            jobs = []
            def evaluate(checkpoint, update, response_tokens, label):
                _event(directory,"packed_evaluation",label=label,update=update,response_tokens=response_tokens)
                row = evaluate_packed(stack,norm,embedding,rotary,dataset,mc_dataset,p,dp,
                    directory/"evaluations"/label,checkpoint,update=update,response_tokens=response_tokens,
                    label=label,guard=guard)
                jobs.append(pool.submit(finish_quality,row,p,dp))
            phase = "raw_packed_baseline"; evaluate(raw,-1,0,"prefit")
            phase = "local_prealignment"
            local = LocalTrainer(stack,norm,embedding,rotary,dataset,p["local_alignment_indices"],
                                  config=LocalConfig(**p["local_configuration"]),guard=guard)
            for indices in p["local_alignment_windows"]:
                progress=local.train_window(indices);training_seconds+=progress['train_window_seconds']
                _event(directory,phase,**progress)
            local_result = local.finish(); write(directory/"local_alignment.json",local_result)
            local.close(); local=None
            phase = "joint_recovery"
            trainer = training.ResponseTrainer(stack,norm,embedding,rotary,dataset,
                teacher_id=p["model"]["revision"],config=training.ResponseConfig(**p["configuration"]),guard=guard)
            before = trainer.parameter_fingerprints()
            checkpoint = trainer.save_checkpoint(directory/"checkpoints/update_0000.pt",identities=_identities(p))
            evaluate(checkpoint,0,0,"update_0000")
            milestones = {row["update"] for row in p["checkpoints"]}
            for indices in p["training"]["windows"]:
                phase = "joint_recovery"; progress = trainer.train_window(indices)
                training_seconds+=progress['train_window_seconds']
                _event(directory,phase,**progress)
                if trainer.updates in milestones:
                    phase = "checkpoint_save"
                    checkpoint = trainer.save_checkpoint(directory/f"checkpoints/update_{trainer.updates:04d}.pt",
                        identities=_identities(p))
                    evaluate(checkpoint,trainer.updates,trainer.supervised_tokens,f"update_{trainer.updates:04d}")
            change = trainer.change_report(before)
            if not all(change["bf16_changed_by_block"]):
                raise RuntimeError("Joint recovery failed to change every compact block's actual BF16 weights")
            if (trainer.supervised_tokens != p["training"]["actual_supervised_tokens"]
                    or trainer.visited != set(p["training"]["order"])):
                raise RuntimeError("One-pass response training budget or membership differs")
            trainer.close(); trainer=None
            phase = "semantic_completion"; _event(directory,phase,candidates=len(jobs))
            candidates = [job.result() for job in jobs]
            chosen = select(candidates)
            selected_checkpoint = None if chosen["selected"] is None else chosen["selected"]["checkpoint"]
            result = dict(status="completed",
                outcome="relative_retention_pass" if chosen["relative_retention_pass"] else "retention_not_established",
                fingerprint=p["fingerprint"],cut=p["cut"],**chosen,candidates=candidates,
                selected_checkpoint=None if selected_checkpoint is None else selected_checkpoint["path"],
                selected_checkpoint_sha256=None if selected_checkpoint is None else selected_checkpoint["sha256"],
                local_alignment=local_result,joint_change=change,
                response_tokens=p["training"]["actual_supervised_tokens"],
                initializer=raw,inheritance=inheritance,lease_id=lease["id"],allocation=lease["allocation"],
                discarded_pilot_sha256=sha(root/'pilot/result.json'),fresh_scientific_initializer_exact=True,
                allocator_mib=limit_mib,elapsed_seconds=time.monotonic()-started,
                training_seconds=training_seconds,compact_trainable_parameters=trainable_count,storage=storage,
                no_additional_epochs=True,application_quality_certified=False,promoted=False)
            result["nal_ledger"] = _ledger(root,p,result)
            write(directory/"result.json",result)
            write(directory/"manifest.json",dict(status="completed",fingerprint=p["fingerprint"],
                artifacts={str(path.relative_to(directory)):sha(path) for path in sorted(directory.rglob("*.json"))},
                checkpoints={c["checkpoint"]["path"]:c["checkpoint"]["sha256"] for c in candidates},
                packed_exports={c["label"]:c["packed_export"]["manifest_sha256"] for c in candidates}))
            return result
    except BaseException as error:
        write(directory/"failure.json",dict(status="failed",fingerprint=p["fingerprint"],phase=phase,
            error_type=type(error).__name__,error=str(error),lease=lease,elapsed_seconds=time.monotonic()-started,
            retained_checkpoints=True,automatic_retry=False))
        raise
    finally:
        if local is not None: local.close()
        if trainer is not None: trainer.close()
        if dataset is not None: dataset.close()
        if mc_dataset is not None: mc_dataset.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("phase",choices=("prepare","pilot","run","check"))
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--fit-root",type=Path)
    parser.add_argument("--cut",type=int,choices=(35,34))
    parser.add_argument("--parent-checkpoint",type=Path)
    parser.add_argument("--limit-mib",type=int,default=11008)
    args=parser.parse_args()
    if args.phase=="prepare":
        if any(v is None for v in (args.data_root,args.fit_root,args.cut)):
            parser.error("prepare requires --data-root, --fit-root and --cut")
        result=prepare(args.root,args.data_root,args.fit_root,args.cut,args.parent_checkpoint)
    elif args.phase=="pilot": result=pilot(args.root,limit_mib=args.limit_mib)
    elif args.phase=="run": result=run_stage(args.root,limit_mib=args.limit_mib)
    else:
        p,_,_=verify(args.root); result=dict(status="prepared",fingerprint=p["fingerprint"],cut=p["cut"])
    print(json.dumps({k:result[k] for k in ("status","fingerprint","cut","relative_retention_pass") if k in result}),flush=True)


if __name__=="__main__": main()
