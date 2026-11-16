"""Three fixed-prefix, full-vocabulary checks of the recorded partial generator.

This diagnostic neither generates a new continuation nor scores a benchmark.
Exact logits are the primary comparison; the predeclared BF16 tolerance is
reported separately and never relabels a nonexact result as exact.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

from . import jspace_3b_broad_assess as assessment
from .jspace_full_lease import wait_ready

base, old, read = assessment.base, assessment.old, assessment.read
generation = assessment.generation
SCHEMA = "jspace.3b-generation-path-probe.v1"
COUNTS = (0, 1, 8)
ATOL, RTOL = .03125, 1 / 128
TEST = base.ROOT / "tests/structure_net/test_jspace_3b_generation_path_probe.py"


def prefixes(canary, record):
    if (canary.get("id") != "concept_heat_transfer" or record.get("id") != canary["id"]
            or record.get("input_ids") != canary["input_ids"] or len(record.get("output_ids", [])) < 8
            or not canary["input_ids"] or len(canary["input_ids"]) + 64 > 416
            or any(type(x) is not int or x < 0 for x in canary["input_ids"] + record["output_ids"])):
        raise ValueError("Original first canary and at least eight archived continuation tokens required")
    return [dict(continuation_tokens=n, input_ids=canary["input_ids"] + record["output_ids"][:n]) for n in COUNTS]


def _context(model):
    device = model.lm_head.weight.device
    return torch.cuda.device(device) if device.type == "cuda" else nullcontext()


@torch.inference_mode()
def wrapper_logits(model, token_ids, guard=lambda: None):
    guard()
    with _context(model):
        ids = torch.tensor([token_ids], dtype=torch.long, device=model.lm_head.weight.device)
        # Reproduce cache-free generate() input preparation on this fixed prefix.
        prepared = model.prepare_inputs_for_generation(ids, attention_mask=torch.ones_like(ids),
            cache_position=torch.arange(ids.shape[1], device=ids.device), use_cache=False, logits_to_keep=1)
        return model(**prepared).logits[:, -1].detach()


@torch.inference_mode()
def direct_logits(model, token_ids, guard=lambda: None):
    """Bypass HF model/suffix forwards: explicit native layers then raw stack."""
    guard()
    with _context(model):
        cut = model.partial_generation_scope["cut"]
        suffix = model.model.layers[cut]
        if (not isinstance(suffix, generation.TrainingRotarySuffix)
                or model.lm_head.weight is not model.model.embed_tokens.weight
                or model.training or not 0 < len(token_ids) <= 416):
            raise ValueError("Frozen tied-readout partial model and bounded singleton prefix required")
        ids = torch.tensor([token_ids], dtype=torch.long, device=model.lm_head.weight.device)
        hidden = model.model.embed_tokens(ids)
        positions = torch.arange(ids.shape[1], device=ids.device)[None]
        mask = torch.full((ids.shape[1], ids.shape[1]), torch.finfo(hidden.dtype).min,
                          device=ids.device, dtype=hidden.dtype).triu(1)[None, None]
        native_rope = model.model.rotary_emb(hidden, positions)
        for layer in model.model.layers[:cut]:
            hidden = layer(hidden, attention_mask=mask, position_ids=positions,
                           cache_position=positions[0], position_embeddings=native_rope,
                           use_cache=False)[0]
        hidden = suffix.stack(hidden, attention_mask=mask, position_ids=positions,
            cache_position=positions[0], position_embeddings=suffix.rotary(hidden, positions), use_cache=False)[0]
        # Keep native RMSNorm's whole-sequence shape; both paths project one row.
        normalized = model.model.norm(hidden)[:, -1:]
        return F.linear(normalized, model.model.embed_tokens.weight)[:, -1].detach()


def compare_logits(actual, reference):
    if (actual.shape != reference.shape or actual.ndim != 2 or actual.shape[0] != 1
            or not torch.isfinite(actual).all() or not torch.isfinite(reference).all()):
        raise ValueError("Aligned finite singleton full-vocabulary logits required")
    a, b = actual.float(), reference.float()
    delta = (a - b).abs()
    return dict(exact=bool(torch.equal(actual, reference)), max_abs=float(delta.max()),
        differing_logits=int((a != b).sum()), vocabulary_size=a.shape[1],
        wrapper_argmax=int(a.argmax(-1)), direct_argmax=int(b.argmax(-1)),
        argmax_exact=bool(torch.equal(a.argmax(-1), b.argmax(-1))),
        tolerance_pass=bool((delta <= ATOL + RTOL * b.abs()).all()), atol=ATOL, rtol=RTOL,
        relative_rms=float(delta.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-30)))


def processed_argmax(logits, token_ids, config):
    """Replay the sole active deterministic processor in the frozen config."""
    zero = ("no_repeat_ngram_size", "encoder_no_repeat_ngram_size", "min_length")
    absent = ("bad_words_ids", "force_words_ids", "forced_bos_token_id", "forced_eos_token_id",
              "suppress_tokens", "begin_suppress_tokens", "sequence_bias", "exponential_decay_length_penalty",
              "constraints", "watermarking_config", "guidance_scale")
    if (config.get("do_sample") is not False or config.get("num_beams") != 1
            or config.get("use_cache") is not False or config.get("max_new_tokens") != 64
            or any(config.get(k, 0) not in (None, 0) for k in zero)
            or any(config.get(k) is not None for k in absent)
            or config.get("min_new_tokens") not in (None, 0)
            or config.get("renormalize_logits", False) or config.get("remove_invalid_values", False)
            or config.get("encoder_repetition_penalty", 1.) != 1.):
        raise ValueError("Unsupported additional generation processor in archived configuration")
    scores = logits.float().clone()
    ids = torch.tensor([token_ids], device=scores.device, dtype=torch.long)
    scores = RepetitionPenaltyLogitsProcessor(float(config.get("repetition_penalty", 1.)))(ids, scores)
    return int(scores.argmax(-1))


def prepare(root, assessment_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU preparation requires CUDA hidden")
    root, source = Path(root).resolve(), Path(assessment_root).resolve()
    if root.exists(): raise FileExistsError(root)
    p = assessment.verify(source)
    selected = assessment.checked(source, "generate_selected", p)
    manifest, result = read(source / "manifest.json"), read(source / "result.json")
    if (manifest.get("status") != "completed" or manifest.get("fingerprint") != p["fingerprint"]
            or manifest.get("artifacts", {}).get("result.json") != base.sha(source / "result.json")
            or result.get("status") != "completed" or result.get("fingerprint") != p["fingerprint"]
            or result["phase_manifest_sha256"]["generate_selected"] != base.sha(source / "generate_selected/manifest.json")):
        raise ValueError("Completed finalized assessment required")
    canary = read(source / "canaries.json")["canaries"][0]
    record = selected["records"][0]
    fixed_prefixes = prefixes(canary, record)
    if base.sha(p["checkpoint"]) != p["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    inputs = dict(p["input_sha256"])
    for name in ("plan.json", "manifest.json", "result.json", "canaries.json",
                 "generate_selected/manifest.json", "generate_selected/result.json"):
        old.pin(inputs, source / name)
    files = assessment.cache.source.inherited.local_import_closure([Path(__file__), TEST])
    plan = dict(schema=SCHEMA, assessment_root=str(source), assessment_fingerprint=p["fingerprint"],
        checkpoint=p["checkpoint"], checkpoint_sha256=p["checkpoint_sha256"], model=p["model"], fits=p["fits"],
        cut=22, expected_fixed_state=p["compact_development_controls"]["selected"]["fixed_state"],
        canary=canary, recorded=record, prefixes=fixed_prefixes, vocabulary_size=151936,
        allocation_mib=11264, allocator_mib=11008, atol=ATOL, rtol=RTOL,
        criterion="Exact full-vocabulary logits and argmax; tolerance is separate numerical diagnostic",
        input_sha256=inputs, input_identity={path: old.file_identity(path) for path in inputs},
        source_sha256={str(path): base.sha(path) for path in sorted(files)},
        scope="Three teacher-forced prefixes from one already recorded canary; no new generation, training or quality claim")
    plan["fingerprint"] = base.digest(plan)
    root.mkdir(parents=True)
    for path, digest in plan["source_sha256"].items():
        target = root / "producer" / Path(path).relative_to(base.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        if base.sha(target) != digest: raise ValueError("Producer archive changed")
    base.write(root / "plan.json", plan)
    return dict(status="prepared", fingerprint=plan["fingerprint"], prefixes=[len(x["input_ids"]) for x in fixed_prefixes])


def verify(root):
    root = Path(root); p = read(root / "plan.json"); assessment.cache.source.check_plan(p)
    if (p.get("schema") != SCHEMA or p.get("cut") != 22 or p.get("vocabulary_size") != 151936
            or p.get("prefixes") != prefixes(p["canary"], p["recorded"])
            or (p.get("allocation_mib"), p.get("allocator_mib"), p.get("atol"), p.get("rtol")) != (11264, 11008, ATOL, RTOL)):
        raise ValueError("Frozen three-prefix diagnostic policy changed")
    for path, identity in p["input_identity"].items():
        if old.file_identity(path) != identity: raise ValueError("Input changed: " + path)
    for path, digest in p["source_sha256"].items():
        if base.sha(path) != digest or base.sha(root / "producer" / Path(path).relative_to(base.ROOT)) != digest:
            raise ValueError("Producer source/archive changed")
    return p


def run(root):
    root = Path(root).resolve(); p = verify(root); out = root / "run"
    if out.exists(): raise FileExistsError(out)
    lease = None; started = time.monotonic()
    try:
        lease = wait_ready([0])
        allocator = assessment.cache.source.inherited.cap_allocator(lease, p)
        torch.cuda.init(); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(); torch.set_num_threads(4)
        def guard():
            current = wait_ready([0])
            if current["id"] != lease["id"] or current["allocation"] != lease["allocation"]:
                raise RuntimeError("Reservation changed")
            if torch.cuda.max_memory_reserved() > allocator["allocator_limit_bytes"]:
                raise MemoryError("Reservation allocator cap exceeded")
        out.mkdir(); records = []; tensors = []
        with assessment.sanity.execution.arithmetic_policy():
            model = generation.build_partial(p["model"]["path"], {int(k): v for k, v in p["fits"].items()}, p["checkpoint"],
                cut=p["cut"], device="cuda:0", guard=guard, expected_fixed_state=p["expected_fixed_state"])
            for prefix in p["prefixes"]:
                ids = prefix["input_ids"]
                wrapper, direct = wrapper_logits(model, ids, guard), direct_logits(model, ids, guard)
                comparison = compare_logits(wrapper, direct)
                if comparison["vocabulary_size"] != p["vocabulary_size"]: raise ValueError("Wrong full vocabulary")
                records.append(dict(prefix, **comparison,
                    processed_wrapper_argmax=processed_argmax(wrapper, ids, p["recorded"]["generation_config"]),
                    processed_direct_argmax=processed_argmax(direct, ids, p["recorded"]["generation_config"])))
                tensors.append(dict(continuation_tokens=prefix["continuation_tokens"],
                    wrapper=wrapper.cpu(), direct=direct.cpu()))
        guard()
        result = dict(status="completed", fingerprint=p["fingerprint"], checkpoint_sha256=p["checkpoint_sha256"],
            records=records, exact_all=all(r["exact"] for r in records),
            argmax_exact_all=all(r["argmax_exact"] for r in records),
            tolerance_all=all(r["tolerance_pass"] for r in records),
            archived_first_token=p["recorded"]["output_ids"][0],
            archived_first_token_exact=records[0]["processed_wrapper_argmax"] == p["recorded"]["output_ids"][0],
            raw_first_token_matches_archived=records[0]["wrapper_argmax"] == p["recorded"]["output_ids"][0],
            fixed_state_control=model.partial_fixed_state_receipt, confirmed=False, quality_claim=False,
            scope=p["scope"], elapsed_seconds=time.monotonic()-started)
        torch.save(tensors, out / "logits.pt"); base.write(out / "result.json", result)
        base.write(out / "manifest.json", dict(status="completed", fingerprint=p["fingerprint"],
            allocation=lease["allocation"], allocator=allocator, lease_id=lease["id"],
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            artifacts={name: base.sha(out / name) for name in ("result.json", "logits.pt")}))
        return result
    except BaseException as error:
        out.mkdir(exist_ok=True)
        base.write(out / "failure.json", dict(status="execution_failure", fingerprint=p["fingerprint"],
            error=repr(error), lease_id=None if lease is None else lease["id"]))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assessment-root", type=Path,
        default=base.ROOT / "data/experiments/jspace_3b_broad_assess/20261115_v1")
    args = parser.parse_args()
    value = prepare(args.root, args.assessment_root) if args.phase == "prepare" else run(args.root)
    print(json.dumps(value, allow_nan=False), flush=True)
