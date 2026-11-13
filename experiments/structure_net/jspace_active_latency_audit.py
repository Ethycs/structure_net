"""Post-hoc matched single-label latency audit, with no new fitting or selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import shutil
import statistics
import time

import torch
import torch.nn.functional as F

from .jspace_active_benchmark import prefix_answer, truncate_model
from .jspace_active_math import move
from .jspace_active_study import DEFAULT_ROOT as PARENT, cpu_only, verify as verify_parent
from .jspace_lens import load_small_qwen
from .jspace_relation_capture import _prompt
from .jspace_relation_study import ROOT, digest, file_hash, write_json
from .jspace_relation_tasks import COLOURS


STUDY_ID = "JSPACE-ACTIVE-LATENCY-AUDIT-01"
DEFAULT_ROOT = ROOT/"data/experiments/jspace_active_latency_audit/20261113_v1"
PROTOCOL = ROOT/"docs/07 - Status Reports/jspace/2026-11-13_jspace-active-latency-audit-plan.md"
TEST = ROOT/"tests/structure_net/test_jspace_active_latency_audit.py"


def teacher_choice(model, tokenizer, prompt, candidate_weight=None):
    """One prefill, no continuation or KV cache; full-vocab ID or class index."""
    encoded = tokenizer(prompt, return_tensors="pt")
    device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        output = model.model(input_ids=encoded["input_ids"].to(device),
                             attention_mask=encoded["attention_mask"].to(device), use_cache=False)
        final = output.last_hidden_state[:, -1]
        logits = model.lm_head(final) if candidate_weight is None else F.linear(final, candidate_weight)
        return int(logits.argmax(1).item())


def prepare(root):
    cpu_only()
    parent = verify_parent(PARENT)
    if root.exists():
        raise FileExistsError(root)
    inputs = [PARENT/name for name in ("plan.json", "tasks.json", "fit.pt", "fit.json",
                                      "evaluation/natural.json", "analysis.json", "amended_result.json", "benchmark/result.json")]
    producers = [Path(__file__), PROTOCOL, TEST]
    value = {"study_id": STUDY_ID, "scope": "EXPLORATORY_POSTHOC_REUSED_EVALUATION",
             "model": parent["model"], "parent_scientific_fingerprint": parent["scientific_fingerprint"],
             "input_sha256": {str(p): file_hash(p) for p in inputs},
             "producer_sha256": {str(p): file_hash(p) for p in producers},
             "config": {"quality_prompts": 64, "timing_prompts": 32, "warmups_per_arm": 4,
                        "repetitions": 5, "batch_size": 1, "seed": 941, "cache": False,
                        "arms": ["teacher_full_vocab", "teacher_four_logits", "prefix_four_logits"]},
             "created_at": datetime.now(timezone.utc).isoformat()}
    value["scientific_fingerprint"] = digest(value)
    root.mkdir(parents=True)
    for path in producers:
        target = root/"producer"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(root/"plan.json", value)
    return {"scientific_fingerprint": value["scientific_fingerprint"]}


def verify(root):
    verify_parent(PARENT)
    plan = json.loads((root/"plan.json").read_text())
    if digest({k: v for k, v in plan.items() if k != "scientific_fingerprint"}) != plan["scientific_fingerprint"]:
        raise ValueError("Latency audit fingerprint changed")
    for section in ("input_sha256", "producer_sha256"):
        for name, expected in plan[section].items():
            if file_hash(name) != expected:
                raise ValueError(f"Frozen input changed: {name}")
            if section == "producer_sha256" and file_hash(root/"producer"/Path(name).relative_to(ROOT)) != expected:
                raise ValueError("Archived producer changed")
    return plan


def measure(functions, prompts, expected, config):
    for function in functions.values():
        for prompt in prompts[:config["warmups_per_arm"]]:
            function(prompt)
    rng, rows = random.Random(config["seed"]), []
    for repeat in range(config["repetitions"]):
        for index, prompt in enumerate(prompts):
            names = list(functions)
            rng.shuffle(names)
            for name in names:
                torch.cuda.synchronize()
                started = time.perf_counter()
                answer = functions[name](prompt)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                if answer != expected[name][index]:
                    raise ValueError(f"Timing prediction drift: {name}, {index}")
                rows.append({"arm": name, "repeat": repeat, "index": index, "answer": answer, "seconds": elapsed})
    return rows


def capture(root):
    from structure_net_resources.client import require_gpu_lease
    lease = require_gpu_lease([0])
    plan = verify(root)
    verify_parent(PARENT, model=True)
    directory = root/"capture"
    directory.mkdir(exist_ok=False)
    write_json(directory/"lease.json", {k: lease[k] for k in ("id", "holder", "state", "ready", "allocation", "end")})
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_small_qwen(plan["model"]["path"], device="cuda:0", dtype=torch.bfloat16)
    colour_ids = [tokenizer.encode(colour, add_special_tokens=False) for colour in COLOURS]
    if any(len(ids) != 1 for ids in colour_ids):
        raise ValueError("Matched label audit requires single-token labels")
    token_to_class = {ids[0]: i for i, ids in enumerate(colour_ids)}
    candidate_weight = model.lm_head.weight[[ids[0] for ids in colour_ids]].detach().clone()
    pairs = json.loads((PARENT/"tasks.json").read_text())["pairs"]["evaluation"]
    natural = json.loads((PARENT/"evaluation/natural.json").read_text())
    analysis = json.loads((PARENT/"analysis.json").read_text())
    selected = analysis["benchmark_candidate"]
    if not analysis["candidate_quality_gate"] or selected != {"layer": 29, "head": "linear_full"}:
        raise ValueError("Audit scope is the frozen qualified block-29 full linear head")
    prompts = [_prompt(tokenizer, p["source"]) for p in pairs]
    expected_teacher = [token_to_class.get(r["source"]["first_token_id"], -1) for r in natural]
    if any(c < 0 or COLOURS[c] != r["source"]["answer"] for c, r in zip(expected_teacher, natural)):
        raise ValueError("First-token decision is not equivalent to original completed answer")
    full = lambda p: token_to_class.get(teacher_choice(model, tokenizer, p), -1)
    four = lambda p: teacher_choice(model, tokenizer, p, candidate_weight)
    for index, prompt in enumerate(prompts):
        if full(prompt) != expected_teacher[index] or four(prompt) != expected_teacher[index]:
            raise ValueError("Single-label teacher changed recorded predictions")
    rows = measure({"teacher_full_vocab": full, "teacher_four_logits": four}, prompts[:32],
                   {"teacher_full_vocab": expected_teacher, "teacher_four_logits": expected_teacher}, plan["config"])
    teacher_peak = torch.cuda.max_memory_reserved()
    teacher_parameters = sum(p.numel() for p in model.parameters())
    require_gpu_lease([0])
    print(json.dumps({"stage": "teacher_complete", "elapsed_seconds": time.perf_counter()-started}), flush=True)
    del candidate_weight, full, four
    truncate_model(model, selected["layer"])
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    head = torch.load(PARENT/"fit.pt", map_location="cpu", weights_only=True)["heads"]["29"]["linear_full"]
    device_head = move(head, "cuda:0")
    prefix = lambda p: prefix_answer(model, tokenizer, p, device_head, "final")
    expected_prefix = analysis["head_predictions"]["29"]["linear_full"]
    if any(prefix(prompt) != expected_prefix[i] for i, prompt in enumerate(prompts)):
        raise ValueError("Matched prefix predictions changed")
    rows.extend(measure({"prefix_four_logits": prefix}, prompts[:32],
                        {"prefix_four_logits": expected_prefix}, plan["config"]))
    prefix_peak = torch.cuda.max_memory_reserved()
    require_gpu_lease([0])
    if max(teacher_peak, prefix_peak) > sum(lease["allocation"].values())*1024**2:
        raise RuntimeError("Reservation exceeded")
    stats = {}
    for name in plan["config"]["arms"]:
        times = [r["seconds"] for r in rows if r["arm"] == name]
        stats[name] = {"calls": len(times), "median_seconds": statistics.median(times),
                       "mean_seconds": statistics.mean(times), "min_seconds": min(times), "max_seconds": max(times)}
    truth = [COLOURS.index(p["source"]["answer"]) for p in pairs]
    write_json(directory/"timings.json", rows)
    result = {"study_id": STUDY_ID, "status": "completed", "confirmed": False,
              "scientific_fingerprint": plan["scientific_fingerprint"], "scope": "single-label lookup latency, one prefill, no continuation, no KV cache",
              "arms": stats, "teacher_correct": sum(a == b for a, b in zip(expected_teacher, truth)),
              "prefix_correct": sum(a == b for a, b in zip(expected_prefix, truth)), "quality_total": len(truth),
              "teacher_one_label_matches_completed_generation": True, "four_logits_match_full_vocab": True,
              "prefix_matches_frozen_head": True,
              "median_speedup_vs_full_vocab": stats["teacher_full_vocab"]["median_seconds"]/stats["prefix_four_logits"]["median_seconds"],
              "median_speedup_vs_four_logits": stats["teacher_four_logits"]["median_seconds"]/stats["prefix_four_logits"]["median_seconds"],
              "full_backbone_parameters": teacher_parameters, "remaining_backbone_parameters": sum(p.numel() for p in model.parameters()),
              "teacher_peak_reserved_bytes": teacher_peak, "prefix_peak_reserved_bytes": prefix_peak,
              "elapsed_seconds": time.perf_counter()-started, "lease_id": lease["id"],
              "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16"},
              "limitations": ["posthoc_reused_evaluation", "one_device_one_run", "prefix_timed_after_teacher", "warm_weights_batch_one", "fixed_synthetic_four_colour_task_not_general_tok_per_second"],
              "artifacts": {str(p.relative_to(root)): file_hash(p) for p in directory.iterdir() if p.is_file()}}
    verify(root)
    write_json(root/"result.json", result)
    write_json(root/"campaign_results.json", {**result, "result_path": "result.json", "result_sha256": file_hash(root/"result.json")})
    return {k: result[k] for k in ("arms", "median_speedup_vs_full_vocab", "median_speedup_vs_four_logits", "elapsed_seconds")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "capture"))
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](args.output), allow_nan=False))
