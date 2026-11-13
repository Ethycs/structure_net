"""Post-hoc head robustness and exact-copy witnesses of non-closed projections.

No new fitting. Selected cases and producer hashes are frozen before replay.
Witnesses concern an intervention-augmented state domain, not solely natural
hidden states or a general impossibility theorem for task-specific compression.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import torch

from .jspace_active_study import DEFAULT_ROOT as PARENT, verify as verify_parent, cpu_only
from .jspace_active_math import bounds, flatten_state, head_scores
from .jspace_donor_localization import forward_record, natural_row, compare_logits
from .jspace_lens import load_small_qwen
from .jspace_relation_capture import _prompt
from .jspace_relation_study import ROOT, digest, file_hash, save_tensor, write_json


STUDY_ID = "JSPACE-ACTIVE-CLOSURE-AUDIT-01"
DEFAULT_ROOT = ROOT/"data/experiments/jspace_active_closure_audit/20261113_v1"
PROTOCOL = ROOT/"docs/07 - Status Reports/jspace/2026-11-13_jspace-active-closure-audit-plan.md"
TEST = ROOT/"tests/structure_net/test_jspace_active_closure_audit.py"


def direct_patch(source, donor, scope):
    """Record actual bitwise equalities at the patched model boundary."""
    diagnostics = {}
    def patch(hidden):
        if not torch.equal(hidden[0].detach().cpu(), source):
            raise ValueError("Replay source boundary differs from natural source")
        first, stop = bounds(hidden.shape[1], scope)
        replacement = donor[first:stop].to(hidden)
        changed = hidden.clone()
        changed[0, first:stop] = replacement
        same = torch.equal(changed[0, first:stop], replacement)
        unchanged = torch.equal(changed[:, :first], hidden[:, :first]) and torch.equal(changed[:, stop:], hidden[:, stop:])
        if not same or not unchanged:
            raise ValueError("Exact-copy witness failed")
        diagnostics.update({"retained_bitwise_equal_to_donor": same, "outside_unchanged_from_source": unchanged,
                            "source_replay_bitwise_equal": True, "first": first, "stop": stop})
        return changed
    return patch, diagnostics


def prepare(root):
    cpu_only()
    parent = verify_parent(PARENT)
    if root.exists():
        raise FileExistsError(root)
    pairs = json.loads((PARENT/"tasks.json").read_text())["pairs"]["evaluation"]
    records = json.loads((PARENT/"evaluation/natural.json").read_text())
    selected = [p for p, r in zip(pairs, records) if r["source"]["original_correct"] and r["answer_donor"]["original_correct"]][:4]
    if len(selected) != 4:
        raise ValueError("Need four clean-correct paired worlds")
    inputs = [PARENT/name for name in ("plan.json", "tasks.json", "fit.pt", "fit.json", "evaluation_manifest.json",
                                      "evaluation/natural.pt", "evaluation/natural.json", "amended_result.json")]
    paths = [Path(__file__), PROTOCOL, TEST]
    value = {"study_id": STUDY_ID, "scope": "EXPLORATORY_POSTHOC_REUSED_EVALUATION",
             "model": parent["model"], "parent_scientific_fingerprint": parent["scientific_fingerprint"],
             "selected_case_ids": [p["source"]["case_id"] for p in selected], "sites": {"24": "question", "29": "final", "35": "final"},
             "input_sha256": {str(p): file_hash(p) for p in inputs}, "producer_sha256": {str(p): file_hash(p) for p in paths},
             "created_at": datetime.now(timezone.utc).isoformat()}
    value["scientific_fingerprint"] = digest(value)
    root.mkdir(parents=True)
    for path in paths:
        target = root/"producer"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(root/"plan.json", value)
    return {"scientific_fingerprint": value["scientific_fingerprint"], "selected_case_ids": value["selected_case_ids"]}


def verify(root):
    verify_parent(PARENT)
    plan = json.loads((root/"plan.json").read_text())
    if digest({k: v for k, v in plan.items() if k != "scientific_fingerprint"}) != plan["scientific_fingerprint"]:
        raise ValueError("Audit fingerprint changed")
    for kind in ("input_sha256", "producer_sha256"):
        for path, expected in plan[kind].items():
            if file_hash(path) != expected:
                raise ValueError("Audit input/producer changed")
            if kind == "producer_sha256" and file_hash(root/"producer"/Path(path).relative_to(ROOT)) != expected:
                raise ValueError("Audit archive changed")
    return plan


def head_audit(root):
    cpu_only()
    from .jspace_active_heads import target_labels
    from .jspace_country_probe import score_predictions
    torch.set_num_threads(4)
    verify(root)
    pairs = json.loads((PARENT/"tasks.json").read_text())["pairs"]["evaluation"]
    natural = json.loads((PARENT/"evaluation/natural.json").read_text())
    models = torch.load(PARENT/"fit.pt", map_location="cpu", weights_only=True)["heads"]
    features = torch.load(PARENT/"evaluation/natural.pt", map_location="cpu", weights_only=True)
    result = {"scope": "posthoc unchanged heads on correlated query/colour counterfactual variants", "heads": {}, "teacher": {}}
    for side in ("source", "donor", "answer_donor"):
        truth = target_labels(pairs, side, "answer")
        result["teacher"][side] = {"correct": sum(r[side]["original_correct"] for r in natural), "total": len(pairs),
                                   "generated_tokens": [len(r[side]["token_ids"]) for r in natural]}
        result["heads"][side] = {}
        for layer, heads in models.items():
            result["heads"][side][layer] = {}
            for name, head in heads.items():
                scores = head_scores(head, features[side][layer].float()).numpy()
                result["heads"][side][layer][name] = score_predictions(scores, truth)
    write_json(root/"head_audit.json", result)
    return {side: {key: {name: value["correct"] for name, value in heads.items()} for key, heads in layers.items()}
            for side, layers in result["heads"].items()}


def capture(root):
    from structure_net_resources.client import require_gpu_lease
    lease = require_gpu_lease([0])
    plan = verify(root)
    for name, expected in plan["model"]["files"].items():
        if file_hash(Path(plan["model"]["path"])/name) != expected:
            raise ValueError("Checkpoint changed")
    directory = root/"capture"
    directory.mkdir(exist_ok=False)
    write_json(directory/"lease.json", {k: lease[k] for k in ("id", "holder", "state", "ready", "allocation", "end")})
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_small_qwen(plan["model"]["path"], device="cuda:0", dtype=torch.bfloat16)
    pairs = json.loads((PARENT/"tasks.json").read_text())["pairs"]["evaluation"]
    selected = [p for p in pairs if p["source"]["case_id"] in plan["selected_case_ids"]]
    rows, natural = [], []
    for index, pair in enumerate(selected):
        require_gpu_lease([0])
        source = forward_record(model, tokenizer, _prompt(tokenizer, pair["source"]), layers=(24, 25, 29, 30, 35))
        donor = forward_record(model, tokenizer, _prompt(tokenizer, pair["answer_donor"]), layers=(24, 25, 29, 30, 35))
        source_row, donor_row = natural_row(source, pair["source"]), natural_row(donor, pair["answer_donor"])
        if not source_row["original_correct"] or not donor_row["original_correct"]:
            raise ValueError("Clean replay drifted from case eligibility")
        natural.append({"source": source_row, "answer_donor": donor_row})
        save_tensor(directory/f"natural_{index}.pt", {"source": source, "donor": donor})
        for key, scope in plan["sites"].items():
            layer = int(key)
            patch, diagnostics = direct_patch(source["states"][layer], donor["states"][layer], scope)
            output = forward_record(model, tokenizer, _prompt(tokenizer, pair["source"]), layer=layer, patch=patch,
                                    layers=(layer+1,) if layer < 35 else ())
            logit_comparison = compare_logits(output["logits"], donor["logits"])
            if layer == 35 and not logit_comparison["exact"]:
                raise ValueError("Terminal exact-logit control failed")
            next_state = None
            if layer < 35:
                actual = flatten_state(output["states"][layer+1], scope).float()
                expected = flatten_state(donor["states"][layer+1], scope).float()
                next_state = {"bitwise_equal": torch.equal(actual, expected), "max_absolute_difference": float((actual-expected).abs().max()),
                              "relative_l2_difference": float((actual-expected).norm()/expected.norm().clamp_min(1e-12))}
            row = {"case_id": pair["source"]["case_id"], "layer": layer, "scope": scope, "exact_copy": diagnostics,
                   "natural_source_answer": source_row["answer"], "natural_donor_answer": donor_row["answer"],
                   "patched_answer": output["generation"]["answer"], "donor_logits": logit_comparison,
                   "next_retained_state": next_state,
                   "task_output_counterexample": layer < 35 and not logit_comparison["argmax_equal"]}
            rows.append(row)
            save_tensor(directory/f"patch_{index}_{layer}.pt", output)
            print(json.dumps(row), flush=True)
    require_gpu_lease([0])
    if torch.cuda.max_memory_reserved() > sum(lease["allocation"].values())*1024**2:
        raise RuntimeError("Reservation exceeded")
    write_json(directory/"rows.json", {"natural": natural, "rows": rows})
    result = {"study_id": STUDY_ID, "status": "completed", "confirmed": False, "scientific_fingerprint": plan["scientific_fingerprint"],
              "scope": "exact retained-state equality in BF16; natural-versus-hybrid intervention domain",
              "case_count": 4, "transition_counterexamples": sum(r["next_retained_state"] is not None and not r["next_retained_state"]["bitwise_equal"] for r in rows),
              "task_output_counterexamples": sum(r["task_output_counterexample"] for r in rows),
              "terminal_logit_controls_passed": sum(r["layer"] == 35 and r["donor_logits"]["exact"] for r in rows),
              "rows": rows, "elapsed_seconds": time.perf_counter()-started, "lease_id": lease["id"],
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "classification": "measured_closure_witnesses_on_intervention_domain",
              "artifacts": {str(p.relative_to(root)): file_hash(p) for p in sorted(directory.rglob("*")) if p.is_file()},
              "head_audit_sha256": file_hash(root/"head_audit.json")}
    verify(root)
    write_json(root/"result.json", result)
    write_json(root/"campaign_results.json", {**result, "result_path": "result.json", "result_sha256": file_hash(root/"result.json")})
    return {k: result[k] for k in ("classification", "transition_counterexamples", "task_output_counterexamples", "terminal_logit_controls_passed", "elapsed_seconds")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "head_audit", "capture"))
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](args.output), allow_nan=False))
