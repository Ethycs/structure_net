"""Dense donor patch localization with exact logit and token-alignment controls."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import torch

from .jspace_causal_components import cpu_only, render_case, query_start, competence, paired_test, success_metrics
from .jspace_lens import load_small_qwen
from .jspace_relation_capture import _prompt
from .jspace_relation_study import ROOT, digest, file_hash, save_tensor, write_json, verify_plan as verify_relation
from .jspace_relation_tasks import COUNTRIES, COLOURS, _make_split
from .jspace_runtime import generate_answer
from .jspace_tasks import normalize_answer

STUDY_ID = "JSPACE-DONOR-LOCALIZATION-01"
PROTOCOL = "docs/07 - Status Reports/jspace/2026-11-13_jspace-donor-localization-plan.md"
RELATION = ROOT/"data/experiments/jspace_relation_propagation/20261113_neural_v1"
CAUSAL = ROOT/"data/experiments/jspace_causal_components/20261113_v1"
CONFIG = {"seed": 173, "shakedown_seed": 1000176, "candidate_count": 512, "shakedown_candidate_count": 128,
          "count": 64, "per_country_colour": 4, "shakedown_count": 4, "layers": [5, 11, 18, 24, 29, 34, 35],
          "scopes": ["final", "query_suffix", "query_except_final"], "max_new_tokens": 8,
          "minimum_clean_accuracy": .9, "minimum_joint_fraction": .8, "minimum_patch_success": .8,
          "primary_layer": 24, "minimum_primary_advantage": .5, "maximum_p": .05,
          "dtype": "bfloat16", "max_prompt_tokens": 256}


def pair_alignment(tokenizer, source, donor):
    sp, dp = _prompt(tokenizer, source), _prompt(tokenizer, donor)
    si, di = tokenizer(sp)["input_ids"], tokenizer(dp)["input_ids"]
    ss, ds = query_start(tokenizer, sp, source["question"]), query_start(tokenizer, dp, donor["question"])
    if len(si) != len(di) or ss != ds or si[:ss] != di[:ds] or not ss < len(si)-1:
        return None
    if len(si) > CONFIG["max_prompt_tokens"]:
        raise ValueError("Prompt exceeds bounded memory protocol")
    return {"source_ids": si, "donor_ids": di, "question_start": ss, "sequence_length": len(si)}


def balanced_selection(aligned, per_cell):
    result = []
    for country in COUNTRIES:
        for colour in COLOURS:
            cell = sorted((pair for pair in aligned if pair["source"]["intermediate"] == country
                           and pair["source"]["answer"] == colour), key=lambda p: p["source"]["case_id"])
            if len(cell) < per_cell:
                raise ValueError(f"Insufficient aligned candidates for {country}/{colour}")
            result.extend(cell[:per_cell])
    return sorted(result, key=lambda p: p["source"]["case_id"])


def make_tasks(tokenizer):
    prior = [json.loads((root/"tasks.json").read_text()) for root in (RELATION, CAUSAL)]
    seen = {row["world_id"] for dataset in prior for rows in dataset["splits"].values() for row in rows}
    excluded_count = len(seen)
    groups, statistics = {}, {}
    for split, count, seed in (("evaluation", CONFIG["candidate_count"], CONFIG["seed"]),
                               ("shakedown", CONFIG["shakedown_candidate_count"], CONFIG["shakedown_seed"])):
        candidates = _make_split(split, count, seed, seen)
        aligned = []
        for raw in candidates:
            source = render_case(raw)
            query = next(name for name, country in source["person_to_country"].items() if country == source["swap_to"])
            donor = render_case(source, query=query)
            alignment = pair_alignment(tokenizer, source, donor)
            if alignment is not None:
                aligned.append({"source": source, "donor": donor, "alignment": alignment})
        selected = (balanced_selection(aligned, CONFIG["per_country_colour"]) if split == "evaluation"
                    else sorted(aligned, key=lambda p: p["source"]["case_id"])[:CONFIG["shakedown_count"]])
        expected = CONFIG["count"] if split == "evaluation" else CONFIG["shakedown_count"]
        if len(selected) != expected:
            raise ValueError("Candidate pool cannot fill frozen count")
        groups[split] = selected
        statistics[split] = {"generated": count, "aligned": len(aligned), "selected": len(selected),
                             "candidate_digest": digest(candidates), "seed": seed}
    worlds = [pair["source"]["world_id"] for pairs in groups.values() for pair in pairs]
    if len(worlds) != len(set(worlds)):
        raise ValueError("Selected named worlds overlap")
    return {"study_id": STUDY_ID, "pairs": groups, "statistics": statistics,
            "excluded_parent_world_count": excluded_count, "scope": "fresh_token_length_matched_named_world_pairs"}


def sources():
    names = [PROTOCOL, "pixi.lock", "pyproject.toml", "tests/structure_net/test_jspace_donor_localization.py"]
    names += [f"experiments/structure_net/jspace_{name}.py" for name in (
        "donor_localization", "causal_components", "relation_study", "relation_tasks", "relation_capture", "lens", "runtime", "tasks")]
    names += [str(p.relative_to(ROOT)) for p in (ROOT/"src/structure_net_resources").glob("*.py")]
    return [ROOT/name for name in sorted(names)]


def prepare(root):
    cpu_only()
    from transformers import AutoTokenizer
    if root.exists():
        raise FileExistsError("A new append-only root is required")
    parent = verify_relation(RELATION, verify_model=True)
    tokenizer = AutoTokenizer.from_pretrained(parent["model"]["path"], local_files_only=True, trust_remote_code=False)
    tasks = make_tasks(tokenizer)
    inputs = [root_/name for root_ in (RELATION, CAUSAL) for name in ("plan.json", "tasks.json", "campaign_results.json")]
    plan = {"study_id": STUDY_ID, "schema_version": 1, "config": CONFIG, "model": parent["model"],
            "prepared_at": datetime.now(timezone.utc).isoformat(), "conformance": "PROSPECTIVE_SINGLE_CHECKPOINT_PILOT",
            "dataset_digest": digest(tasks), "producer_sha256": {str(p): file_hash(p) for p in sources()},
            "input_sha256": {str(p): file_hash(p) for p in inputs}}
    plan["scientific_fingerprint"] = digest(plan)
    root.mkdir(parents=True)
    for path in sources():
        target = root/"producer"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(root/"tasks.json", tasks)
    write_json(root/"plan.json", plan)
    return {"status": "prepared", "statistics": tasks["statistics"], "scientific_fingerprint": plan["scientific_fingerprint"]}


def verify(root, *, model=False):
    plan = json.loads((root/"plan.json").read_text())
    if (plan["study_id"] != STUDY_ID or plan["config"] != CONFIG
            or digest({k: v for k, v in plan.items() if k != "scientific_fingerprint"}) != plan["scientific_fingerprint"]):
        raise ValueError("Plan/configuration fingerprint mismatch")
    for section in ("producer_sha256", "input_sha256"):
        for path, expected in plan[section].items():
            if file_hash(path) != expected:
                raise ValueError(f"Changed frozen source/input: {path}")
            if section == "producer_sha256" and file_hash(root/"producer"/Path(path).relative_to(ROOT)) != expected:
                raise ValueError("Archived producer changed")
    if digest(json.loads((root/"tasks.json").read_text())) != plan["dataset_digest"]:
        raise ValueError("Frozen cases changed")
    if model:
        for name, expected in plan["model"]["files"].items():
            if file_hash(Path(plan["model"]["path"])/name) != expected:
                raise ValueError("Model changed")
    return plan


def forward_record(model, tokenizer, prompt, *, layers=(), layer=None, patch=None):
    """Wrap the frozen greedy decoder, capturing whole prefill states and first logits."""
    states, logits, handles = {}, [], []
    def block_hook(index):
        def hook(module, inputs, output):
            if index not in states:
                state = output[0] if isinstance(output, tuple) else output
                state = state[0].detach().cpu().clone()
                if not bool(torch.isfinite(state).all()):
                    raise FloatingPointError("Nonfinite layer state")
                states[index] = state
        return hook
    def head_hook(module, inputs, output):
        if not logits:
            logits.append(output[0].detach().float().cpu().clone())
    try:
        for index in layers:
            handles.append(model.model.layers[index].register_forward_hook(block_hook(index)))
        handles.append(model.lm_head.register_forward_hook(head_hook))
        result = generate_answer(model, tokenizer, prompt, layer=layer, patch=patch, max_new_tokens=8)
    finally:
        for handle in handles:
            handle.remove()
    if len(logits) != 1 or set(states) != set(layers) or not bool(torch.isfinite(logits[0]).all()):
        raise ValueError("Incomplete or nonfinite forward capture")
    return {"generation": {k: v for k, v in result.items() if k != "observed"}, "states": states, "logits": logits[0]}


def make_donor_patch(donor_suffix, boundary, scope):
    if scope not in CONFIG["scopes"] or donor_suffix.ndim != 2:
        raise ValueError("Invalid scope or suffix")
    diagnostics = {}
    def patch(hidden):
        length = boundary+len(donor_suffix)
        if hidden.shape != (1, length, donor_suffix.shape[1]) or not 0 <= boundary < length-1:
            raise ValueError("Source/donor sequence alignment mismatch")
        if scope == "final":
            first, last = length-1, length
        else:
            first, last = boundary, length-(scope == "query_except_final")
        replacement = donor_suffix[first-boundary:last-boundary].to(device=hidden.device, dtype=hidden.dtype)
        updated = hidden.clone()
        updated[0, first:last] = replacement
        if (not torch.equal(updated[0, first:last], replacement) or not torch.equal(updated[:, :first], hidden[:, :first])
                or not torch.equal(updated[:, last:], hidden[:, last:])):
            raise ValueError("Exact copy or unchanged-region invariant failed")
        diagnostics.update({"first": first, "stop": last, "positions": last-first, "exact_copy": True,
                            "relative_delta_l2": float((updated.float()-hidden.float()).norm()/hidden.float().norm().clamp_min(1e-12))})
        return updated
    return patch, diagnostics


def compare_logits(actual, expected):
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("Invalid logit comparison")
    return {"exact": torch.equal(actual, expected), "max_absolute_difference": float((actual-expected).abs().max()),
            "argmax_equal": int(actual.argmax()) == int(expected.argmax())}


def natural_row(forward, case):
    generation = forward["generation"]
    answer = normalize_answer(generation["answer"]) if generation["answer"] else None
    return {"case_id": case["case_id"], "expected_answer": case["answer"],
            "original_correct": answer == normalize_answer(case["answer"]), **generation}


def patch_row(forward, case, layer, scope, donor_logits, diagnostics):
    generation = forward["generation"]
    answer = normalize_answer(generation["answer"]) if generation["answer"] else None
    return {"case_id": case["case_id"], "layer": layer, "scope": scope,
            "original_correct": answer == normalize_answer(case["answer"]),
            "target_correct": answer == normalize_answer(case["swap_answer"]),
            "expected_answer": case["answer"], "expected_target": case["swap_answer"],
            "donor_logits": compare_logits(forward["logits"], donor_logits), "diagnostics": diagnostics, **generation}


def capture(root):
    from structure_net_resources.client import require_gpu_lease
    lease = require_gpu_lease([0])  # Must precede CUDA queries/loading.
    plan = verify(root, model=True)
    directory = root/"capture"
    directory.mkdir(exist_ok=False)
    write_json(directory/"lease.json", {key: lease[key] for key in ("id", "holder", "state", "ready", "allocation", "end")})
    started, counts = time.perf_counter(), Counter()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Reserved device must support BF16")
    torch.set_num_threads(4)
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_small_qwen(plan["model"]["path"], device="cuda:0", dtype=torch.bfloat16)
    tasks = json.loads((root/"tasks.json").read_text())

    def event(stage, row):
        counts[stage] += 1
        with (directory/"events.jsonl").open("a") as stream:
            stream.write(json.dumps({"stage": stage, "row": row, "elapsed_seconds": time.perf_counter()-started}, allow_nan=False)+"\n")
        if counts[stage] == 1 or counts[stage] % 32 == 0:
            require_gpu_lease([0])
            print(json.dumps({"stage": stage, "completed": counts[stage], "elapsed_seconds": round(time.perf_counter()-started, 2)}), flush=True)

    def finish(classification, **extra):
        require_gpu_lease([0])
        if torch.cuda.max_memory_reserved() > sum(lease["allocation"].values())*1024**2:
            raise RuntimeError("Memory exceeded the reservation")
        result = {"study_id": STUDY_ID, "status": "completed", "classification": classification,
                  "scientific_fingerprint": plan["scientific_fingerprint"], "counts": dict(counts), "lease_id": lease["id"],
                  "elapsed_seconds": time.perf_counter()-started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "release_owner": "inference-ctl on child exit",
                  "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
                  "artifacts": {str(p.relative_to(root)): file_hash(p) for p in sorted(directory.rglob("*")) if p.is_file()}, **extra}
        write_json(root/"capture_manifest.json", result)
        return {key: value for key, value in result.items() if key != "artifacts"}

    def clean_pair(pair, label, save_path):
        if pair_alignment(tokenizer, pair["source"], pair["donor"]) != pair["alignment"]:
            raise ValueError("Tokenizer alignment changed after freeze")
        boundary = pair["alignment"]["question_start"]
        outputs = []
        for side in ("source", "donor"):
            output = forward_record(model, tokenizer, _prompt(tokenizer, pair[side]), layers=CONFIG["layers"])
            event(label+"_"+side, natural_row(output, pair[side]))
            outputs.append(output)
        source, donor = outputs
        equal = all(torch.equal(source["states"][layer][:boundary], donor["states"][layer][:boundary]) for layer in CONFIG["layers"])
        if not equal:
            raise ValueError("Identical token prefixes produced unequal hidden prefixes")
        saved = {"boundary": boundary, "source_logits": source["logits"], "donor_logits": donor["logits"],
                 "source_suffix": {layer: source["states"][layer][boundary:].clone() for layer in CONFIG["layers"]},
                 "donor_suffix": {layer: donor["states"][layer][boundary:].clone() for layer in CONFIG["layers"]}}
        save_tensor(save_path, saved)
        return saved, {"source": natural_row(source, pair["source"]), "donor": natural_row(donor, pair["donor"]),
                       "hidden_prefix_equal_all_layers": equal}

    def patched(pair, saved, layer, scope, *, self_copy=False):
        suffix = saved["source_suffix" if self_copy else "donor_suffix"][layer]
        patch, diagnostic = make_donor_patch(suffix, saved["boundary"], scope)
        output = forward_record(model, tokenizer, _prompt(tokenizer, pair["source"]), layer=layer, patch=patch)
        if not diagnostic:
            raise RuntimeError("Patch did not run")
        row = patch_row(output, pair["source"], layer, scope, saved["donor_logits"], diagnostic)
        if self_copy:
            row["source_logits"] = compare_logits(output["logits"], saved["source_logits"])
        return output, row

    smoke_rows = []
    for index, pair in enumerate(tasks["pairs"]["shakedown"]):
        path = directory/f"shakedown/pair_{index}.pt"
        saved, natural = clean_pair(pair, "shakedown", path)
        restored = torch.load(path, map_location="cpu", weights_only=True)
        if not torch.equal(restored["donor_logits"], saved["donor_logits"]) or any(
                not torch.equal(restored["donor_suffix"][layer], saved["donor_suffix"][layer]) for layer in CONFIG["layers"]):
            raise ValueError("Shakedown save/reload failed")
        _, identity = patched(pair, restored, 35, "final", self_copy=True)
        _, terminal = patched(pair, restored, 35, "final")
        _, suffix = patched(pair, restored, 5, "query_suffix")
        if not identity["source_logits"]["exact"] or not terminal["donor_logits"]["exact"] or not suffix["donor_logits"]["exact"]:
            return finish("shakedown_logit_control_failed")
        smoke_rows.append({"natural": natural, "identity": identity, "terminal": terminal, "suffix": suffix})
    write_json(directory/"shakedown/result.json", {"classification": "systems_lifecycle_only_not_quality_evidence",
               "roundtrip_equal": True, "rows": smoke_rows, "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
    if torch.cuda.max_memory_reserved() > sum(lease["allocation"].values())*1024**2:
        raise RuntimeError("Memory pilot exceeds reservation")
    natural_rows = []
    for index, pair in enumerate(tasks["pairs"]["evaluation"]):
        _, row = clean_pair(pair, "evaluation", directory/f"pairs/{index:03d}.pt")
        natural_rows.append(row)
    write_json(directory/"natural.json", natural_rows)
    gate = competence([row["source"] for row in natural_rows], [row["donor"] for row in natural_rows])
    terminal_rows, identities, grid = [], [], []
    for index, pair in enumerate(tasks["pairs"]["evaluation"]):
        saved = torch.load(directory/f"pairs/{index:03d}.pt", map_location="cpu", weights_only=True)
        _, identity = patched(pair, saved, 35, "final", self_copy=True)
        terminal_output, terminal = patched(pair, saved, 35, "final")
        event("identity", identity)
        event("terminal", terminal)
        identities.append(identity)
        terminal_rows.append(terminal)
        save_tensor(directory/f"terminal_logits/{index:03d}.pt", terminal_output["logits"])
        if (not identity["source_logits"]["exact"] or identity["token_ids"] != natural_rows[index]["source"]["token_ids"]
                or not terminal["donor_logits"]["exact"]):
            write_json(directory/"controls.json", {"identity": identities, "terminal": terminal_rows})
            return finish("terminal_logit_control_failed", competence=gate)
    write_json(directory/"controls.json", {"identity": identities, "terminal": terminal_rows})
    if not gate["passed"]:
        return finish("clean_competence_failed", competence=gate)
    terminal_success = sum(row["target_correct"] for row, eligible in zip(terminal_rows, gate["eligible"]) if eligible)/gate["eligible_count"]
    if terminal_success < CONFIG["minimum_patch_success"]:
        return finish("terminal_answer_control_failed", competence=gate, terminal_success=terminal_success)
    grid.extend(terminal_rows)
    for layer in CONFIG["layers"]:
        for scope in CONFIG["scopes"]:
            if layer == 35 and scope == "final":
                continue
            for index, pair in enumerate(tasks["pairs"]["evaluation"]):
                saved = torch.load(directory/f"pairs/{index:03d}.pt", map_location="cpu", weights_only=True)
                _, row = patched(pair, saved, layer, scope)
                event("grid", row)
                grid.append(row)
                if scope == "query_suffix" and not row["donor_logits"]["exact"]:
                    write_json(directory/"grid.json", grid)
                    return finish("suffix_logit_control_failed", competence=gate, failed_layer=layer)
    write_json(directory/"grid.json", grid)
    return finish("localization_capture_completed", competence=gate, terminal_success=terminal_success)


def aggregate(rows, cases, gate):
    eligible = {pair["source"]["case_id"] for pair, yes in zip(cases, gate["eligible"]) if yes}
    if len(rows) != len(cases)*len(CONFIG["layers"])*len(CONFIG["scopes"]):
        raise ValueError("Incomplete grid")
    metrics, all_metrics, grouped = {}, {}, {}
    expected_ids = {pair["source"]["case_id"] for pair in cases}
    for layer in CONFIG["layers"]:
        metrics[str(layer)], all_metrics[str(layer)] = {}, {}
        for scope in CONFIG["scopes"]:
            selected = [row for row in rows if row["layer"] == layer and row["scope"] == scope]
            if len(selected) != len(cases) or {r["case_id"] for r in selected} != expected_ids:
                raise ValueError("Missing/duplicate grid cases")
            group = [row for row in selected if row["case_id"] in eligible]
            grouped[(layer, scope)] = group
            score = success_metrics(group)
            score.update({"first_token_donor_agreement": sum(r["donor_logits"]["argmax_equal"] for r in group)/len(group),
                          "exact_donor_logit_fraction": sum(r["donor_logits"]["exact"] for r in group)/len(group),
                          "maximum_donor_logit_error": max(r["donor_logits"]["max_absolute_difference"] for r in group)})
            metrics[str(layer)][scope] = score
            all_metrics[str(layer)][scope] = success_metrics(selected)
    primary = paired_test(grouped[(24, "query_except_final")], grouped[(24, "final")])
    supported = primary["difference"] >= CONFIG["minimum_primary_advantage"] and primary["two_sided_exact_p"] <= CONFIG["maximum_p"]
    usable = {scope: [layer for layer in CONFIG["layers"] if metrics[str(layer)][scope]["target_success"] >= CONFIG["minimum_patch_success"]]
              for scope in CONFIG["scopes"]}
    return {"primary_metric": primary["difference"], "primary_comparison": primary, "metrics": metrics,
            "all_case_metrics": all_metrics, "usable_layers": usable, "primary_metric_status": "measured",
            "classification": "distributed_question_state_supported" if supported else "patching_valid_no_large_distributed_advantage"}


def analyze(root):
    cpu_only()
    plan = verify(root)
    manifest = json.loads((root/"capture_manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["scientific_fingerprint"] != plan["scientific_fingerprint"]:
        raise ValueError("Incomplete/mismatched capture")
    for name, expected in manifest["artifacts"].items():
        if not (root/name).resolve().is_relative_to(root.resolve()) or file_hash(root/name) != expected:
            raise ValueError("Capture artifact changed")
    result = {"study_id": STUDY_ID, "schema_version": 1, "status": "completed", "confirmed": False,
              "conformance": plan["conformance"], "config": CONFIG, "scientific_fingerprint": plan["scientific_fingerprint"],
              "classification": manifest["classification"], "capture": manifest,
              "primary_metric": None, "primary_metric_status": "not_measured_gate_stopped",
              "primary_metric_scope": "block24_counterfactual_success_query_except_final_minus_final_on_clean_correct_pairs",
              "requested_runs": 1, "completed_runs": 1, "failed_runs": 0, "retries": 0,
              "capture_manifest_sha256": file_hash(root/"capture_manifest.json")}
    if manifest["classification"] == "localization_capture_completed":
        tasks = json.loads((root/"tasks.json").read_text())
        result.update(aggregate(json.loads((root/"capture/grid.json").read_text()), tasks["pairs"]["evaluation"], manifest["competence"]))
    write_json(root/"result.json", result)
    write_json(root/"campaign_results.json", {**result, "result_path": "result.json", "result_sha256": file_hash(root/"result.json")})
    return {key: result[key] for key in ("status", "classification", "primary_metric", "primary_metric_status")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "capture", "analyze"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = {"prepare": prepare, "capture": capture, "analyze": analyze}[args.stage](args.output)
    except BaseException as exc:
        path = args.output/f"failure_{args.stage}.json"
        if args.output.exists() and not path.exists() and not isinstance(exc, FileExistsError):
            write_json(path, {"status": "invalid_stage", "stage": args.stage, "error_type": type(exc).__name__, "error": str(exc)})
        raise
    print(json.dumps(result))


if __name__ == "__main__":
    main()
