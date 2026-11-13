"""Prospective active-position CAV interventions and task-head comparison."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import torch

from .jspace_active_math import flatten_state, make_patch, task_dictionary, head_scores, move, tensor_bytes
from .jspace_causal_components import cpu_only, render_case, competence, paired_test, success_metrics, intervention_record
from .jspace_donor_localization import pair_alignment, balanced_selection, forward_record, natural_row, make_donor_patch, compare_logits
from .jspace_lens import load_small_qwen
from .jspace_relation_capture import _prompt
from .jspace_relation_study import ROOT, digest, file_hash, save_tensor, write_json, verify_plan as verify_relation
from .jspace_relation_tasks import COUNTRIES, COLOURS, _make_split, world_signature


STUDY_ID = "JSPACE-ACTIVE-CAV-HEADS-01"
PROTOCOL = "docs/07 - Status Reports/jspace/2026-11-13_jspace-active-cav-heads-plan.md"
DEFAULT_ROOT = ROOT/"data/experiments/jspace_active_cav_heads/20261113_v1"
RELATION = ROOT/"data/experiments/jspace_relation_propagation/20261113_neural_v1"
PREVIOUS = [RELATION, ROOT/"data/experiments/jspace_causal_components/20261113_v1",
            ROOT/"data/experiments/jspace_donor_localization/20261113_v1"]
INVENTORY = ROOT/"data/experiments/jspace_cav_pilot/20261113T0728Z_pilot02/runs/pilot/seed_7/lens.pt"
CONFIG = {"seed": 233, "random_seed": 251, "head_seed": 263,
          "counts": {"train": 512, "development": 64, "evaluation": 64, "shakedown": 4},
          "sites": {"24": "question", "29": "final"}, "suffix_tokens": 21,
          "calibration_count": 32, "vjp_batch": 2, "dictionary_size": 94,
          "sparse_budget": 25, "heads": ["linear_full", "linear", "quadratic", "vanishing", "mlp"],
          "encoder_width": 8, "polynomials_per_class": 4, "max_new_tokens": 8,
          "main_arms": ["dense", "full", "J", "remainder", "random_full", "random_J", "remainder_norm_J", "colour"],
          "answer_only_arms": ["dense", "full", "colour"],
          "minimum_clean_accuracy": .9, "minimum_joint_fraction": .8, "minimum_donor_success": .8,
          "minimum_cav_target_success": .2, "minimum_country_specific_retention": .8,
          "minimum_J_advantage": .1, "maximum_p": .05,
          "head_minimum_accuracy": .95, "maximum_head_extra_errors": 1,
          "dtype": "bfloat16", "maximum_prompt_tokens": 256}


def answer_donor(source):
    row = copy.deepcopy(source)
    first, second = source["intermediate"], source["swap_to"]
    colours = row["country_to_colour"]
    colours[first], colours[second] = colours[second], colours[first]
    row.update({"case_id": source["case_id"]+":answer_donor", "answer": source["swap_answer"],
                "swap_answer": source["answer"], "world_id": world_signature(row["person_to_country"], colours)})
    return render_case(row)


def make_tasks(tokenizer):
    seen = set()
    for root in PREVIOUS:
        old = json.loads((root/"tasks.json").read_text())
        for rows in old.get("splits", {}).values():
            seen.update(r["world_id"] for r in rows)
        for pairs in old.get("pairs", {}).values():
            seen.update(p["source"]["world_id"] for p in pairs)
    excluded = len(seen)
    groups, statistics = {}, {}
    for split, count in CONFIG["counts"].items():
        candidate_count = 128 if split == "shakedown" else max(1024, count*8)
        candidates = _make_split(split, candidate_count, CONFIG["seed"]+(1000003 if split == "shakedown" else 0), seen)
        aligned = []
        for raw in candidates:
            source = render_case(raw)
            query = next(n for n, c in source["person_to_country"].items() if c == source["swap_to"])
            donor = render_case(source, query=query)
            alignment = pair_alignment(tokenizer, source, donor)
            if alignment is None or alignment["sequence_length"]-alignment["question_start"] != CONFIG["suffix_tokens"]:
                continue
            altered = answer_donor(source)
            if altered["world_id"] in seen or pair_alignment(tokenizer, source, altered) is not None:
                # Changed facts must not be mistaken for an identical-prefix donor.
                continue
            altered_ids = tokenizer(_prompt(tokenizer, altered))["input_ids"]
            if len(altered_ids) != alignment["sequence_length"]:
                continue
            seen.add(altered["world_id"])
            aligned.append({"source": source, "donor": donor, "answer_donor": altered, "alignment": alignment})
        chosen = (balanced_selection(aligned, count//16) if split != "shakedown"
                  else sorted(aligned, key=lambda p: p["source"]["case_id"])[:count])
        if len(chosen) != count:
            raise ValueError("Candidate pool does not fill the declared split")
        groups[split] = chosen
        statistics[split] = {"candidates": candidate_count, "aligned_fixed_suffix": len(aligned), "selected": len(chosen),
                             "candidate_digest": digest(candidates)}
    all_worlds = [p[side]["world_id"] for group in groups.values() for p in group for side in ("source", "answer_donor")]
    if len(all_worlds) != len(set(all_worlds)):
        raise ValueError("Selected source/answer-donor worlds overlap")
    calibration = balanced_selection(groups["train"], 2)
    return {"study_id": STUDY_ID, "pairs": groups, "statistics": statistics,
            "calibration_ids": [p["source"]["case_id"] for p in calibration], "excluded_prior_worlds": excluded,
            "independent_unit": "source world, with query and answer-only donor variants",
            "scope": "fresh worlds; disjoint split name pools; one-token query names; shared JSON format"}


def source_paths():
    names = [PROTOCOL, "pyproject.toml", "pixi.lock", "tests/structure_net/test_jspace_active_cav_heads.py"]
    names += [f"experiments/structure_net/jspace_{name}.py" for name in (
        "active_study", "active_math", "active_heads", "active_benchmark", "causal_components", "donor_localization", "lens", "runtime",
        "relation_capture", "relation_study", "relation_tasks", "tasks", "country_probe", "residual_expansion")]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/"src/structure_net_resources").glob("*.py"))]
    return [ROOT/name for name in names]


def prepare(root):
    cpu_only()
    from transformers import AutoTokenizer
    if root.exists():
        raise FileExistsError("A fresh append-only study root is required")
    parent = verify_relation(RELATION, verify_model=True)
    tokenizer = AutoTokenizer.from_pretrained(parent["model"]["path"], local_files_only=True)
    tasks = make_tasks(tokenizer)
    inventory = torch.load(INVENTORY, map_location="cpu", weights_only=True)["metadata"]
    if len(inventory["token_ids"]) != CONFIG["dictionary_size"] or not set(parent["concept_token_ids"]).issubset(inventory["token_ids"]):
        raise ValueError("Dictionary inventory changed")
    inputs = [INVENTORY]+[r/name for r in PREVIOUS for name in ("plan.json", "tasks.json", "campaign_results.json")]
    plan = {"study_id": STUDY_ID, "schema_version": 1, "config": CONFIG, "model": parent["model"],
            "token_ids": inventory["token_ids"], "token_strings": inventory["token_strings"],
            "country_token_ids": parent["concept_token_ids"], "dataset_digest": digest(tasks),
            "conformance": "PROSPECTIVE_SINGLE_CHECKPOINT_PILOT", "prepared_at": datetime.now(timezone.utc).isoformat(),
            "producer_sha256": {str(p): file_hash(p) for p in source_paths()},
            "input_sha256": {str(p): file_hash(p) for p in inputs}}
    plan["scientific_fingerprint"] = digest(plan)
    root.mkdir(parents=True)
    for path in source_paths():
        target = root/"producer"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(root/"tasks.json", tasks)
    write_json(root/"plan.json", plan)
    return {"scientific_fingerprint": plan["scientific_fingerprint"], "statistics": tasks["statistics"]}


def verify(root, *, model=False):
    plan = json.loads((root/"plan.json").read_text())
    if plan["study_id"] != STUDY_ID or plan["config"] != CONFIG or digest({k: v for k, v in plan.items() if k != "scientific_fingerprint"}) != plan["scientific_fingerprint"]:
        raise ValueError("Scientific fingerprint mismatch")
    for section in ("producer_sha256", "input_sha256"):
        for path, expected in plan[section].items():
            if file_hash(path) != expected:
                raise ValueError(f"Frozen producer/input changed: {path}")
            if section == "producer_sha256" and file_hash(root/"producer"/Path(path).relative_to(ROOT)) != expected:
                raise ValueError("Archived producer changed")
    if digest(json.loads((root/"tasks.json").read_text())) != plan["dataset_digest"]:
        raise ValueError("Cases changed")
    if model:
        for name, expected in plan["model"]["files"].items():
            if file_hash(Path(plan["model"]["path"])/name) != expected:
                raise ValueError("Checkpoint changed")
    return plan


def checked_manifest(root, stage):
    plan = verify(root)
    record = json.loads((root/f"{stage}_manifest.json").read_text())
    if record["scientific_fingerprint"] != plan["scientific_fingerprint"] or record["status"] != "completed":
        raise ValueError("Incomplete or mismatched stage")
    for path, expected in record["artifacts"].items():
        if file_hash(root/path) != expected:
            raise ValueError(f"Stage artifact changed: {path}")
    return record


class Stage:
    def __init__(self, root, name):
        from structure_net_resources.client import require_gpu_lease
        self.lease = require_gpu_lease([0])  # Before any CUDA query or model load.
        self.check_lease = require_gpu_lease
        self.root, self.name = root, name
        self.plan = verify(root, model=True)
        self.path = root/name
        self.path.mkdir(exist_ok=False)
        self.started, self.counts = time.perf_counter(), Counter()
        write_json(self.path/"lease.json", {k: self.lease[k] for k in ("id", "holder", "state", "ready", "allocation", "end")})
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Reserved CUDA BF16 device required")
        torch.set_num_threads(4)
        torch.manual_seed(CONFIG["seed"])
        torch.cuda.reset_peak_memory_stats()
        self.model, self.tokenizer = load_small_qwen(self.plan["model"]["path"], device="cuda:0", dtype=torch.bfloat16)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.tasks = json.loads((root/"tasks.json").read_text())

    def event(self, label, row):
        self.counts[label] += 1
        with (self.path/"events.jsonl").open("a") as stream:
            stream.write(json.dumps({"stage": label, "row": row, "elapsed_seconds": time.perf_counter()-self.started}, allow_nan=False)+"\n")
        if self.counts[label] == 1 or self.counts[label] % 16 == 0:
            self.check_lease([0])
            print(json.dumps({"stage": label, "completed": self.counts[label], "elapsed_seconds": round(time.perf_counter()-self.started, 2)}), flush=True)

    def clean(self, pairs, sides, label):
        features = {side: {key: [] for key in CONFIG["sites"]} for side in sides}
        records = []
        for pair in pairs:
            if pair_alignment(self.tokenizer, pair["source"], pair["donor"]) != pair["alignment"]:
                raise ValueError("Token alignment changed")
            row = {}
            for side in sides:
                case = pair[side]
                output = forward_record(self.model, self.tokenizer, _prompt(self.tokenizer, case), layers=[int(k) for k in CONFIG["sites"]])
                row[side] = natural_row(output, case)
                for key, scope in CONFIG["sites"].items():
                    features[side][key].append(flatten_state(output["states"][int(key)], scope))
                self.event(label+"_"+side, row[side])
            records.append(row)
        features = {side: {key: torch.stack(values) for key, values in layers.items()} for side, layers in features.items()}
        save_tensor(self.path/f"{label}.pt", features)
        write_json(self.path/f"{label}.json", records)
        return features, records

    def finish(self, classification, **extra):
        self.check_lease([0])
        verify(self.root)
        if torch.cuda.max_memory_reserved() > sum(self.lease["allocation"].values())*1024**2:
            raise RuntimeError("Exceeded the reservation memory budget")
        result = {"study_id": STUDY_ID, "status": "completed", "classification": classification,
                  "scientific_fingerprint": self.plan["scientific_fingerprint"], "counts": dict(self.counts),
                  "lease_id": self.lease["id"], "release_owner": "inference-ctl on process exit",
                  "elapsed_seconds": time.perf_counter()-self.started, "model_parameters": self.parameter_count,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
                  "artifacts": {str(p.relative_to(self.root)): file_hash(p) for p in sorted(self.path.rglob("*")) if p.is_file()}, **extra}
        write_json(self.root/f"{self.name}_manifest.json", result)
        return {k: v for k, v in result.items() if k not in ("artifacts",)}


def training(root):
    stage = Stage(root, "training")
    smoke = []
    for pair in stage.tasks["pairs"]["shakedown"]:
        outputs = [forward_record(stage.model, stage.tokenizer, _prompt(stage.tokenizer, pair[side]), layers=(24, 29, 35))
                   for side in ("source", "donor")]
        boundary = pair["alignment"]["question_start"]
        for index, side in enumerate(("source", "donor")):
            patch, _ = make_donor_patch(outputs[index]["states"][35][boundary:], boundary, "final")
            restored = forward_record(stage.model, stage.tokenizer, _prompt(stage.tokenizer, pair["source"]), layer=35, patch=patch)
            if not compare_logits(restored["logits"], outputs[index]["logits"])["exact"]:
                raise ValueError("Terminal identity/donor control failed")
        probe = task_dictionary(stage.model, stage.tokenizer, [_prompt(stage.tokenizer, pair["source"])],
                                stage.plan["token_ids"][:4], {24: "question", 29: "final"}, batch=2)
        save_tensor(stage.path/f"shakedown_{len(smoke)}.pt", probe)
        loaded = torch.load(stage.path/f"shakedown_{len(smoke)}.pt", map_location="cpu", weights_only=True)
        if not all(torch.equal(probe[l], loaded[l]) for l in probe):
            raise ValueError("VJP save/reload failed")
        smoke.append({"source": natural_row(outputs[0], pair["source"]), "donor": natural_row(outputs[1], pair["donor"]),
                      "identity_and_terminal_logits_exact": True, "vjp_roundtrip_exact": True})
        stage.event("shakedown", smoke[-1])
        if torch.cuda.max_memory_reserved() > sum(stage.lease["allocation"].values())*1024**2:
            raise RuntimeError("Representative memory shakedown exceeds reservation")
    write_json(stage.path/"shakedown.json", {"scope": "systems_lifecycle_only_not_quality_evidence", "rows": smoke})
    for split in ("train", "development"):
        stage.clean(stage.tasks["pairs"][split], ("source", "donor"), split)
    calibration_ids = set(stage.tasks["calibration_ids"])
    prompts = [_prompt(stage.tokenizer, p["source"]) for p in stage.tasks["pairs"]["train"] if p["source"]["case_id"] in calibration_ids]
    dictionary = task_dictionary(stage.model, stage.tokenizer, prompts, stage.plan["token_ids"],
                                  {int(k): v for k, v in CONFIG["sites"].items()}, batch=CONFIG["vjp_batch"],
                                  progress=lambda row: stage.event("dictionary", row))
    save_tensor(stage.path/"dictionary.pt", dictionary)
    write_json(stage.path/"dictionary.json", {"formula": "mean_training_prompts W_U[token] @ d(h35_last_pre_norm)/d(H_layer_fixed_positions)",
                "source_positions_preserved_separately": True, "final_norm_included": False,
                "global_J_space_replication": False, "calibration_count": len(prompts), "token_ids": stage.plan["token_ids"]})
    return stage.finish("training_capture_completed")


def evaluation(root):
    checked_manifest(root, "training")
    fitted = json.loads((root/"fit.json").read_text())
    if fitted["fit_sha256"] != file_hash(root/"fit.pt") or fitted["training_manifest_sha256"] != file_hash(root/"training_manifest.json"):
        raise ValueError("Training/fitted model identity mismatch")
    fit_hash = file_hash(root/"fit.pt")
    fits = torch.load(root/"fit.pt", map_location="cpu", weights_only=True)["components"]
    stage = Stage(root, "evaluation")
    pairs = stage.tasks["pairs"]["evaluation"]
    features, natural = stage.clean(pairs, ("source", "donor", "answer_donor"), "natural")
    gate = competence([r["source"] for r in natural], [r["donor"] for r in natural])
    answer_eligible = [r["source"]["original_correct"] and r["answer_donor"]["original_correct"] for r in natural]
    if not gate["passed"]:
        return stage.finish("clean_competence_failed", competence=gate, answer_eligible=answer_eligible)
    dense, edits = [], []
    for key, scope in CONFIG["sites"].items():
        for donor_type, arms in (("donor", CONFIG["main_arms"]), ("answer_donor", CONFIG["answer_only_arms"])):
            for arm in arms:
                for index, pair in enumerate(pairs):
                    patch, diagnostic = make_patch(features[donor_type][key][index], fits[key], scope, arm)
                    output = forward_record(stage.model, stage.tokenizer, _prompt(stage.tokenizer, pair["source"]), layer=int(key), patch=patch)
                    row = intervention_record(output["generation"], pair["source"], int(key), scope, arm, diagnostic)
                    row["donor_type"] = donor_type
                    stage.event("edits", row)
                    edits.append(row)
                if donor_type == "donor" and arm == "dense":
                    rows = [r for r, yes in zip(edits[-len(pairs):], gate["eligible"]) if yes]
                    score = success_metrics(rows)
                    dense.append({"layer": int(key), **score})
                    if score["target_success"] < CONFIG["minimum_donor_success"]:
                        write_json(stage.path/"edits.json", edits)
                        return stage.finish("dense_positive_control_failed", competence=gate, dense=dense, answer_eligible=answer_eligible)
    if file_hash(root/"fit.pt") != fit_hash:
        raise ValueError("Fitted components changed during evaluation")
    write_json(stage.path/"edits.json", edits)
    return stage.finish("evaluation_capture_completed", competence=gate, dense=dense, answer_eligible=answer_eligible,
                        fit_sha256=fit_hash, fit_selection_sha256=file_hash(root/"fit.json"))


def analyze(root):
    cpu_only()
    from .jspace_country_probe import score_predictions, paired_difference
    from .jspace_active_heads import target_labels
    import numpy as np
    plan = verify(root)
    manifest = checked_manifest(root, "evaluation")
    fitted = json.loads((root/"fit.json").read_text())
    if file_hash(root/"fit.pt") != fitted["fit_sha256"]:
        raise ValueError("Head artifact changed")
    models = torch.load(root/"fit.pt", map_location="cpu", weights_only=True)
    features = torch.load(root/"evaluation/natural.pt", map_location="cpu", weights_only=True)
    pairs = json.loads((root/"tasks.json").read_text())["pairs"]["evaluation"]
    labels = target_labels(pairs, "source", "answer")
    natural = json.loads((root/"evaluation/natural.json").read_text())
    heads, predictions = {}, {}
    for key in CONFIG["sites"]:
        heads[key], predictions[key] = {}, {}
        for name, model in models["heads"][key].items():
            scores = head_scores(model, features["source"][key].float()).detach().numpy()
            heads[key][name] = {**score_predictions(scores, labels), "numeric_tensor_bytes": tensor_bytes(model)}
            predictions[key][name] = scores.argmax(1).tolist()
    primary_heads = paired_difference(np.array(predictions["29"]["vanishing"]), np.array(predictions["29"]["mlp"]), labels)
    candidate = fitted["benchmark_candidate"]
    candidate_score = heads[str(candidate["layer"])][candidate["head"]]
    teacher_correct = sum(row["source"]["original_correct"] for row in natural)
    quality_gate = candidate_score["accuracy"] >= CONFIG["head_minimum_accuracy"] and candidate_score["correct"] >= teacher_correct-CONFIG["maximum_head_extra_errors"]
    components, comparisons = {}, {}
    classification = manifest["classification"]
    if classification == "evaluation_capture_completed":
        rows = json.loads((root/"evaluation/edits.json").read_text())
        expected = len(pairs)*len(CONFIG["sites"])*(len(CONFIG["main_arms"])+len(CONFIG["answer_only_arms"]))
        if len(rows) != expected:
            raise ValueError("Incomplete intervention grid")
        for key in CONFIG["sites"]:
            groups = {}
            components[key] = {}
            for donor_type, arms, eligibility in (("donor", CONFIG["main_arms"], manifest["competence"]["eligible"]),
                                                  ("answer_donor", CONFIG["answer_only_arms"], manifest["answer_eligible"])):
                eligible_ids = {p["source"]["case_id"] for p, yes in zip(pairs, eligibility) if yes}
                if not eligible_ids:
                    raise ValueError("No eligible control pairs")
                components[key][donor_type] = {}
                for arm in arms:
                    subset = [r for r in rows if r["layer"] == int(key) and r["donor_type"] == donor_type and r["arm"] == arm]
                    if len(subset) != len(pairs) or {r["case_id"] for r in subset} != {p["source"]["case_id"] for p in pairs}:
                        raise ValueError("Missing/duplicate intervention cases")
                    group = [r for r in subset if r["case_id"] in eligible_ids]
                    groups[(donor_type, arm)] = group
                    components[key][donor_type][arm] = success_metrics(group)
            comparisons[key] = {name: paired_test(groups[("donor", a)], groups[("donor", b)]) for name, a, b in
                                (("full_vs_random", "full", "random_full"), ("J_vs_random", "J", "random_J"),
                                 ("J_vs_matched_remainder", "J", "remainder_norm_J"))}
        main, contrast = components["24"], comparisons["24"]
        causal = main["donor"]["full"]["target_success"] >= .2 and contrast["full_vs_random"]["difference"] > 0 and contrast["full_vs_random"]["two_sided_exact_p"] <= .05
        retention = main["answer_donor"]["full"]["original_retained"]/main["answer_donor"]["full"]["total"]
        specific = retention >= .8
        j_advantage = all(contrast[k]["difference"] >= .1 and contrast[k]["two_sided_exact_p"] <= .05
                          for k in ("J_vs_random", "J_vs_matched_remainder")) and main["donor"]["J"]["target_success"] >= .2
        classification = ("country_CAV_causality_not_established" if not causal else
                          "causal_direction_country_specificity_not_established" if not specific else
                          "preliminary_country_CAV_and_J_advantage" if j_advantage else "preliminary_country_CAV_J_advantage_not_established")
    result = {"study_id": STUDY_ID, "status": "analyzed", "confirmed": False, "scientific_fingerprint": plan["scientific_fingerprint"],
              "classification": classification, "components": components, "comparisons": comparisons,
              "heads": heads, "head_predictions": predictions, "primary_head_comparison_29_vanishing_minus_mlp": primary_heads,
              "teacher": {"correct": teacher_correct, "total": len(pairs), "accuracy": teacher_correct/len(pairs)},
              "benchmark_candidate": candidate, "candidate_quality_gate": quality_gate,
              "quality_gate_scope": "observed >=95% and at most one extra error versus teacher; exploratory, not noninferiority proof",
              "evaluation_manifest_sha256": file_hash(root/"evaluation_manifest.json"),
              "fit_sha256": file_hash(root/"fit.pt"), "fit_selection_sha256": file_hash(root/"fit.json"),
              "compression_benchmarked": False}
    write_json(root/"analysis.json", result)
    return {"classification": classification, "teacher": result["teacher"], "head_correct": {k: {n: v["correct"] for n, v in h.items()} for k, h in heads.items()},
            "candidate": candidate, "candidate_quality_gate": quality_gate, "components": components}


def finalize(root):
    cpu_only()
    verify(root)
    result = json.loads((root/"analysis.json").read_text())
    measured = checked_manifest(root, "benchmark") if result["candidate_quality_gate"] else None
    result.update({"status": "completed", "benchmark_status": measured["classification"] if measured else "not_run_quality_gate_failed",
                   "benchmark": measured,
                   "neural_speedup": measured["measured"]["median_task_latency_speedup"] if measured else None,
                   "compression_benchmarked": measured is not None,
                   "confirmed": False, "analysis_sha256": file_hash(root/"analysis.json")})
    write_json(root/"result.json", result)
    write_json(root/"campaign_results.json", {**result, "result_path": "result.json", "result_sha256": file_hash(root/"result.json")})
    return {"status": "completed", "classification": result["classification"], "benchmark_status": result["benchmark_status"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "training", "fit", "evaluation", "analyze", "benchmark", "finalize"))
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.stage == "fit":
        from .jspace_active_heads import fit
        operation = fit
    elif args.stage == "benchmark":
        from .jspace_active_benchmark import benchmark
        operation = benchmark
    else:
        operation = globals()[args.stage]
    try:
        value = operation(args.output)
    except BaseException as exc:
        path = args.output/f"failure_{args.stage}.json"
        if args.output.exists() and not path.exists() and not isinstance(exc, FileExistsError):
            write_json(path, {"stage": args.stage, "status": "invalid_stage", "error": str(exc), "type": type(exc).__name__})
        raise
    print(json.dumps(value, allow_nan=False))
