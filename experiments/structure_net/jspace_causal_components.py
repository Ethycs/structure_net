"""Prospective learned-CAV component interventions with mandatory GPU leases."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .jspace_lens import load_small_qwen
from .jspace_relation_capture import _prompt, capture_natural
from .jspace_relation_study import ROOT, digest, file_hash, save_tensor, verify_plan as verify_parent, write_json
from .jspace_relation_tasks import COUNTRIES, _make_split, _render, world_signature
from .jspace_runtime import generate_answer
from .jspace_tasks import normalize_answer

STUDY_ID = "JSPACE-CAUSAL-COMPONENTS-01"
PROTOCOL = "docs/07 - Status Reports/jspace/2026-11-13_jspace-causal-components-plan.md"
PARENT = ROOT/"data/experiments/jspace_relation_propagation/20261113_neural_v1"
EXPANSION = ROOT/"data/experiments/jspace_residual_expansion/20261113_v1"
ARMS = ("full", "J", "remainder", "random_J", "random_full", "remainder_norm_J")
CONFIG = {"seed": 151, "layers": [18, 24], "counts": {"train": 256, "development": 32, "evaluation": 96, "shakedown": 16},
          "selection_order": [[24, "final"], [24, "query_suffix"], [18, "final"], [18, "query_suffix"]],
          "arms": list(ARMS), "sparse_budget": 25, "random_seed_base": 211, "max_new_tokens": 8,
          "minimum_clean_accuracy": .9, "minimum_joint_fraction": .8, "minimum_donor_success": .8,
          "minimum_causal_success": .2, "minimum_J_advantage": .1, "paired_p_threshold": .05,
          "dtype": "bfloat16", "hidden_width": 2048, "model_depth": 36}


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU stage requires CUDA_VISIBLE_DEVICES='' before Python")


def render_case(case, *, query=None):
    row = copy.deepcopy(case)
    if query is not None:
        row["case_id"] += ":donor"
        row["query_person"] = query
        row["intermediate"], row["swap_to"] = case["swap_to"], case["intermediate"]
        row["answer"], row["swap_answer"] = case["swap_answer"], case["answer"]
    people = [(name, row["person_to_country"][name]) for name in row["person_display_order"]]
    colours = [(country, row["country_to_colour"][country]) for country in row["country_display_order"]]
    row["prompt"], row["question"] = _render("evaluation_json", people, colours, row["query_person"])
    row["template_id"] = "causal_shared_json_v1"
    return row


def build_tasks(parent_tasks, counts=None):
    counts = CONFIG["counts"] if counts is None else counts
    excluded = {case["world_id"] for rows in parent_tasks["splits"].values() for case in rows}
    seen = set(excluded)
    splits = {}
    for split, count in counts.items():
        seed = CONFIG["seed"] + (1_000_003 if split == "shakedown" else 0)
        splits[split] = [render_case(case) for case in _make_split(split, count, seed, seen)]
    donors = {split: [render_case(case, query=next(name for name, country in case["person_to_country"].items()
                                                if country == case["swap_to"])) for case in splits[split]]
              for split in ("development", "evaluation")}
    dataset = {"study_id": STUDY_ID, "schema_version": 1, "splits": splits, "donors": donors,
               "excluded_parent_world_count": len(excluded), "same_template_across_splits": True,
               "independent_unit": "source_world_with_paired_donor_question"}
    validate_tasks(dataset, excluded)
    return dataset


def validate_tasks(dataset, excluded):
    seen, names = set(excluded), {}
    for split, rows in dataset["splits"].items():
        names[split] = set()
        for case in rows:
            world = world_signature(case["person_to_country"], case["country_to_colour"])
            if world != case["world_id"] or world in seen:
                raise ValueError("Repeated, excluded or invalid named world")
            seen.add(world)
            names[split].update(case["person_to_country"])
            if (case["person_to_country"][case["query_person"]] != case["intermediate"]
                    or case["country_to_colour"][case["intermediate"]] != case["answer"]
                    or case["country_to_colour"][case["swap_to"]] != case["swap_answer"]
                    or case["answer"] == case["swap_answer"]):
                raise ValueError("Invalid composed answer")
        cells = Counter((row["intermediate"], row["answer"]) for row in rows)
        if rows and (len(rows) % 16 or set(cells.values()) != {len(rows)//16} or len(cells) != 16):
            raise ValueError("Expected balanced country/answer cells")
    for i, first in enumerate(names):
        for second in list(names)[i+1:]:
            if names[first] & names[second]:
                raise ValueError("Names overlap across splits")
    for split, rows in dataset["donors"].items():
        if len(rows) != len(dataset["splits"][split]):
            raise ValueError("Missing paired donor")
        for source, donor in zip(dataset["splits"][split], rows):
            if (source["world_id"] != donor["world_id"] or source["person_to_country"] != donor["person_to_country"]
                    or source["country_to_colour"] != donor["country_to_colour"]
                    or donor["answer"] != source["swap_answer"]
                    or donor["person_to_country"][donor["query_person"]] != source["swap_to"]):
                raise ValueError("Donor is not the declared same-world counterfactual")


def source_paths():
    names = [PROTOCOL, "pyproject.toml", "pixi.lock", "tests/structure_net/test_jspace_causal_components.py"]
    names += [f"experiments/structure_net/jspace_{name}.py" for name in (
        "causal_components", "lens", "tasks", "runtime", "relation_tasks", "relation_capture",
        "relation_study", "residual_expansion")]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/"src/structure_net_resources").glob("*.py"))]
    return [ROOT/name for name in names]


def prepare(root):
    cpu_only()
    if root.exists():
        raise FileExistsError("Use a fresh append-only root")
    parent = verify_parent(PARENT, verify_model=True)
    parent_tasks = json.loads((PARENT/"tasks.json").read_text())
    tasks = build_tasks(parent_tasks)
    expansion = json.loads((EXPANSION/"result.json").read_text())
    paths = [PARENT/"plan.json", PARENT/"tasks.json", EXPANSION/"plan.json", EXPANSION/"result.json"]
    paths += [EXPANSION/f"layer_{layer}_residuals.pt" for layer in CONFIG["layers"]]
    for path in paths[-2:]:
        if file_hash(path) != expansion["artifacts"][path.name]:
            raise ValueError("Parent dictionary changed")
    plan = {"study_id": STUDY_ID, "schema_version": 1, "config": CONFIG, "model": parent["model"],
            "parent_fingerprint": parent["scientific_fingerprint"], "dataset_digest": digest(tasks),
            "concept_token_ids": parent["concept_token_ids"],
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "conformance": "PROSPECTIVE_SINGLE_CHECKPOINT_PILOT",
            "input_sha256": {str(p): file_hash(p) for p in paths},
            "producer_sha256": {str(p): file_hash(p) for p in source_paths()}}
    plan["scientific_fingerprint"] = digest(plan)
    root.mkdir(parents=True)
    for path in source_paths():
        target = root/"producer"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(root/"tasks.json", tasks)
    write_json(root/"plan.json", plan)
    return {"status": "prepared", "scientific_fingerprint": plan["scientific_fingerprint"]}


def verify(root, *, model=False):
    plan = json.loads((root/"plan.json").read_text())
    if (plan["study_id"] != STUDY_ID or plan["config"] != CONFIG
            or digest({k: v for k, v in plan.items() if k != "scientific_fingerprint"}) != plan["scientific_fingerprint"]):
        raise ValueError("Study or plan fingerprint mismatch")
    for section in ("producer_sha256", "input_sha256"):
        for path, expected in plan[section].items():
            if file_hash(path) != expected:
                raise ValueError(f"Frozen input/producer changed: {path}")
            if section == "producer_sha256" and file_hash(root/"producer"/Path(path).relative_to(ROOT)) != expected:
                raise ValueError("Archived producer changed")
    if digest(json.loads((root/"tasks.json").read_text())) != plan["dataset_digest"]:
        raise ValueError("Dataset changed after freeze")
    if model:
        for name, expected in plan["model"]["files"].items():
            if file_hash(Path(plan["model"]["path"])/name) != expected:
                raise ValueError("Checkpoint changed")
    return plan


def write_capture(path, data):
    path.mkdir(exist_ok=False)
    save_tensor(path/"hidden.pt", data["hidden"])
    write_json(path/"records.json", data["records"])


def competence(source, donor):
    if not source or len(source) != len(donor):
        raise ValueError("Paired records required")
    if any(d["case_id"] != s["case_id"]+":donor" for s, d in zip(source, donor)):
        raise ValueError("Donor row alignment changed")
    eligible = [bool(s["original_correct"] and d["original_correct"]) for s, d in zip(source, donor)]
    n = len(source)
    sa, da, joint = sum(r["original_correct"] for r in source)/n, sum(r["original_correct"] for r in donor)/n, sum(eligible)/n
    return {"source_accuracy": sa, "donor_accuracy": da, "joint_fraction": joint, "eligible": eligible,
            "total": n, "eligible_count": sum(eligible), "passed": sa >= CONFIG["minimum_clean_accuracy"]
            and da >= CONFIG["minimum_clean_accuracy"] and joint >= CONFIG["minimum_joint_fraction"]}


class GPUStage:
    def __init__(self, root, name):
        from structure_net_resources.client import require_gpu_lease
        self.lease = require_gpu_lease([0])  # Before CUDA queries/model loading.
        self.plan = verify(root, model=True)
        self.root, self.path = root, root/name
        self.path.mkdir(exist_ok=False)
        self.started, self.counts = time.perf_counter(), Counter()
        self.input_hashes = {}
        self.lease_check = require_gpu_lease
        write_json(self.path/"lease.json", {k: self.lease[k] for k in ("id", "holder", "state", "ready", "allocation", "end")})
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Reserved device must support CUDA and BF16")
        torch.set_num_threads(4)
        torch.manual_seed(CONFIG["seed"])
        torch.cuda.manual_seed_all(CONFIG["seed"])
        torch.cuda.reset_peak_memory_stats()
        self.model, self.tokenizer = load_small_qwen(self.plan["model"]["path"], device="cuda:0", dtype=torch.bfloat16)
        self.tasks = json.loads((root/"tasks.json").read_text())

    def event(self, stage, row):
        self.counts[stage] += 1
        with (self.path/"events.jsonl").open("a") as stream:
            stream.write(json.dumps({"stage": stage, "elapsed_seconds": time.perf_counter()-self.started,
                                     "row": row}, allow_nan=False)+"\n")
        if self.counts[stage] == 1 or self.counts[stage] % 32 == 0:
            self.lease_check([0])
            print(json.dumps({"stage": stage, "completed": self.counts[stage],
                              "elapsed_seconds": round(time.perf_counter()-self.started, 2)}), flush=True)

    def natural(self, label, cases):
        data = capture_natural(self.model, self.tokenizer, cases, layers=CONFIG["layers"],
                               progress=lambda row: self.event(label, row))
        write_capture(self.path/label, data)
        return data

    def finish(self, classification, **extra):
        self.lease_check([0])
        for name, expected in self.input_hashes.items():
            if file_hash(self.root/name) != expected:
                raise ValueError("Stage input changed during execution")
        allocation = sum(self.lease["allocation"].values())
        if torch.cuda.max_memory_reserved() > allocation*1024**2:
            raise RuntimeError("Capture exceeded reserved GPU memory")
        result = {"study_id": STUDY_ID, "status": "completed", "classification": classification,
                  "scientific_fingerprint": self.plan["scientific_fingerprint"], "counts": dict(self.counts),
                  "lease_id": self.lease["id"], "release_owner": "inference-ctl supervisor on exit",
                  "elapsed_seconds": time.perf_counter()-self.started,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda,
                              "device": torch.cuda.get_device_name(0)},
                  "artifacts": {**{str(p.relative_to(self.root)): file_hash(p) for p in sorted(self.path.rglob("*")) if p.is_file()},
                                **self.input_hashes}, **extra}
        write_json(self.path/"manifest.json", result)
        return result


def training_capture(root):
    stage = GPUStage(root, "training")
    smoke = stage.natural("shakedown", stage.tasks["splits"]["shakedown"])
    restored = torch.load(stage.path/"shakedown/hidden.pt", map_location="cpu", weights_only=True)
    if not torch.equal(restored, smoke["hidden"]):
        raise ValueError("Shakedown checkpoint roundtrip failed")
    case = stage.tasks["splits"]["shakedown"][0]
    identity = generate_answer(stage.model, stage.tokenizer, _prompt(stage.tokenizer, case),
                               layer=24, patch=lambda h: h, max_new_tokens=8)
    if identity["token_ids"] != smoke["records"][0]["token_ids"]:
        raise ValueError("Identity hook changed generated answer")
    write_json(stage.path/"shakedown/result.json", {"classification": "systems_lifecycle_only_not_quality_evidence",
               "roundtrip_equal": True, "identity_hook_equal": True, "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
    if torch.cuda.max_memory_reserved() > sum(stage.lease["allocation"].values())*1024**2:
        raise RuntimeError("Memory pilot exceeded reservation")
    source = stage.natural("development", stage.tasks["splits"]["development"])
    donor = stage.natural("development_donor", stage.tasks["donors"]["development"])
    gate = competence(source["records"], donor["records"])
    if not gate["passed"]:
        return stage.finish("development_competence_failed", development_competence=gate)
    training = stage.natural("train", stage.tasks["splits"]["train"])
    accuracy = sum(row["original_correct"] for row in training["records"])/len(training["records"])
    return stage.finish("training_capture_completed" if accuracy >= CONFIG["minimum_clean_accuracy"] else "training_competence_failed",
                        development_competence=gate, training_accuracy=accuracy)


def verify_manifest(root, path):
    manifest = json.loads(path.read_text())
    if manifest["status"] != "completed" or manifest["scientific_fingerprint"] != verify(root)["scientific_fingerprint"]:
        raise ValueError("Incomplete/mismatched stage")
    for name, expected in manifest["artifacts"].items():
        if not (root/name).resolve().is_relative_to(root.resolve()) or file_hash(root/name) != expected:
            raise ValueError("Stage artifact changed")
    return manifest


def fit_components(root):
    cpu_only()
    from .jspace_residual_expansion import nonnegative_pursuit
    plan = verify(root)
    manifest = verify_manifest(root, root/"training/manifest.json")
    if manifest["classification"] != "training_capture_completed":
        raise ValueError("Training competence gate failed; analyze the stopped run")
    data = torch.load(root/"training/train/hidden.pt", map_location="cpu", weights_only=True).double()
    records = json.loads((root/"training/train/records.json").read_text())
    y = torch.tensor([COUNTRIES.index(row["intermediate"]) for row in records])
    fits, diagnostics = {}, {}
    for index, layer in enumerate(CONFIG["layers"]):
        hidden = data[:, index]
        mean = hidden.mean(0)
        full = torch.stack([hidden[y == i].mean(0)-mean for i in range(4)])
        dictionary_file = torch.load(EXPANSION/f"layer_{layer}_residuals.pt", map_location="cpu", weights_only=True)
        dictionary = dictionary_file["dictionary"].double()
        initial = [dictionary_file["token_ids"].index(token) for token in plan["concept_token_ids"]]
        sparse = nonnegative_pursuit(full.numpy(), dictionary.numpy(), initial, budgets=(4, 25, 94))
        jpart = torch.from_numpy(sparse["coefficients"][25]) @ dictionary
        residual = full-jpart
        if not torch.isfinite(full).all() or not torch.allclose(jpart+residual, full, rtol=1e-12, atol=1e-12):
            raise ValueError("Nonfinite/invalid CAV decomposition")
        for i in range(4):
            for j in range(i+1, 4):
                if torch.linalg.matrix_rank(full[[i, j]], rtol=1e-6) != 2:
                    raise ValueError("Degenerate full CAV pair")
        generator = torch.Generator().manual_seed(CONFIG["random_seed_base"]+layer)
        fits[str(layer)] = {"mean": mean, "full": full, "J": jpart, "remainder": residual,
                           "permutation": torch.randperm(hidden.shape[-1], generator=generator),
                           "sign": 2*torch.randint(2, (hidden.shape[-1],), generator=generator)-1,
                           "coefficients": torch.from_numpy(sparse["coefficients"][25]), "dictionary": dictionary}
        diagnostics[str(layer)] = {"sparse_metrics": sparse["metrics"],
            "reconstruction_max_absolute_error": float((full-jpart-residual).abs().max()),
            "J_energy_fraction_per_country": (jpart.square().sum(1)/full.square().sum(1)).tolist(),
            "full_cav_norm": full.norm(dim=1).tolist(),
            "selected_tokens": {country: [dictionary_file["token_strings"][i] for i in np.flatnonzero(sparse["coefficients"][25][c] > 1e-10)]
                                for c, country in enumerate(COUNTRIES)}}
    save_tensor(root/"fit.pt", fits)
    write_json(root/"fit.json", {"study_id": STUDY_ID, "status": "completed", "scientific_fingerprint": plan["scientific_fingerprint"],
               "fit_sha256": file_hash(root/"fit.pt"), "training_manifest_sha256": file_hash(root/"training/manifest.json"),
               "diagnostics": diagnostics, "runtime": {"numpy": np.__version__, "torch": str(torch.__version__)}})
    return {"status": "fitted", "diagnostics": diagnostics}


def query_start(tokenizer, prompt, question):
    start = prompt.rfind(question)
    if start < 0:
        raise ValueError("Question not found in formatted prompt")
    encoded = tokenizer(prompt, return_offsets_mapping=True)
    positions = [i for i, (_, end) in enumerate(encoded["offset_mapping"]) if end > start]
    if not positions:
        raise ValueError("No question tokens found")
    return positions[0]


def edit_parts(work, fitted, source, target):
    pair = fitted["full"][[source, target]]
    inverse = torch.linalg.pinv(pair.double(), rtol=1e-6).to(device=work.device, dtype=torch.float32)
    coefficients = (work-fitted["mean"].to(work.device).float()) @ inverse
    swapped = coefficients.flip(-1)-coefficients
    return {arm: swapped @ fitted[arm][[source, target]].to(work.device).float() for arm in ("full", "J", "remainder")}


def make_patch(fitted, source, target, *, arm, scope, start=0):
    if arm not in ARMS or scope not in ("final", "query_suffix"):
        raise ValueError("Unknown arm or scope")
    pair = fitted["full"][[source, target]].double()
    inverse = torch.linalg.pinv(pair, rtol=1e-6).float()
    diagnostics = {}

    def patch(hidden):
        first = hidden.shape[1]-1 if scope == "final" else start
        if not 0 <= first < hidden.shape[1]:
            raise ValueError("Invalid edit boundary")
        work = hidden[:, first:, :].float()
        coefficients = (work-fitted["mean"].to(work.device).float()) @ inverse.to(work.device)
        swapped = coefficients.flip(-1)-coefficients
        parts = {name: swapped @ fitted[name][[source, target]].to(work.device).float()
                 for name in ("full", "J", "remainder")}
        if arm.startswith("random_"):
            delta = parts[arm.removeprefix("random_")][..., fitted["permutation"].to(work.device)]*fitted["sign"].to(work.device)
        elif arm == "remainder_norm_J":
            jnorm, rnorm = parts["J"].norm(dim=-1, keepdim=True), parts["remainder"].norm(dim=-1, keepdim=True)
            if bool(((jnorm > 1e-8) & (rnorm <= 1e-12)).any()):
                raise ValueError("Nonzero J edit cannot be matched by zero remainder")
            delta = parts["remainder"]*(jnorm/rnorm.clamp_min(1e-12))
        else:
            delta = parts[arm]
        changed = (work+delta).to(hidden.dtype)
        if not bool(torch.isfinite(changed).all()):
            raise FloatingPointError("Nonfinite edited activations")
        actual = changed.float()-work
        diagnostics.update({"positions": work.shape[1], "start": first, "requested_l2": float(delta.norm()),
                            "actual_l2": float(actual.norm()), "J_reference_l2": float(parts["J"].norm()),
                            "full_reference_l2": float(parts["full"].norm()),
                            "relative_actual_l2": float(actual.norm()/work.norm().clamp_min(1e-12)),
                            "decomposition_max_absolute_error": float((parts["full"]-parts["J"]-parts["remainder"]).abs().max())})
        updated = hidden.clone()
        updated[:, first:, :] = changed
        return updated
    return patch, diagnostics


def intervention_record(result, case, layer, scope, arm, diagnostics):
    answer = normalize_answer(result["answer"]) if result["answer"] else None
    return {"case_id": case["case_id"], "layer": layer, "scope": scope, "arm": arm,
            "expected_answer": case["answer"], "expected_target": case["swap_answer"],
            "original_correct": answer == normalize_answer(case["answer"]),
            "target_correct": answer == normalize_answer(case["swap_answer"]),
            "diagnostics": diagnostics, **{k: v for k, v in result.items() if k != "observed"}}


def run_edits(stage, cases, fits, layer, scope, arms, label):
    rows = []
    for case in cases:
        prompt = _prompt(stage.tokenizer, case)
        start = query_start(stage.tokenizer, prompt, case["question"]) if scope == "query_suffix" else 0
        for arm in arms:
            patch, diagnostics = make_patch(fits[str(layer)], COUNTRIES.index(case["intermediate"]), COUNTRIES.index(case["swap_to"]),
                                             arm=arm, scope=scope, start=start)
            result = generate_answer(stage.model, stage.tokenizer, prompt, layer=layer, patch=patch, max_new_tokens=8)
            if not diagnostics:
                raise RuntimeError("Edit hook did not run")
            row = intervention_record(result, case, layer, scope, arm, diagnostics)
            stage.event(label, row)
            rows.append(row)
    return rows


def select_configuration(rows, eligible_ids):
    if not eligible_ids:
        raise ValueError("No eligible development cases")
    scores = []
    for order, (layer, scope) in enumerate(CONFIG["selection_order"]):
        arm_scores = {}
        for arm in ("full", "random_full"):
            chosen = [r for r in rows if r["layer"] == layer and r["scope"] == scope and r["arm"] == arm and r["case_id"] in eligible_ids]
            if len(chosen) != len(eligible_ids) or {r["case_id"] for r in chosen} != set(eligible_ids):
                raise ValueError("Incomplete or duplicated development configuration")
            arm_scores[arm] = sum(r["target_correct"] for r in chosen)/len(chosen)
        scores.append({"layer": layer, "scope": scope, "order": order, **arm_scores,
                       "advantage": arm_scores["full"]-arm_scores["random_full"]})
    winner = max(scores, key=lambda row: (row["advantage"], row["full"], -row["order"]))
    return {"selected": winner, "configurations": scores, "frozen_before_evaluation_calls": True}


def causal_capture(root):
    verify_manifest(root, root/"training/manifest.json")
    fitting = json.loads((root/"fit.json").read_text())
    if (file_hash(root/"fit.pt") != fitting["fit_sha256"]
            or file_hash(root/"training/manifest.json") != fitting["training_manifest_sha256"]
            or fitting["scientific_fingerprint"] != verify(root)["scientific_fingerprint"]):
        raise ValueError("Frozen component fit changed")
    fits = torch.load(root/"fit.pt", map_location="cpu", weights_only=True)
    stage = GPUStage(root, "interventions")
    stage.input_hashes = {name: file_hash(root/name) for name in ("fit.pt", "fit.json", "training/manifest.json")}
    training = json.loads((root/"training/manifest.json").read_text())
    dev_cases = stage.tasks["splits"]["development"]
    eligible = {case["case_id"] for case, yes in zip(dev_cases, training["development_competence"]["eligible"]) if yes}
    rows = []
    for layer, scope in CONFIG["selection_order"]:
        rows.extend(run_edits(stage, dev_cases, fits, layer, scope, ("full", "random_full"), "development_edits"))
    write_json(stage.path/"development_edits.json", rows)
    selection = select_configuration(rows, eligible)
    write_json(stage.path/"selection.json", selection)
    layer, scope = selection["selected"]["layer"], selection["selected"]["scope"]
    cases = stage.tasks["splits"]["evaluation"]
    source = stage.natural("evaluation", cases)
    donor = stage.natural("evaluation_donor", stage.tasks["donors"]["evaluation"])
    gate = competence(source["records"], donor["records"])
    if not gate["passed"]:
        return stage.finish("evaluation_competence_failed", evaluation_competence=gate, selection=selection)
    index = CONFIG["layers"].index(layer)
    positives = []
    for row_index, case in enumerate(cases):
        donor_hidden = donor["hidden"][row_index, index]
        def patch(hidden):
            changed = hidden.clone()
            changed[:, -1, :] = donor_hidden.to(device=hidden.device, dtype=hidden.dtype)
            return changed
        result = generate_answer(stage.model, stage.tokenizer, _prompt(stage.tokenizer, case), layer=layer, patch=patch, max_new_tokens=8)
        row = intervention_record(result, case, layer, "final", "dense_donor", {})
        stage.event("dense_donor", row)
        positives.append(row)
    write_json(stage.path/"dense_donor.json", positives)
    positive_rate = sum(row["target_correct"] for row, yes in zip(positives, gate["eligible"]) if yes)/gate["eligible_count"]
    if positive_rate < CONFIG["minimum_donor_success"]:
        return stage.finish("dense_donor_control_failed", evaluation_competence=gate, selection=selection, dense_donor_success=positive_rate)
    rows = run_edits(stage, cases, fits, layer, scope, ARMS, "evaluation_edits")
    write_json(stage.path/"evaluation_edits.json", rows)
    return stage.finish("causal_capture_completed", evaluation_competence=gate, selection=selection, dense_donor_success=positive_rate)


def success_metrics(rows):
    n = len(rows)
    if not n:
        raise ValueError("No rows to score")
    correct = sum(r["target_correct"] for r in rows)
    p, z = correct/n, 1.959963984540054
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {"target_correct": correct, "total": n, "target_success": p,
            "wilson_95": [center-half, center+half], "original_retained": sum(r["original_correct"] for r in rows),
            "other_output": sum(not r["original_correct"] and not r["target_correct"] for r in rows)}


def paired_test(left, right):
    if len({r["case_id"] for r in left}) != len(left) or len({r["case_id"] for r in right}) != len(right):
        raise ValueError("Duplicate paired case")
    left = {r["case_id"]: bool(r["target_correct"]) for r in left}
    right = {r["case_id"]: bool(r["target_correct"]) for r in right}
    if not left or left.keys() != right.keys():
        raise ValueError("Paired test requires matching case sets")
    wins = sum(left[key] and not right[key] for key in left)
    losses = sum(right[key] and not left[key] for key in left)
    discordant = wins+losses
    p = min(1., 2*sum(math.comb(discordant, i) for i in range(min(wins, losses)+1))/2**discordant) if discordant else 1.
    return {"left_only_success": wins, "right_only_success": losses,
            "difference": (wins-losses)/len(left), "two_sided_exact_p": p, "total_pairs": len(left)}


def classify(metrics, comparisons):
    full = metrics["full"]["target_success"] >= CONFIG["minimum_causal_success"] and comparisons["full_vs_random"]["difference"] > 0 and comparisons["full_vs_random"]["two_sided_exact_p"] <= CONFIG["paired_p_threshold"]
    jgate = metrics["J"]["target_success"] >= CONFIG["minimum_causal_success"] and all(
        comparisons[key]["difference"] >= CONFIG["minimum_J_advantage"] and comparisons[key]["two_sided_exact_p"] <= CONFIG["paired_p_threshold"]
        for key in ("J_vs_random", "J_vs_matched_remainder"))
    classification = ("learned_CAV_causality_not_established" if not full else
                      "preliminary_J_component_causal_advantage" if jgate else "full_CAV_causal_J_advantage_not_established")
    return {"full_causal_gate": full, "conditional_J_gate": bool(full and jgate), "classification": classification, "confirmed": False}


def analyze(root):
    cpu_only()
    plan = verify(root)
    training = verify_manifest(root, root/"training/manifest.json")
    result = {"schema_version": 1, "study_id": STUDY_ID, "status": "completed", "confirmed": False,
              "conformance": plan["conformance"], "scientific_fingerprint": plan["scientific_fingerprint"], "config": CONFIG,
              "training": training, "classification": training["classification"],
              "accuracy_scope": "model_counterfactual_target_answer_on_preselected_clean_correct_pairs",
              "primary_metric": None, "primary_metric_status": "not_measured_gate_stopped",
              "model_checkpoint": plan["model"], "requested_runs": 1, "completed_runs": 1, "failed_runs": 0, "retries": 0}
    if (root/"interventions/manifest.json").exists():
        manifest = verify_manifest(root, root/"interventions/manifest.json")
        result.update({"interventions": manifest, "classification": manifest["classification"]})
        if manifest["classification"] == "causal_capture_completed":
            tasks = json.loads((root/"tasks.json").read_text())
            eligible = {case["case_id"] for case, yes in zip(tasks["splits"]["evaluation"], manifest["evaluation_competence"]["eligible"]) if yes}
            rows = json.loads((root/"interventions/evaluation_edits.json").read_text())
            paired = {arm: [r for r in rows if r["arm"] == arm and r["case_id"] in eligible] for arm in ARMS}
            if len(rows) != CONFIG["counts"]["evaluation"]*len(ARMS) or any(len(rs) != len(eligible) or len({r["case_id"] for r in rs}) != len(eligible) for rs in paired.values()):
                raise ValueError("Incomplete/duplicate evaluation arms")
            metrics = {arm: success_metrics(rs) for arm, rs in paired.items()}
            comparisons = {"full_vs_random": paired_test(paired["full"], paired["random_full"]),
                           "J_vs_random": paired_test(paired["J"], paired["random_J"]),
                           "J_vs_matched_remainder": paired_test(paired["J"], paired["remainder_norm_J"])}
            diagnostics = {arm: {key: float(np.median([r["diagnostics"][key] for r in rs]))
                                for key in ("actual_l2", "relative_actual_l2", "J_reference_l2", "full_reference_l2")}
                           for arm, rs in paired.items()}
            result.update({**classify(metrics, comparisons), "primary_metric": metrics["J"]["target_success"],
                           "primary_metric_status": "measured", "primary_metrics": metrics, "paired_comparisons": comparisons,
                           "all_case_metrics": {arm: success_metrics([r for r in rows if r["arm"] == arm]) for arm in ARMS},
                           "median_edit_diagnostics": diagnostics, "eligible_case_ids": sorted(eligible)})
    elif training["classification"] == "training_capture_completed":
        raise ValueError("Causal stage is still pending; do not finalize as complete")
    result["source_sha256"] = {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("manifest.json"))}
    for name in ("fit.pt", "fit.json", "tasks.json", "plan.json"):
        if (root/name).exists():
            result["source_sha256"][name] = file_hash(root/name)
    write_json(root/"result.json", result)
    write_json(root/"campaign_results.json", {**result, "result_sha256": file_hash(root/"result.json"), "result_path": "result.json"})
    return {key: result[key] for key in ("status", "classification", "primary_metric", "primary_metric_status")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "fit", "intervene", "analyze"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = {"prepare": prepare, "train": training_capture, "fit": fit_components,
                  "intervene": causal_capture, "analyze": analyze}[args.stage](args.output)
    except BaseException as exc:
        path = args.output/f"failure_{args.stage}.json"
        if args.output.exists() and not path.exists() and not isinstance(exc, FileExistsError):
            write_json(path, {"status": "invalid_stage", "stage": args.stage, "error_type": type(exc).__name__,
                             "error": str(exc), "recorded_at": datetime.now(timezone.utc).isoformat()})
        raise
    print(json.dumps({key: value for key, value in result.items() if key not in ("artifacts", "diagnostics")}))


if __name__ == "__main__":
    main()
