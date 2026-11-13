"""Reserved, frozen-model pilot of J-derived concept interventions.

Preparation is CPU-only; ``run`` refuses CUDA without a live controller lease.
See docs/07 - Status Reports/jspace/2026-11-12_jspace-cav-pilot-plan.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

from .jspace_lens import (
    LensFit, UPSTREAM_COMMIT, capture_residuals, fit_token_rows, load_small_qwen,
)
from .jspace_tasks import (
    TwoHopCase, eligible_cases, load_cases, normalize_answer, paired_difference,
    pca_basis, random_basis, score_predictions, split_cases, split_manifest,
)

HYPOTHESIS_ID = "JSPACE-CAV-PILOT-01"
SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "docs/07 - Status Reports/jspace/2026-11-12_jspace-cav-pilot-plan.md"
CACHE = Path.home() / ".cache/huggingface/hub"
MODEL_PATHS = [
    CACHE / "models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987",
    CACHE / "models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323",
]
DEFAULT_DATA = ROOT / ".pixi/jspace-jacobian-lens/data/experiments/probe-swap.json"
DEFAULT_CALIBRATION = ROOT / ".pixi/jspace-calibration.json"
UPSTREAM_DATA_SHA256 = "a0edd27ca23f7b4d0fbe90448c2ddcc7457a3d812121bf024ed12a032ff86796"
CALIBRATION_SHA256 = "6b5546932ef0062d87e046158b2bf0c54514d533124ba53ec2f0261b847e4dfc"
RANKS = [8, 16, 32, 64]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    """Exclusive strict JSON: results are append-only, including failed runs."""
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(text + "\n")
        f.flush()
        os.fsync(f.fileno())


def source_files():
    files = [PROTOCOL, "pyproject.toml", "pixi.lock"]
    files += [f"experiments/structure_net/jspace_{name}.py"
              for name in ("lens", "tasks", "runtime", "pilot")]
    files += [str(p.relative_to(ROOT)) for p in sorted(
        (ROOT / "src/structure_net_resources").glob("*.py"))]
    return files


def selected_layers(depth):
    candidates = sorted({depth // 3 - 1, depth // 2 - 1, (2 * depth) // 3 - 1})
    if not candidates or candidates[0] < 0 or candidates[-1] + 1 >= depth - 1:
        raise ValueError("model is too shallow for this pilot")
    return candidates, sorted(set(candidates + [i + 1 for i in candidates]))


def validate_calibration(path, cases):
    if file_hash(path) != CALIBRATION_SHA256:
        raise ValueError("Calibration differs from the pinned, audited row selection")
    calib = json.loads(Path(path).read_text())
    if (len(calib["rows"]) != 16 or calib["split"] != "train" or calib["seed"] != 7
            or file_hash(calib["source_path"]) != calib["source_sha256"]):
        raise ValueError("Calibration source or selection mismatch")
    prompts = {" ".join(case.prompt.split()) for case in cases}
    for row in calib["rows"]:
        if (hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]
                or len(row["text"].split()) != row["word_count"] or row["word_count"] < 80
                or " ".join(row["text"].split()) in prompts):
            raise ValueError("Calibration row failed text/eligibility/overlap validation")
    return calib


def prepare(out, *, data=DEFAULT_DATA, calibration=DEFAULT_CALIBRATION):
    from transformers import AutoTokenizer

    out = Path(out)
    if out.exists():
        raise FileExistsError(f"New artifact root required: {out}")
    if file_hash(data) != UPSTREAM_DATA_SHA256:
        raise ValueError("Tasks differ from the pinned upstream dataset")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[0], local_files_only=True)
    cases = load_cases(data)
    accepted, exclusions = eligible_cases(
        cases, lambda text: tokenizer.encode(text, add_special_tokens=False))
    splits = split_cases(accepted, seed=7, fit_count=8, development_count=12, evaluation_count=48)
    vocabulary = sorted({name for case in accepted for name in (case.intermediate, case.swap_to)})
    concept_ids = {name: tokenizer.encode(" " + name, add_special_tokens=False)[0]
                   for name in vocabulary}
    validate_calibration(calibration, cases)
    models = []
    for path in MODEL_PATHS:
        other = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if any(other.encode(" " + name, add_special_tokens=False) != [tid]
               for name, tid in concept_ids.items()):
            raise ValueError("Candidate model token inventories differ")
        files = sorted(p for p in path.iterdir() if p.is_file())
        config = json.loads((path / "config.json").read_text())
        candidates, fitted = selected_layers(config["num_hidden_layers"])
        models.append({"path": str(path), "revision": path.name,
                       "files": {p.name: file_hash(p) for p in files},
                       "layers": candidates, "fit_layers": fitted,
                       "width": config["hidden_size"]})
    sources = {name: file_hash(ROOT / name) for name in source_files()}
    plan = {
        "schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
        "conformance": "EXPLORATORY", "upstream_commit": UPSTREAM_COMMIT,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "producer_sha256": sources, "models": models,
        "data_sha256": file_hash(data), "calibration_sha256": file_hash(calibration),
        "split_manifest": split_manifest(splits), "exclusions": exclusions,
        "eligible_rows": len(accepted), "eligible_pairs": len({c.pair_group for c in accepted}),
        "concept_ids": concept_ids, "token_ids": sorted(set(concept_ids.values())),
        "seed": 7, "ranks": RANKS, "max_new_tokens": 8, "dtype": "float16",
        "lens": {"dim_batch": 4, "max_seq_len": 64, "skip_first": 16},
        "gates": {"development_parent_correct": 8, "evaluation_parent_correct": 24,
                  "answer_retention": .95, "accuracy_loss_max": .05,
                  "next_concept_relative_l2_p95_max": .05,
                  "retained_coordinate_relative_error_max": .005},
        "not_claimed": ["full_J_space", "paper_replication", "global_closure", "compression_speedup"],
    }
    plan["scientific_fingerprint"] = digest(plan)
    out.mkdir(parents=True)
    shutil.copy2(data, out / "tasks.json")
    shutil.copy2(calibration, out / "calibration.json")
    for name in sources:
        target = out / "producer" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    write_json(out / "plan.json", plan)
    return {"status": "prepared", "root": str(out), "fingerprint": plan["scientific_fingerprint"],
            "splits": plan["split_manifest"]["counts"], "dictionary_rows": len(plan["token_ids"])}


def verify_plan(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    body = {k: v for k, v in plan.items() if k != "scientific_fingerprint"}
    if digest(body) != plan["scientific_fingerprint"]:
        raise ValueError("Plan fingerprint mismatch")
    for name, expected in plan["producer_sha256"].items():
        if file_hash(ROOT / name) != expected or file_hash(root / "producer" / name) != expected:
            raise ValueError(f"Producer changed after freeze: {name}; prepare a new root")
    for name, key in (("tasks.json", "data_sha256"), ("calibration.json", "calibration_sha256")):
        if file_hash(root / name) != plan[key]:
            raise ValueError(f"Frozen data changed: {name}")
    for model in plan["models"]:
        for name, expected in model["files"].items():
            if file_hash(Path(model["path"]) / name) != expected:
                raise ValueError(f"Model file changed: {name}")
    return plan


def case_seed(case_id, layer=0):
    return int(digest([7, case_id, layer])[:8], 16)


def normalized(rows):
    rows = rows.float()
    norms = rows.norm(dim=-1, keepdim=True)
    if bool((norms <= 1e-12).any()) or not bool(torch.isfinite(rows).all()):
        raise ValueError("Zero or nonfinite concept row")
    return rows / norms


def factor_bank(rows):
    rows = normalized(rows)
    _, s, vh = torch.linalg.svd(rows, full_matrices=False)
    rank = int((s > 1e-6 * s[0]).sum())
    return rows, vh[:rank], s, rows @ vh[:rank].T


def matched_delta(delta, reference):
    norm = delta.norm(dim=-1, keepdim=True)
    scale = reference.norm(dim=-1, keepdim=True)
    if bool(((norm < 1e-12) & (scale > 1e-8)).any()):
        raise ValueError("Degenerate norm-matching control")
    return delta * (scale / norm.clamp_min(1e-12))


def swap_delta(hidden, pair):
    work = hidden.float()
    pair = normalized(pair).to(work.device)
    if int(torch.linalg.matrix_rank(pair, rtol=1e-6)) != 2:
        raise ValueError("Dependent concept pair")
    coordinates = work @ torch.linalg.pinv(pair, rtol=1e-6)
    return (coordinates.flip(-1) - coordinates) @ pair


def functional_patch(pair, control_pair, arm, seed):
    def patch(hidden):
        delta = swap_delta(hidden, pair)
        if arm == "unembedding":
            delta = matched_delta(swap_delta(hidden, control_pair), delta)
        elif arm == "random":
            rng = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(hidden.shape, generator=rng, dtype=torch.float32).to(hidden.device)
            delta = matched_delta(noise, delta)
        elif arm != "jacobian":
            raise ValueError(arm)
        return (hidden.float() + delta).to(hidden.dtype)
    return patch


def retained_patch(basis, donor, *, alpha=1.0):
    """Only replace the final token's complement; collect post-cast diagnostics."""
    stats = {}

    def patch(hidden):
        q = basis.to(device=hidden.device, dtype=torch.float32)
        source = hidden[:, -1, :].float()
        target = donor.to(source).reshape_as(source)
        delta = target - source
        delta = alpha * (delta - (delta @ q.T) @ q)
        changed = (source + delta).to(hidden.dtype)
        before = source @ q.T
        after = changed.float() @ q.T
        stats.update({"retained_coordinate_relative_error": float(
            (after - before).norm() / before.norm().clamp_min(1e-8)),
            "retained_coordinate_max_absolute_error": float((after - before).abs().max()),
            "perturbation_relative_l2": float((changed.float() - source).norm()
                                               / source.norm().clamp_min(1e-8))})
        result = hidden.clone()
        result[:, -1, :] = changed
        return result

    return patch, stats


def answer_map(records):
    return {key: row["answer"] or "" for key, row in records.items()}


def correctness(cases, records):
    return [normalize_answer(records[c.case_id]["answer"] or "") == normalize_answer(c.answer)
            for c in cases]


def metric_summary(cases, baseline, records, diagnostics):
    score = score_predictions(cases, answer_map(baseline), answer_map(records), target="original")
    retained = [normalize_answer(records[c.case_id]["answer"] or "") ==
                normalize_answer(baseline[c.case_id]["answer"] or "")
                and bool(records[c.case_id]["answer"]) for c in cases]
    drift = torch.tensor([diagnostics[c.case_id]["next_concept_relative_l2"] for c in cases], dtype=torch.float64)
    errors = [diagnostics[c.case_id]["retained_coordinate_relative_error"] for c in cases]
    return {"score": score, "answer_retention": sum(retained) / len(retained),
            "retention_outcomes": retained,
            "next_concept_relative_l2_median": float(torch.quantile(drift, .5)),
            "next_concept_relative_l2_p95": float(torch.quantile(drift, .95)),
            "retained_coordinate_relative_error_max": max(errors)}


def passes_sufficiency(summary, gates):
    score = summary["score"]
    baseline = score["baseline_accuracy"]["accuracy"]
    accuracy = score["raw_accuracy"]["accuracy"]
    return bool(summary["answer_retention"] >= gates["answer_retention"]
                and baseline - accuracy <= gates["accuracy_loss_max"] + 1e-12
                and summary["next_concept_relative_l2_p95"] <= gates["next_concept_relative_l2_p95_max"]
                and summary["retained_coordinate_relative_error_max"] <= gates["retained_coordinate_relative_error_max"])


class Pilot:
    def __init__(self, root, plan, lease):
        self.root, self.plan, self.lease = Path(root), plan, lease
        self.run_root = self.root / "runs" / "pilot" / "seed_7"
        self.run_root.mkdir(parents=True, exist_ok=False)
        self.events = (self.run_root / "events.jsonl").open("x")
        self.start = time.perf_counter()
        self.split = {name: [TwoHopCase(**row) for row in rows]
                      for name, rows in plan["split_manifest"]["splits"].items()}
        self.prompts = [r["text"] for r in json.loads((self.root / "calibration.json").read_text())["rows"]]
        self.model = self.tokenizer = None
        self.baseline = {}
        self.observed = {}

    def event(self, kind, **data):
        payload = {"event": kind, "elapsed_seconds": time.perf_counter() - self.start, **data}
        self.events.write(json.dumps(payload, allow_nan=False) + "\n")
        self.events.flush()

    def generation(self, case, *, layer=None, patch=None, observe_layers=()):
        from .jspace_runtime import generate_answer
        result = generate_answer(self.model, self.tokenizer, case.prompt, layer=layer, patch=patch,
                                 observe_layers=observe_layers, max_new_tokens=self.plan["max_new_tokens"])
        observed = result.pop("observed")
        result["first_token_label_match"] = (normalize_answer(result["first_token_text"]) ==
                                              normalize_answer(case.answer))
        return result, observed

    def baseline_cases(self, split_name, *, observe_layers=()):
        records, observed = {}, {}
        for case in self.split[split_name]:
            records[case.case_id], observed[case.case_id] = self.generation(case, observe_layers=observe_layers)
            self.event("baseline", split=split_name, case_id=case.case_id, result=records[case.case_id])
        self.baseline[split_name], self.observed[split_name] = records, observed
        return records

    def choose_model(self):
        screens = []
        for index, spec in enumerate(self.plan["models"]):
            self.model, self.tokenizer = load_small_qwen(spec["path"], device="cuda:0", dtype=torch.float16)
            torch.cuda.reset_peak_memory_stats()
            if index == 0:
                smoke = fit_token_rows(self.model, self.tokenizer, self.prompts[:1],
                                       source_layers=spec["layers"][:1], token_ids=self.plan["token_ids"][:2],
                                       dim_batch=2, max_seq_len=32, skip_first=16)
                path = self.run_root / "shakedown_lens.pt"
                smoke.save(path)
                loaded = LensFit.load(path)
                if any(not torch.equal(rows, loaded.token_rows[k]) for k, rows in smoke.token_rows.items()):
                    raise RuntimeError("Lens save/reload roundtrip failed")
                self.event("shakedown", status="systems_lifecycle_only_not_quality_evidence",
                           peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
                del smoke, loaded
            records = self.baseline_cases("development", observe_layers=spec["fit_layers"])
            count = sum(correctness(self.split["development"], records))
            screen = {"model": spec["path"], "correct": count, "total": len(records), "records": records}
            screens.append(screen)
            write_json(self.run_root / f"competence_model_{index}.json", screen)
            self.event("competence_gate", model_index=index, correct=count, total=len(records))
            if count >= self.plan["gates"]["development_parent_correct"]:
                self.spec = spec
                return screens
            del self.model
            self.model = None
            self.baseline.clear()
            self.observed.clear()
            gc.collect()
            torch.cuda.empty_cache()
        return screens

    def fit(self):
        self.fit_result = fit_token_rows(self.model, self.tokenizer, self.prompts,
                                        source_layers=self.spec["fit_layers"], token_ids=self.plan["token_ids"],
                                        progress_callback=lambda p: self.event("lens_progress", **p),
                                        **self.plan["lens"])
        self.fit_result.save(self.run_root / "lens.pt")
        self.rows = {k: normalized(v) for k, v in self.fit_result.token_rows.items()}
        self.token_index = {tid: index for index, tid in enumerate(self.plan["token_ids"])}

    def functional(self, split_name, layers):
        results = {}
        for layer in layers:
            layer_result = {}
            for arm in ("jacobian", "unembedding", "random"):
                records = {}
                for case in self.split[split_name]:
                    ids = [self.plan["concept_ids"][name] for name in (case.intermediate, case.swap_to)]
                    pair = self.rows[layer][[self.token_index[i] for i in ids]]
                    control = self.model.get_output_embeddings().weight[ids].detach().float().cpu()
                    patch = functional_patch(pair, control, arm, case_seed(case.case_id, layer))
                    records[case.case_id], _ = self.generation(case, layer=layer, patch=patch)
                    self.event("functional", split=split_name, layer=layer, arm=arm,
                               case_id=case.case_id, result=records[case.case_id])
                layer_result[arm] = score_predictions(self.split[split_name], answer_map(self.baseline[split_name]),
                                                     answer_map(records), target="counterfactual")
            results[layer] = layer_result
        return results

    def build_banks(self, layer):
        bank, basis, singular, coeff = factor_bank(self.rows[layer])
        if len(basis) < max(RANKS):
            raise ValueError(f"Candidate bank rank {len(basis)} below prespecified rank64")
        samples = []
        for prompt in self.prompts:
            activations, _ = capture_residuals(self.model, self.tokenizer, prompt,
                                              source_layers=[layer], max_seq_len=64)
            samples.append(activations[layer][16:-1])
        activations = torch.cat(samples)
        self.mean = activations.mean(0)
        rank_max = len(basis)
        self.banks = {"jacobian": basis, "pca": pca_basis(activations, rank_max),
                      "random": random_basis(basis.shape[1], rank_max, seed=7)}
        self.full_rank = rank_max
        self.bank_metadata = {
            "full_rank": rank_max, "calibration_activations": len(activations),
            "truncation_relative_frobenius": {r: float((bank - (bank @ basis[:r].T) @ basis[:r]).norm()
                                                       / bank.norm()) for r in RANKS + [rank_max]},
        }
        torch.save({"semantic_bank": bank, "basis": basis, "coefficient_matrix": coeff,
                    "singular_values": singular, "controls": self.banks, "mean": self.mean,
                    "metadata": self.bank_metadata}, self.run_root / "banks.pt")
        self.baseline_cases("fit", observe_layers=[layer])

    def sufficiency(self, split_name, layer, ranks, *, include_quarter=False):
        result = {}
        modes = [("donor", 1.0), ("mean", 1.0)]
        if include_quarter:
            modes.append(("donor", .25))
        donors = self.split["fit"]
        output_rows = self.rows[layer + 1]
        for rank in sorted(set(ranks)):
            result[rank] = {}
            for name, bank in self.banks.items():
                conditions = {}
                for mode, alpha in modes:
                    records, diagnostics = {}, {}
                    for case in self.split[split_name]:
                        donor_case = donors[case_seed(case.case_id) % len(donors)]
                        donor = (self.mean if mode == "mean" else
                                 self.observed["fit"][donor_case.case_id][layer])
                        patch, stats = retained_patch(bank[:rank], donor, alpha=alpha)
                        records[case.case_id], observed = self.generation(
                            case, layer=layer, patch=patch, observe_layers=[layer, layer + 1])
                        before = self.observed[split_name][case.case_id][layer + 1] @ output_rows.T
                        after = observed[layer + 1] @ output_rows.T
                        stats["next_concept_relative_l2"] = float((after - before).norm()
                                                                   / before.norm().clamp_min(1e-8))
                        diagnostics[case.case_id] = dict(stats)
                        self.event("sufficiency", split=split_name, layer=layer, rank=rank, bank=name,
                                   mode=mode, alpha=alpha, donor_case_id=donor_case.case_id,
                                   case_id=case.case_id, diagnostics=stats, result=records[case.case_id],
                                   next_concept_before=before.tolist(), next_concept_after=after.tolist())
                    summary = metric_summary(self.split[split_name], self.baseline[split_name], records, diagnostics)
                    summary["passes"] = passes_sufficiency(summary, self.plan["gates"])
                    conditions[f"{mode}_{alpha:g}"] = summary
                result[rank][name] = conditions
                self.event("rank_finished", split=split_name, rank=rank, bank=name)
        return result

    def execute(self):
        screens = self.choose_model()
        if self.model is None:
            return {"classification": "parent_competence_insufficient", "competence_screens": screens,
                    "causal_test": "not_run", "sufficiency_test": "not_run"}
        self.fit()
        dev = self.functional("development", self.spec["layers"])

        def selection_score(layer):
            arms = dev[layer]
            counts = {name: arm["parent_correct_accuracy"]["successes"] for name, arm in arms.items()}
            return counts["jacobian"] - max(counts["unembedding"], counts["random"]), -layer

        layer = max(self.spec["layers"], key=selection_score)
        self.build_banks(layer)
        suff_dev = self.sufficiency("development", layer, RANKS + [self.full_rank])
        passing = [r for r in RANKS if all(suff_dev[r]["jacobian"][m]["passes"]
                                          for m in ("donor_1", "mean_1"))]
        selected_rank = min(passing) if passing else 64
        selection = {"layer": layer, "rank": selected_rank, "development_rank_passed": bool(passing),
                     "full_rank": self.full_rank, "model": self.spec["path"],
                     "rule": "development_only; rank64_fallback_when_no_rank_passes"}
        write_json(self.run_root / "locked_selection.json", selection)
        write_json(self.run_root / "development_results.json", {"functional": dev, "sufficiency": suff_dev})
        self.event("selection_locked", **selection)
        self.baseline_cases("evaluation", observe_layers=[layer, layer + 1])
        functional = self.functional("evaluation", [layer])[layer]
        suff = self.sufficiency("evaluation", layer, [selected_rank, self.full_rank], include_quarter=True)
        n_parent = sum(correctness(self.split["evaluation"], self.baseline["evaluation"]))
        enough_parent = n_parent >= self.plan["gates"]["evaluation_parent_correct"]
        differences = {}
        for control in ("unembedding", "random"):
            parent_ids = {row["case_id"] for row in functional["jacobian"]["cases"] if row["baseline_correct"]}
            if parent_ids:
                a = [row["correct"] for row in functional[control]["cases"] if row["case_id"] in parent_ids]
                b = [row["correct"] for row in functional["jacobian"]["cases"] if row["case_id"] in parent_ids]
                differences[control] = paired_difference(a, b)
        causal = (enough_parent and functional["jacobian"]["parent_correct_accuracy"]["accuracy"] >= .5
                  and all(differences[c]["bootstrap_ci95"][0] > 0 for c in ("unembedding", "random")))
        retained = all(suff[selected_rank]["jacobian"][m]["passes"] for m in ("donor_1", "mean_1"))
        retention_differences = {
            c: {m: paired_difference(suff[selected_rank][c][m]["retention_outcomes"],
                                     suff[selected_rank]["jacobian"][m]["retention_outcomes"])
                for m in ("donor_1", "mean_1")} for c in ("pca", "random")}
        specific = all(retention_differences[c][m]["bootstrap_ci95"][0] > 0
                       for c in ("pca", "random") for m in ("donor_1", "mean_1"))
        classification = ("underpowered_parent_accuracy" if not enough_parent else
                          "promising_local_causal_and_sufficiency_result" if causal and retained and specific else
                          "joint_gate_not_met")
        torch.save(self.observed, self.run_root / "baseline_residuals.pt")
        return {"classification": classification, "selection": selection,
                "competence_screens": screens, "evaluation_parent_correct": n_parent,
                "functional": functional, "paired_differences": differences, "sufficiency": suff,
                "retention_differences": retention_differences,
                "bank_metadata": self.bank_metadata,
                "gates": {"enough_parent_correct": enough_parent, "causal": bool(causal),
                          "local_sufficiency": retained, "j_specific_retention": specific},
                "autonomous_reduced_transition_established": False,
                "next_experiment": ("context_compressed_closure_test" if enough_parent and causal and retained and specific
                                    else "diagnose_failed_gate")}


def run(root):
    from structure_net_resources.client import require_gpu_lease

    # No CUDA query, allocation, or model loading precedes live lease validation.
    lease = require_gpu_lease([0])
    plan = verify_plan(root)
    torch.set_num_threads(4)
    torch.manual_seed(plan["seed"])
    pilot = Pilot(root, plan, lease)
    try:
        result = pilot.execute()
        result.update({"schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
                       "scientific_fingerprint": plan["scientific_fingerprint"], "status": "completed",
                       "conformance": "EXPLORATORY", "confirmed": False,
                       "environment": {"torch": str(torch.__version__), "device": torch.cuda.get_device_name(),
                                       "dtype": "float16", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                       "lease_id": lease["id"]},
                       "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                       "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                       "elapsed_seconds": time.perf_counter() - pilot.start,
                       "timing_is_benchmark": False, "not_claimed": plan["not_claimed"]})
        write_json(pilot.run_root / "result.json", result)
        write_json(Path(root) / "campaign_results.json", {
            "schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
            "scientific_fingerprint": plan["scientific_fingerprint"], "config": plan,
            "completion_counts": {"requested": 1, "completed": 1, "failed": 0, "reused": 0},
            "result": result, "artifacts": {"run": str(pilot.run_root)},
        })
        return {"status": "completed", "classification": result["classification"], "root": str(root)}
    except BaseException as exc:
        failure = {"status": "failed", "error_type": type(exc).__name__, "message": str(exc),
                   "scientific_fingerprint": plan["scientific_fingerprint"]}
        write_json(pilot.run_root / "failure.json", failure)
        if not (Path(root) / "campaign_results.json").exists():
            write_json(Path(root) / "campaign_results.json", {
                "schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
                "config": plan, "classification": "execution_failure", "failure": failure,
                "completion_counts": {"requested": 1, "completed": 0, "failed": 1, "reused": 0},
            })
        raise
    finally:
        pilot.events.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    prep.add_argument("--data", default=str(DEFAULT_DATA))
    prep.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    execute = sub.add_parser("run")
    execute.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        result = prepare(args.output, data=args.data, calibration=args.calibration)
    else:
        result = run(args.root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
