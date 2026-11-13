"""Bounded CPU algebra controls plus a separately scoped frozen neural audit."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import sympy
import torch

from . import jspace_relations as relations
from .jspace_groebner_controls import CONTROL_IDS, run_controls
from .jspace_relation_analysis import _fingerprint
from .jspace_relation_study import (
    ROOT, _load_split, digest, file_hash, project_scores, verify_capture,
    verify_plan, write_json,
)


STUDY_ID = "JSPACE-GROEBNER-CLASSIFICATION-01"
PARENT = ROOT/"data/experiments/jspace_relation_propagation/20261113_neural_v1"
DEFAULT_OUTPUT = ROOT/"data/experiments/jspace_groebner_classification/20261113_v1"
PROTOCOL = "docs/07 - Status Reports/jspace/2026-11-13_jspace-groebner-classification-plan.md"
CONFIG = {"controls": list(CONTROL_IDS), "maximum_graph_generators": 4,
          "maximum_symbolic_calls": 18, "timeout_seconds_per_call": 5,
          "rational_max_denominator": 1000, "rationalization_rms_tolerance": .01,
          "maximum_normalized_rms": .10, "maximum_observed_to_shuffle": 1/3,
          "shuffle_controls": 32, "coefficient_field": "QQ", "monomial_order": "lex",
          "source_ideal": "zero", "evaluation_reuse": "exploratory_posthoc_not_fresh_confirmation"}


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("This CPU study requires CUDA_VISIBLE_DEVICES='' explicitly")


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "sympy": sympy.__version__, "torch": torch.__version__}


def sources():
    return [str(Path(__file__).relative_to(ROOT)),
            "experiments/structure_net/jspace_groebner_controls.py",
            "experiments/structure_net/jspace_relation_propagation.py",
            "experiments/structure_net/jspace_relations.py",
            "experiments/structure_net/jspace_relation_analysis.py",
            "experiments/structure_net/jspace_relation_study.py",
            "tests/structure_net/test_jspace_groebner_classification.py",
            PROTOCOL, "pyproject.toml", "pixi.lock"]


def input_paths():
    return ["plan.json", "capture_manifest.json", "tasks.json", "analysis/frozen_models.json",
            "capture/banks.pt"] + [f"capture/{split}/{name}" for split in ("development", "evaluation")
                                    for name in ("hidden.pt", "records.json")]


def prepare(output):
    cpu_only()
    parent_plan = verify_plan(PARENT, verify_model=False)
    verify_capture(PARENT, parent_plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    archive = output/"archive"
    source_hashes = {name: file_hash(ROOT/name) for name in sources()}
    for name in source_hashes:
        target = archive/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    scientific = {"study_id": STUDY_ID, "config": CONFIG, "source_sha256": source_hashes,
                  "parent_root": str(PARENT), "parent_scientific_fingerprint": parent_plan["scientific_fingerprint"],
                  "input_sha256": {name: file_hash(PARENT/name) for name in input_paths()},
                  "versions": versions()}
    plan = {**scientific, "scientific_fingerprint": digest(scientific), "status": "frozen_before_execution",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "conformance": "EXPLORATORY", "gpu_used": False,
            "reservation": "not_required_cpu_only_cuda_hidden"}
    write_json(output/"plan.json", plan)
    return {"root": str(output), "scientific_fingerprint": plan["scientific_fingerprint"]}


def verify(output):
    output = Path(output)
    plan = json.loads((output/"plan.json").read_text())
    scientific = {name: plan[name] for name in ("study_id", "config", "source_sha256", "parent_root",
                                               "parent_scientific_fingerprint", "input_sha256", "versions")}
    if plan["study_id"] != STUDY_ID or digest(scientific) != plan["scientific_fingerprint"]:
        raise ValueError("Plan fingerprint mismatch")
    if plan["config"] != CONFIG or plan["versions"] != versions() or plan["parent_root"] != str(PARENT):
        raise ValueError("Frozen configuration, environment or parent changed")
    for name, expected in plan["source_sha256"].items():
        if file_hash(ROOT/name) != expected or file_hash(output/"archive"/name) != expected:
            raise ValueError(f"Producer/archive changed: {name}")
    for name, expected in plan["input_sha256"].items():
        if file_hash(PARENT/name) != expected:
            raise ValueError(f"Parent input changed: {name}")
    return plan


def graph_consequences(joint, source_dev, destination_dev):
    """Eliminate source variables with a zero source ideal; no neural proof."""
    if not joint["validated"] or not joint.get("groups"):
        raise ValueError("A development-frozen mixed bank is required")
    width = joint["groups"]
    source_dev = relations._matrix(source_dev, width)
    destination_dev = relations._matrix(destination_dev, len(joint["variable_names"])-width)
    if len(source_dev) != len(destination_dev):
        raise ValueError("Development source/destination rows must align")
    chosen = sorted((c for c in joint["candidates"] if c["accepted"] and c["mixed"]),
                    key=lambda c: c["dev_rms"])[:CONFIG["maximum_graph_generators"]]
    base = {"source_width": width, "source_constraints_used": 0,
            "interpretation": "destination consequences of an approximate graph; not transport of source constraints",
            "exact_neural_claim": False, "selected_ids": [c["id"] for c in chosen],
            "destination_mean": joint["mean"][width:], "destination_scale": joint["scale"][width:],
            "selection": {"max_rms": CONFIG["maximum_normalized_rms"],
                          "max_control_ratio": CONFIG["maximum_observed_to_shuffle"]}}
    if not chosen:
        return {**base, "status": "no_accepted_graph_relations", "rationalization": [],
                "generators": [], "consequences": [], "symbolic_attempted": False}
    features = relations._features(joint, np.column_stack((source_dev, destination_dev)))
    checks, generators = [], []
    for candidate in chosen:
        coefficients = np.asarray(candidate["coefficients"])
        rational = [str(Fraction(float(c)).limit_denominator(CONFIG["rational_max_denominator"]))
                    for c in coefficients]
        rounded = np.array([float(Fraction(c)) for c in rational])
        original = relations._rms(features@coefficients)
        changed = relations._rms(features@rounded)
        delta = relations._rms(features@(rounded-coefficients))
        accepted = bool(np.any(rounded[1:] != 0) and delta <= CONFIG["rationalization_rms_tolerance"]
                        and changed <= max(original*1.5, original+.01))
        checks.append({"id": candidate["id"], "original_dev_rms": original,
                       "rationalized_dev_rms": changed, "change_dev_rms": delta, "accepted": accepted})
        generators.append({"exponents": joint["exponents"], "rational_coefficients": rational})
    if not all(check["accepted"] for check in checks):
        return {**base, "status": "rationalization_rejected", "rationalization": checks,
                "generators": generators, "consequences": [], "symbolic_attempted": False}
    linear_rank = None
    if all(c["degree"] == 1 for c in chosen):
        columns = [next(i for i, powers in enumerate(joint["exponents"])
                        if powers[j] == 1 and sum(powers) == 1) for j in range(width)]
        linear_rank = int(sympy.Matrix([[sympy.Rational(g["rational_coefficients"][i]) for i in columns]
                                       for g in generators]).rank())
    result = relations._bounded_groebner(
        {"relations": generators, "width": len(joint["variable_names"]), "source_width": width},
        CONFIG["timeout_seconds_per_call"])
    consequences = []
    z = (destination_dev-np.asarray(base["destination_mean"]))/np.asarray(base["destination_scale"])
    for consequence in result["consequences"]:
        values = relations._monomials(z, consequence["exponents"])
        coefficients = np.asarray(consequence["coefficients"])
        nonconstant = np.sum(consequence["exponents"], axis=1) > 0
        normalizer = float(np.sqrt(np.mean(np.sum((values[:, nonconstant]*coefficients[nonconstant])**2, axis=1))))
        if not np.isfinite(normalizer) or normalizer <= 1e-12:
            raise ValueError("Nonfinite or degenerate destination normalizer")
        rms = relations._rms(values@coefficients)/normalizer
        consequences.append({**consequence, "normalizer_dev": normalizer, "normalized_dev_rms": rms,
                             "passes_dev_residual_gate": rms <= CONFIG["maximum_normalized_rms"]})
    return {**base, **result, "rationalization": checks, "generators": generators,
            "linear_source_rank": linear_rank, "symbolic_attempted": True,
            "consequences": consequences}


def aggregate_audit(frozen, evaluations):
    summary = {}
    for name, edges in frozen.items():
        statuses = Counter(row["graph"]["status"] for row in edges.values())
        summary[name] = {"cells": len(edges), "status_counts": dict(statuses),
                         "symbolic_attempted": sum(row["graph"]["symbolic_attempted"] for row in edges.values()),
                         "destination_polynomials": sum(len(row["graph"]["consequences"]) for row in edges.values()),
                         "passing_development_and_reused_evaluation": sum(
                             r["passes_development_and_reused_evaluation"]
                             for edge in evaluations[name].values() for r in edge["relations"])}
    successes = sum(row["passing_development_and_reused_evaluation"] for row in summary.values())
    unresolved = sum(count for row in summary.values() for status, count in row["status_counts"].items()
                     if status in ("timeout", "worker_failed", "symbolic_error", "degree_explosion"))
    return {"by_bank": summary, "unresolved_symbolic_cells": unresolved,
            "classification": ("exploratory_graph_consequences_detected" if successes else
                               "no_validated_graph_consequences_with_unresolved_cells" if unresolved else
                               "no_validated_graph_consequences_in_bounded_audit"),
            "scope": "frozen approximate joint relations; reused evaluation; not exact neural or causal proof"}


def run(output):
    cpu_only()
    torch.set_num_threads(4)
    output = Path(output)
    plan = verify(output)
    if (output/"controls.json").exists():
        raise FileExistsError("Use a new artifact root; completed or partial execution is append-only")
    started = time.perf_counter()
    controls = run_controls()
    write_json(output/"controls.json", controls)
    if not controls["all_passed"]:
        raise RuntimeError("Algebra control failed; do not run neural audit")
    parent_models = json.loads((PARENT/"analysis/frozen_models.json").read_text())
    parent_digest = _fingerprint(parent_models)
    if parent_digest != parent_models["content_sha256"]:
        raise ValueError("Parent frozen-model content hash mismatch")
    tasks = json.loads((PARENT/"tasks.json").read_text())
    banks = torch.load(PARENT/"capture/banks.pt", map_location="cpu", weights_only=True)
    hidden, _ = _load_split(PARENT, "development", tasks["splits"]["development"])
    development = project_scores(hidden, banks)
    positions = {layer: i for i, layer in enumerate(parent_models["layers"])}
    frozen = {}
    attempts = 0
    for name, bank in parent_models["banks"].items():
        frozen[name] = {}
        for key, edge in bank["edges"].items():
            attempts += 1
            if attempts > CONFIG["maximum_symbolic_calls"]:
                raise ValueError("Exceeded bounded schedule")
            source, target = positions[edge["source"]], positions[edge["destination"]]
            graph = graph_consequences(edge["joint_bank"], development[name][:, source], development[name][:, target])
            dev = relations.evaluate_propagated_relations(graph, development[name][:, target],
                                                          seed=edge["evaluation_seed"]+100000, controls=32)
            frozen[name][key] = {"source": edge["source"], "destination": edge["destination"],
                                 "graph": graph, "development": dev,
                                 "dev_accepted": [r["polynomial"] for r in dev["relations"]
                                                  if r["passes_frozen_test_gate"] and r["passes_dev_residual_gate"]]}
    write_json(output/"frozen_graphs.json", frozen)
    frozen_sha = file_hash(output/"frozen_graphs.json")
    # New decisions are durably frozen before loading the already-used evaluation.
    hidden, _ = _load_split(PARENT, "evaluation", tasks["splits"]["evaluation"])
    scores = project_scores(hidden, banks)
    evaluation = {}
    for name, edges in frozen.items():
        evaluation[name] = {}
        for key, edge in edges.items():
            values = scores[name][:, positions[edge["destination"]]]
            checked = relations.evaluate_propagated_relations(
                edge["graph"], values, seed=parent_models["banks"][name]["edges"][key]["evaluation_seed"]+100000,
                controls=32)
            for row in checked["relations"]:
                row["passes_development_and_reused_evaluation"] = (
                    row["polynomial"] in edge["dev_accepted"] and row["passes_frozen_test_gate"])
            evaluation[name][key] = checked
    write_json(output/"graph_evaluation.json", evaluation)
    verify(output)
    if frozen_sha != file_hash(output/"frozen_graphs.json"):
        raise ValueError("Frozen graphs changed during evaluation")
    summary = aggregate_audit(frozen, evaluation)
    result = {"schema_version": "1.0", "study_id": STUDY_ID, "status": "completed", "conformance": "EXPLORATORY",
              "scientific_fingerprint": plan["scientific_fingerprint"], "confirmed": False,
              "classification": summary["classification"], "controls": controls, "graph_audit": summary,
              "neural_task_accuracy": None, "neural_task_accuracy_status": "not_measured_this_study",
              "neural_compression_tested": False, "neural_parameter_reduction": None,
              "neural_speedup": None, "exact_neural_claim": False, "gpu_used": False,
              "artifacts": {name: file_hash(output/name) for name in
                            ("controls.json", "frozen_graphs.json", "graph_evaluation.json")},
              "elapsed_seconds": time.perf_counter()-started, "completed_at": datetime.now(timezone.utc).isoformat()}
    write_json(output/"result.json", result)
    write_json(output/"campaign_results.json", {**result, "result_path": "result.json",
                                               "result_sha256": file_hash(output/"result.json")})
    return {"root": str(output), "controls_passed": controls["passed"], "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](args.output), allow_nan=False))
