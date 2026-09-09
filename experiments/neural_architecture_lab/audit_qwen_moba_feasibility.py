#!/usr/bin/env python3
"""Audit immutable Qwen/MoBA evidence independently of the GPU producer."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import statistics


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_read(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def quality_pass(cell):
    if cell["status"] != "completed" or not cell["sparse_active"]:
        return False
    m = cell["metrics"]
    return (m["heads_cosine_mean"] >= .95 and m["ov_cosine_mean"] >= .95
            and m["change_pearson"] is not None and m["change_pearson"] >= .95
            and m["top_quarter_change_f1"] >= .90)


def audit(path, expected_sha):
    path = Path(path)
    if sha256(path) != expected_sha:
        raise ValueError("campaign hash mismatch")
    campaign = strict_read(path)
    producer = strict_read(path.parent / "producer.json")
    identity = {k: v for k, v in producer.items() if k != "fingerprint"}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if fingerprint != producer["fingerprint"] or fingerprint != campaign["producer_fingerprint"]:
        raise ValueError("producer fingerprint mismatch")
    snapshots = {}
    for snapshot in (path.parent / "producer_source").iterdir():
        name = snapshot.name.split("_", 1)[1]
        if name in snapshots:
            raise ValueError("ambiguous source snapshot")
        snapshots[name] = snapshot
    for original, digest in producer["sources"].items():
        if sha256(snapshots[Path(original).name]) != digest:
            raise ValueError("source snapshot mismatch")
    if campaign["status"] != "completed" or not campaign["validity"]["passed"]:
        raise ValueError("incomplete or invalid campaign")
    if not campaign["model_weights_unchanged"] or campaign["frozen_weights_before"] != campaign["frozen_weights_after"]:
        raise ValueError("frozen model weights changed")
    if campaign["application_accuracy_tested"]:
        raise ValueError("this proxy experiment cannot measure application accuracy")
    stage = campaign["stage"]
    if stage != producer["stage"]:
        raise ValueError("stage mismatch")
    settings, lengths = producer["settings"], producer["lengths"]
    sources = list(producer["inputs"]["tokens"])
    seeds = [7]
    if stage == "pilot":
        sources = ["wikipedia"]
        settings = [{"block_size": 128, "top_k": 2, "layers": "all"}]
        lengths = [512, max(lengths)]
    elif stage in {"final", "shared-final"}:
        selected = producer["selection_artifact"]
        if sha256(selected["path"]) != selected["sha256"] or strict_read(selected["path"]) != selected["contents"]:
            raise ValueError("sealed selection changed")
        screen = selected["contents"]
        eligible = "shared_gpu_candidate_selected" if stage == "shared-final" else "zero_shot_candidate_selected"
        if screen["classification"] != eligible:
            raise ValueError("final stage lacks eligible validation selection")
        screen_producer = strict_read(Path(selected["path"]).parent / "producer.json")
        for key in ("inputs", "schedules", "model", "execution", "sources", "settings", "lengths"):
            if screen_producer[key] != producer[key]:
                raise ValueError(f"final producer changed {key}")
        settings = [screen["selection"]["setting"]]
        lengths = [screen["selection"]["length"]]
        seeds = [7, 17, 29, 41, 53]
    elif stage not in {"screen", "readouts", "shared-screen"}:
        raise ValueError("unknown stage")
    def name(setting):
        return f"{setting['layers']}_b{setting['block_size']}_k{setting['top_k']}"
    expected = set(itertools.product(sources, seeds, lengths, map(name, settings)))
    cells = campaign["cells"]
    keys = [(c["source"], c["seed"], c["length"], c["name"]) for c in cells]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("missing, duplicate, or unexpected cells")
    for cell in cells:
        cell_path = path.parent / "cells" / f"{cell['source']}_seed{cell['seed']}_n{cell['length']}_{cell['name']}.json"
        if strict_read(cell_path) != cell or name(cell["setting"]) != cell["name"]:
            raise ValueError("cell artifact or identity mismatch")
        schedule = producer["schedules"][cell["source"]]
        final = stage in {"final", "shared-final"}
        start = schedule["final_starts"][str(cell["seed"])] if final else schedule["validation_start"]
        lower = schedule["validation_end"] if final else schedule["train_end"]
        upper = producer["inputs"]["tokens"][cell["source"]]["count"] if final else schedule["validation_end"]
        if cell["start"] != start or not lower <= start < start + cell["length"] <= upper:
            raise ValueError("window pairing or split violation")
        if cell["status"] in {"out_of_memory", "baseline_out_of_memory"}:
            if cell["gate"] is not False or not cell["error"]:
                raise ValueError("invalid capacity failure")
            continue
        if cell["status"] != "completed" or cell["application_accuracy"] is not None:
            raise ValueError("unknown cell status or unsupported accuracy")
        if stage == "readouts":
            if any(cell[k] is not None for k in ("timing", "prefix_speedup", "gate")):
                raise ValueError("readout-only evidence must not claim speed or joint passage")
        else:
            for timing in cell["timing"].values():
                samples = timing["samples_ms"]
                if len(samples) != 5 or min(samples) <= 0 or timing["median_ms"] != statistics.median(samples):
                    raise ValueError("timing samples inconsistent")
            ratio = cell["timing"]["full"]["median_ms"] / cell["timing"]["moba"]["median_ms"]
            if ratio != cell["prefix_speedup"] or cell["gate"] != (quality_pass(cell) and ratio >= 1.05):
                raise ValueError("joint gate or timing ratio mismatch")
    counts = Counter(c["status"] for c in cells)
    expected_summary = {"requested": len(expected), "completed": counts["completed"],
                        "failed": len(expected) - counts["completed"],
                        "joint_passes": sum(c["gate"] is True for c in cells)}
    if campaign["summary"] != expected_summary:
        raise ValueError("completion or gate counts inconsistent")
    if stage.startswith("shared-"):
        if campaign["uncontended_speed_claim_allowed"] or producer["timing_configuration"] != {"warmups": 1, "samples": 5, "repeats": 1, "shared_gpu": True}:
            raise ValueError("shared GPU limitation or timing configuration changed")
    if stage in {"screen", "shared-screen"}:
        selected = None
        for length in lengths:
            candidates = []
            for setting in settings:
                rows = [c for c in cells if c["length"] == length and c["name"] == name(setting)]
                if len(rows) == len(sources) and all(c["gate"] for c in rows):
                    candidates.append((statistics.mean(c["prefix_speedup"] for c in rows),
                                       statistics.mean(c["metrics"]["change_pearson"] for c in rows), setting))
            candidates.sort(key=lambda row: row[:2], reverse=True)
            if candidates:
                selected = {"length": length, "setting": candidates[0][2]}
                break
        if campaign["selection"] != selected:
            raise ValueError("selection does not match paired joint gates")
        expected_classification = ("shared_gpu_candidate_selected" if selected else "no_shared_gpu_candidate_passed") if stage == "shared-screen" else ("zero_shot_candidate_selected" if selected else "no_zero_shot_candidate_passed")
    elif stage in {"final", "shared-final"}:
        passed = all(c["gate"] for c in cells)
        expected_classification = ("shared_gpu_prefix_feasibility_supported" if passed else "shared_gpu_prefix_feasibility_not_confirmed") if stage == "shared-final" else ("prefix_feasibility_supported" if passed else "prefix_feasibility_not_confirmed")
    else:
        expected_classification = "systems_lifecycle_only_not_quality_evidence" if stage == "pilot" else "readout_screen_completed_timing_pending"
    if campaign["classification"] != expected_classification:
        raise ValueError("campaign classification does not match its evidence")
    return {"verified": True, "campaign": str(path), "campaign_sha256": expected_sha,
            "producer_fingerprint": fingerprint, "stage": stage, "classification": campaign["classification"],
            "summary": expected_summary, "sparse_quality_passes": sum(quality_pass(c) for c in cells),
            "application_accuracy_tested": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.campaign, args.sha256), indent=2, allow_nan=False))
