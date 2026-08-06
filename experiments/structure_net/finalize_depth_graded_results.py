#!/usr/bin/env python3
"""Apply declared adaptive initialization checks to a depth-graded campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_depth_graded_quotient/"
            "20260805_d8_seed7/results.json"
        ),
    )
    parser.add_argument(
        "--initial-resolution",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_depth_graded_quotient/"
            "20260805_d8_seed7/initial_resolution.json"
        ),
    )
    args = parser.parse_args()
    campaign = json.loads(args.results.read_text(encoding="utf-8"))
    diagnostic = json.loads(args.initial_resolution.read_text(encoding="utf-8"))
    if campaign["schema_version"] != "nal.tinyllm-depth-graded-quotient.v1":
        raise ValueError("unsupported campaign schema")
    if not diagnostic["all_rows_resolved"]:
        raise ValueError("initialization diagnostic did not resolve every target row")
    records = {float(item["depth"]): item for item in diagnostic["records"]}
    replaced = 0
    for run in campaign["runs"]:
        if run["quotient"] != "phase_circle":
            continue
        initial = next(item for item in run["checkpoints"] if item["step"] == 0)
        for cell in initial["depth_cells"]:
            depth = float(cell["depth"])
            if depth in records and not cell["circle_map"]["sampling_resolved"]:
                record = dict(records[depth])
                record.pop("depth", None)
                cell["circle_map"] = record
                replaced += 1
        initial["all_depth_rows_resolved"] = all(
            cell["circle_map"]["sampling_resolved"]
            for cell in initial["depth_cells"]
        )
        initial["adaptive_resolution_artifact"] = str(args.initial_resolution)
        initial["depth_defect_charge"]["common_grid_phase_points"] = initial[
            "phase_grid_points"
        ]
        initial["depth_defect_charge"]["common_grid_all_rows_resolved"] = False
        initial["depth_defect_charge"]["evidentiary_status"] = (
            "exploratory_initial_charge_decomposition_on_under-resolved_common_grid"
        )
        for checkpoint in run["checkpoints"]:
            if checkpoint["step"] > 0:
                checkpoint["depth_defect_charge"][
                    "common_grid_all_rows_resolved"
                ] = checkpoint["all_depth_rows_resolved"]
                checkpoint["depth_defect_charge"]["common_grid_phase_points"] = (
                    checkpoint["phase_grid_points"]
                )
                checkpoint["depth_defect_charge"]["evidentiary_status"] = (
                    "resolved_trained_depth_charge"
                )
    if replaced != len(records) * 3:
        raise ValueError(f"expected {len(records) * 3} replacements, found {replaced}")

    criteria = campaign["pre_registered_criteria"]
    original = dict(criteria)
    criteria.pop("all_phase_depth_charge_identities_hold", None)
    criteria.pop("all_phase_depth_rows_resolved", None)
    criteria["all_phase_rows_resolved_with_targeted_initial_checks"] = all(
        checkpoint["all_depth_rows_resolved"]
        for run in campaign["runs"]
        if run["quotient"] == "phase_circle"
        for checkpoint in run["checkpoints"]
    )
    criteria["all_trained_phase_depth_charge_identities_hold"] = all(
        checkpoint["depth_defect_charge"]["charge_identity_holds"]
        and checkpoint["depth_defect_charge"]["common_grid_all_rows_resolved"]
        for run in campaign["runs"]
        if run["quotient"] == "phase_circle"
        for checkpoint in run["checkpoints"]
        if checkpoint["step"] > 0
    )
    supported = all(criteria.values())
    campaign["claim_status"] = {
        "confirmed": False,
        "numerical_depth_family_supported": supported,
        "interpretation": (
            "Every trained depth-family, adaptive-resolution, and provenance gate passed."
            if supported
            else "At least one trained depth-family or provenance gate failed."
        ),
        "confirmation_limit": (
            "A single-seed gated mapping-telescope proxy is not a neural-ODE "
            "continuum, Whitney stratification, Reeb cosheaf, or multi-seed result."
        ),
    }
    campaign["post_run_resolution_audit"] = {
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "artifact": str(args.initial_resolution),
        "original_criteria": original,
        "replaced_initial_rows": replaced,
        "degree_values_changed": False,
        "criterion_revision": (
            "The under-resolved initialization charge decomposition is exploratory; "
            "trained checkpoints are the evidentiary defect-charge set."
        ),
    }
    partial = args.results.with_name("results.partial.json")
    _write(args.results, campaign)
    _write(partial, campaign)
    print(json.dumps(campaign["claim_status"], indent=2, sort_keys=True))
    print(args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
