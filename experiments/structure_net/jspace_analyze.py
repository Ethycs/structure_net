"""Post-hoc CPU audit of perturbation size and residual norm, without rescoring."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from .jspace_pilot import case_seed, file_hash, retained_patch, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    plan = json.loads((args.root / "plan.json").read_text())
    campaign = json.loads((args.root / "campaign_results.json").read_text())
    run = args.root / "runs/pilot/seed_7"
    observations = torch.load(run / "baseline_residuals.pt", map_location="cpu", weights_only=True)
    banks = torch.load(run / "banks.pt", map_location="cpu", weights_only=True)
    layer = campaign["result"]["selection"]["layer"]
    donors = plan["split_manifest"]["splits"]["fit"]
    groups = defaultdict(list)
    for line in (run / "events.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["event"] != "sufficiency" or row["split"] != "evaluation":
            continue
        name = row["case_id"]
        donor_name = donors[case_seed(name) % len(donors)]["case_id"]
        if donor_name != row["donor_case_id"]:
            raise ValueError("Stored donor differs from frozen selection rule")
        source = observations["evaluation"][name][layer]
        donor = banks["mean"] if row["mode"] == "mean" else observations["fit"][donor_name][layer]
        basis = banks["controls"][row["bank"]][:row["rank"]]
        patch, stats = retained_patch(basis, donor, alpha=row["alpha"])
        reconstructed = patch(source[None, None, :].to(getattr(torch, plan["dtype"]))).float()[0, 0]
        # CPU/GPU matmul rounding may differ slightly; detect the wrong intervention.
        if abs(stats["perturbation_relative_l2"] - row["diagnostics"]["perturbation_relative_l2"]) > .002:
            raise ValueError("CPU reconstruction disagrees with recorded perturbation size")
        norm_change = float((reconstructed.norm() - source.norm()) / source.norm().clamp_min(1e-8))
        key = f"rank{row['rank']}/{row['bank']}/{row['mode']}_{row['alpha']:g}"
        groups[key].append({"case_id": name, "relative_norm_change": norm_change,
                            **row["diagnostics"]})
    output = {}
    for key, rows in groups.items():
        norm = torch.tensor([abs(r["relative_norm_change"]) for r in rows])
        perturbation = torch.tensor([r["perturbation_relative_l2"] for r in rows])
        drift = torch.tensor([r["next_concept_relative_l2"] for r in rows])
        correlation = float(torch.corrcoef(torch.stack([norm, drift]))[0, 1]) if norm.std() > 0 and drift.std() > 0 else None
        output[key] = {"count": len(rows), "median_absolute_norm_change": float(norm.median()),
                       "p95_absolute_norm_change": float(torch.quantile(norm, .95)),
                       "median_relative_perturbation": float(perturbation.median()),
                       "norm_change_vs_next_score_drift_correlation": correlation, "cases": rows}
    write_json(args.output, {"kind": "post_hoc_mechanistic_audit_not_a_new_quality_test",
                             "source_sha256": file_hash(__file__), "groups": output,
                             "not_claimed": "Correlation does not establish that norm preservation fixes closure"})
    print(json.dumps({k:{n:v for n,v in s.items() if n != "cases"} for k,s in output.items()}, indent=2))


if __name__ == "__main__":
    main()
