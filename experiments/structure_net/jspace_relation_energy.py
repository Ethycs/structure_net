"""Post-hoc activation-span energy accounting, separate from polynomial ideals.

This diagnostic does not rerun the neural model or change frozen study gates.
It quantifies geometric projection energy, never explained task capability.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def span_energy(hidden, rows, mean):
    """Account for h=P h+r and h=mean+P(h-mean)+r in FP64.

    ``rows`` need not be orthogonal. Its SVD defines the orthogonal projector
    onto its linear span; this is not the sum of squared raw concept scores.
    Ratios pool squared norms over examples. Mean must come from training data.
    """
    hidden, rows, mean = (x.detach().double().cpu() for x in (hidden, rows, mean))
    if (hidden.ndim != 2 or not len(hidden) or rows.ndim != 2 or not len(rows)
            or rows.shape[1] != hidden.shape[1] or mean.shape != (hidden.shape[1],)
            or not all(bool(torch.isfinite(x).all()) for x in (hidden, rows, mean))):
        raise ValueError("Need finite hidden[N,d], basis[k,d], and frozen training mean[d]")
    _, singular, vh = torch.linalg.svd(rows, full_matrices=False)
    rank = int((singular > max(float(singular[0])*1e-6, 1e-12)).sum())
    orthonormal = vh[:rank]
    output = {"rows": len(hidden), "hidden_width": hidden.shape[1], "span_rank": rank,
              "singular_values": singular.tolist(), "rank_relative_tolerance": 1e-6}
    for label, values in (("raw", hidden), ("around_training_mean", hidden - mean)):
        projected = (values @ orthonormal.T) @ orthonormal
        remainder = values - projected
        total = float(values.square().sum())
        explained, residual = float(projected.square().sum()), float(remainder.square().sum())
        cross = float((projected * remainder).sum())
        output[label] = {
            "total_squared_norm": total, "projected_squared_norm": explained, "residual_squared_norm": residual,
            "projected_energy_fraction": explained / total if total else None,
            "residual_energy_fraction": residual / total if total else None,
            "pooled_relative_residual_norm": (residual / total)**.5 if total else None,
            "projection_residual_inner_product": cross,
            "relative_pythagorean_error": abs(total-explained-residual)/total if total else 0.,
        }
    return output


def run(root):
    from .jspace_relation_study import _cpu_only, file_hash, verify_capture, verify_plan, write_json

    _cpu_only()
    torch.set_num_threads(4)
    root = Path(root)
    plan = verify_plan(root, verify_model=False)
    verify_capture(root, plan)
    hidden = torch.load(root / "capture/evaluation/hidden.pt", map_location="cpu", weights_only=True)
    banks = torch.load(root / "capture/banks.pt", map_location="cpu", weights_only=True)
    result = {
        "study_id": plan["study_id"], "status": "completed_posthoc_descriptive_diagnostic",
        "scientific_fingerprint": plan["scientific_fingerprint"],
        "producer_sha256": file_hash(__file__), "input_sha256": {
            name: file_hash(root / name) for name in ("capture/evaluation/hidden.pt", "capture/banks.pt")},
        "interpretation": "orthogonal vector-span energy; not a Groebner normal form or explained task accuracy",
        "centering": "mean fixed from 512 natural training activations, never refitted on evaluation",
        "aggregation": "pooled squared norms over all 128 evaluation prompts, separately for each layer",
        "banks": {},
    }
    for arm in ("jacobian", "pca", "random"):
        result["banks"][arm] = {
            str(layer): span_energy(hidden[:, i], banks[arm][i], banks["pca_training_mean"][i])
            for i, layer in enumerate(plan["config"]["layers"])
        }
    write_json(root / "analysis/posthoc_span_energy.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Existing completed study root")
    args = parser.parse_args()
    result = run(args.output)
    for arm, layers in result["banks"].items():
        print(arm, {layer: {mode: round(values[mode]["projected_energy_fraction"], 6)
                           for mode in ("raw", "around_training_mean")} for layer, values in layers.items()})


if __name__ == "__main__":
    main()
