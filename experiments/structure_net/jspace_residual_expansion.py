"""CPU residual expansion in a frozen J dictionary, with signed/conic fits separate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import scipy
from scipy.optimize import nnls
import torch

from .jspace_relation_study import ROOT, digest, file_hash, save_tensor, verify_capture, verify_plan, write_json


STUDY_ID = "JSPACE-RESIDUAL-EXPANSION-01"
PROTOCOL = "docs/07 - Status Reports/jspace/2026-11-13_jspace-residual-expansion-plan.md"
PARENT = ROOT / "data/experiments/jspace_relation_propagation/20261113_neural_v1"
OLD_ROOT = ROOT / "data/experiments/jspace_cav_pilot/20261113T0728Z_pilot02"
OLD_LENS = OLD_ROOT / "runs/pilot/seed_7/lens.pt"
LAYERS = (11, 12, 17, 18, 23, 24)
BUDGETS = (4, 8, 16, 25, 32, 64, 94)
SPARSE_BUDGETS = (4, 8, 16, 25, 32, 94)


def matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not min(values.shape) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite matrix")
    return values


def normalize(rows):
    rows = matrix(rows)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero dictionary atom")
    return rows / norms


def orthogonal_rows(rows):
    _, singular, vh = np.linalg.svd(matrix(rows), full_matrices=False)
    return vh[singular > max(1e-12, singular[0]*1e-6)]


def remainder(values, basis):
    return values - (values @ basis.T) @ basis


def _append(basis, vector):
    vector = remainder(vector[None, :], basis)[0]
    vector = remainder(vector[None, :], basis)[0]
    norm = np.linalg.norm(vector)
    if norm <= 1e-6:
        return basis, False
    return np.concatenate((basis, (vector/norm)[None, :])), True


def fit_expansions(training, dictionary, initial_indices, *, max_rank, seed=7):
    """Choose all shared directions using training states only."""
    training, dictionary = matrix(training), normalize(dictionary)
    if training.shape[1] != dictionary.shape[1] or max_rank > min(training.shape):
        raise ValueError("Incompatible width or insufficient training rank budget")
    initial = list(initial_indices)
    if len(set(initial)) != len(initial) or any(not 0 <= i < len(dictionary) for i in initial):
        raise ValueError("Initial atom indices must be distinct and in range")
    base = orthogonal_rows(dictionary[initial])
    if len(base) != len(initial) or not len(base) <= max_rank <= training.shape[1]:
        raise ValueError("Independent initial atoms and a feasible rank budget are required")
    mean = training.mean(0)
    centered = training - mean
    dictionary_scores = centered @ dictionary.T
    basis = base.copy()
    chosen = list(initial)
    gains = []
    while len(basis) < max_rank:
        innovations = remainder(dictionary, basis)
        innovations = remainder(innovations, basis)
        norms = np.linalg.norm(innovations, axis=1)
        valid = norms > 1e-6
        valid[chosen] = False
        if not valid.any():
            break
        units = innovations / np.maximum(norms[:, None], 1e-30)
        residual_scores = dictionary_scores - (centered @ basis.T) @ (basis @ dictionary.T)
        gain = np.sum((residual_scores / np.maximum(norms[None, :], 1e-30))**2, axis=0)
        gain[~valid] = -np.inf
        index = int(gain.argmax())
        basis, appended = _append(basis, units[index])
        if not appended:
            raise RuntimeError("Selected nondegenerate dictionary innovation disappeared")
        chosen.append(index)
        gains.append(float(gain[index]))
    residual = remainder(centered, base)
    _, _, principal = np.linalg.svd(residual, full_matrices=False)
    pca = base.copy()
    for vector in principal:
        if len(pca) == max_rank:
            break
        pca, _ = _append(pca, vector)
    random = base.copy()
    rng = np.random.default_rng(seed)
    while len(random) < max_rank:
        random, _ = _append(random, rng.normal(size=training.shape[1]))
    return {"mean": mean, "jacobian": basis, "pca": pca, "random": random,
            "initial_rank": len(base), "selected_indices": chosen, "training_incremental_squared_norm": gains}


def energy(values, residual):
    total = float(np.sum(values**2))
    squared = float(np.sum(residual**2))
    return {"total_squared_norm": total, "residual_squared_norm": squared,
            "residual_energy_fraction": squared/total if total else None,
            "captured_energy_fraction": 1-squared/total if total else None}


def evaluate_expansions(evaluation, fitted, budgets):
    evaluation = matrix(evaluation)
    results = {}
    for arm in ("jacobian", "pca", "random"):
        basis = fitted[arm]
        rows = []
        previous = {"raw": float("inf"), "around_training_mean": float("inf")}
        for budget in budgets:
            prefix = basis[:budget]
            row = {"requested_rank": budget, "achieved_rank": len(prefix)}
            for label, values in (("raw", evaluation), ("around_training_mean", evaluation-fitted["mean"])):
                residual = remainder(values, prefix)
                score = energy(values, residual)
                if score["residual_squared_norm"] > previous[label] + 1e-10*max(1., previous[label]):
                    raise RuntimeError("Nested projection increased residual energy")
                previous[label] = score["residual_squared_norm"]
                row[label] = score
            rows.append(row)
        results[arm] = rows
    initial = fitted["jacobian"][:fitted["initial_rank"]]
    first = remainder(evaluation, initial)
    repeated = (first @ initial.T) @ initial
    results["fixed_projector_second_pass_energy_fraction"] = float(np.sum(repeated**2)/np.sum(evaluation**2))
    return results


def nonnegative_pursuit(values, dictionary, initial_indices, *, budgets=(4, 8, 16, 25, 32, 94)):
    """Greedy new atoms + exact support NNLS, with a separate full-cone endpoint.

    Unlike a signed linear span, coefficients remain nonnegative. This bounded
    matching-pursuit variant is not the paper's gradient-pursuit implementation.
    """
    values, dictionary = matrix(values), normalize(dictionary)
    if values.shape[1] != dictionary.shape[1]:
        raise ValueError("Activation/dictionary width mismatch")
    budgets = list(budgets)
    initial = list(initial_indices)
    if (not budgets or budgets != sorted(set(budgets)) or budgets[0] != len(initial)
            or budgets[-1] != len(dictionary) or len(set(initial)) != len(initial)
            or any(not 0 <= index < len(dictionary) for index in initial)):
        raise ValueError("Budgets must start at initial support and end at full dictionary size")
    # D.T = U @ small_A. Orthogonal output coordinates reduce NNLS work.
    u, singular, vh = np.linalg.svd(dictionary.T, full_matrices=False)
    small_a = singular[:, None]*vh
    small_b = values @ u
    coefficients = {budget: np.zeros((len(values), len(dictionary))) for budget in budgets}
    selections = []
    for row, target in enumerate(small_b):
        selected = list(initial)

        def solve(indices):
            local, _ = nnls(small_a[:, indices], target, maxiter=30*len(indices))
            full = np.zeros(len(dictionary))
            full[indices] = local
            return full

        current = solve(selected)
        stopped = False
        for budget in budgets[:-1]:
            while len(selected) < budget and not stopped:
                correlation = small_a.T @ (target-small_a @ current)
                correlation[selected] = -np.inf
                atom = int(correlation.argmax())
                if correlation[atom] <= 1e-10*max(1., np.linalg.norm(target)):
                    stopped = True
                    break
                selected.append(atom)
                current = solve(selected)
            coefficients[budget][row] = current
        coefficients[budgets[-1]][row] = solve(list(range(len(dictionary))))
        selections.append(selected)
    rows = []
    previous = float("inf")
    residuals = {}
    for budget in budgets:
        coeff = coefficients[budget]
        residual = values-coeff @ dictionary
        residuals[budget] = residual
        score = energy(values, residual)
        if score["residual_squared_norm"] > previous+1e-10*max(1., previous):
            raise RuntimeError("Nested NNLS support increased residual energy")
        previous = score["residual_squared_norm"]
        rows.append({"selected_atom_budget": budget, **score,
                     "nonzero_coefficients_per_example": np.sum(coeff > 1e-10, axis=1).tolist()})
    final = residuals[budgets[-1]]
    kkt = final @ dictionary.T
    return {"metrics": rows, "coefficients": coefficients, "selection_order": selections,
            "final_residual": final, "max_full_cone_positive_correlation": max(0., float(kkt.max()))}


def run(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run this CPU diagnostic with CUDA_VISIBLE_DEVICES='' before Python")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new append-only artifact root")
    torch.set_num_threads(4)
    started = time.perf_counter()
    parent = verify_plan(PARENT, verify_model=False)
    verify_capture(PARENT, parent)
    old = torch.load(OLD_LENS, map_location="cpu", weights_only=True)
    new = torch.load(PARENT/"capture/lens.pt", map_location="cpu", weights_only=True)
    fields = ("method", "upstream_commit", "model_name_or_path", "model_dtype", "target_layer", "d_model",
              "prompts_sha256", "n_prompts", "max_seq_len", "skip_first", "aggregation")
    for field in fields:
        if old["metadata"][field] != new["metadata"][field]:
            raise ValueError(f"Dictionary/capture estimator mismatch: {field}")
    paths = [OLD_LENS, OLD_ROOT/"plan.json", PARENT/"plan.json", PARENT/"capture_manifest.json",
             PARENT/"capture/lens.pt", PARENT/"capture/banks.pt",
             PARENT/"capture/train/hidden.pt", PARENT/"capture/evaluation/hidden.pt"]
    source_paths = [Path(__file__), ROOT/PROTOCOL, ROOT/"pixi.lock", ROOT/"pyproject.toml"]
    plan = {"study_id": STUDY_ID, "schema_version": 1, "conformance": "EXPLORATORY_POSTHOC",
            "prepared_at": datetime.now(timezone.utc).isoformat(), "seed": 7, "layers": list(LAYERS),
            "budgets": list(BUDGETS), "sparse_budgets": list(SPARSE_BUDGETS),
            "parent_fingerprint": parent["scientific_fingerprint"],
            "input_sha256": {str(p): file_hash(p) for p in paths},
            "producer_sha256": {str(p): file_hash(p) for p in source_paths},
            "old_dictionary_torch": old["metadata"]["torch_version"],
            "new_dictionary_torch": new["metadata"]["torch_version"],
            "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "scipy": scipy.__version__,
                        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS")},
            "data_reuse": "Previously inspected capture; train selects shared spans; evaluation is exploratory reuse",
            "not_claimed": ["full_J_space", "neural_quality_preservation", "causal_CAVs", "compression_speedup", "Groebner_propagation"]}
    plan["scientific_fingerprint"] = digest(plan)
    output.mkdir(parents=True)
    for source in source_paths:
        target = output/"producer"/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    write_json(output/"plan.json", plan)
    training = torch.load(PARENT/"capture/train/hidden.pt", map_location="cpu", weights_only=True).double().numpy()
    evaluation = torch.load(PARENT/"capture/evaluation/hidden.pt", map_location="cpu", weights_only=True).double().numpy()
    banks = torch.load(PARENT/"capture/banks.pt", map_location="cpu", weights_only=True)
    old_ids = old["metadata"]["token_ids"]
    initial = [old_ids.index(token) for token in parent["concept_token_ids"]]
    outputs = {}
    for layer in LAYERS:
        index = parent["config"]["layers"].index(layer)
        dictionary = normalize(old["token_rows"][layer].double().numpy())
        anchors = normalize(banks["jacobian"][index].double().numpy())
        discrepancy = np.linalg.norm(dictionary[initial]-anchors, axis=1).tolist()
        dictionary[initial] = anchors
        fitted = fit_expansions(training[:, index], dictionary, initial, max_rank=94, seed=7+layer)
        # Persist the shared fit before this layer's evaluation encoding/scoring.
        packed_fit = {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                      for key, value in fitted.items()}
        save_tensor(output/f"layer_{layer}_fit.pt", packed_fit)
        signed = evaluate_expansions(evaluation[:, index], fitted, BUDGETS)
        sparse = nonnegative_pursuit(evaluation[:, index], dictionary, initial, budgets=SPARSE_BUDGETS)
        final_residual = remainder(evaluation[:, index], fitted["jacobian"])
        reconstructed = (evaluation[:, index] @ fitted["jacobian"].T) @ fitted["jacobian"]
        reconstruction_error = float(np.max(np.abs(evaluation[:, index]-(reconstructed+final_residual))))
        save_tensor(output/f"layer_{layer}_residuals.pt", {
            "orthogonal_94_residual": torch.from_numpy(final_residual),
            "orthogonal_94_coefficients": torch.from_numpy(evaluation[:, index] @ fitted["jacobian"].T),
            "nonnegative_94_residual": torch.from_numpy(sparse["final_residual"]),
            "nonnegative_coefficients": {k: torch.from_numpy(v) for k,v in sparse["coefficients"].items()},
            "dictionary": torch.from_numpy(dictionary), "token_ids": old_ids,
            "token_strings": old["metadata"]["token_strings"],
        })
        outputs[str(layer)] = {"signed_expansion": signed, "nonnegative_pursuit": sparse["metrics"],
                              "old_new_anchor_unit_l2": discrepancy,
                              "selected_token_strings": [old["metadata"]["token_strings"][i] for i in fitted["selected_indices"]],
                              "per_example_sparse_selected_indices": sparse["selection_order"],
                              "max_full_cone_positive_correlation": sparse["max_full_cone_positive_correlation"],
                              "decomposition_max_absolute_error": reconstruction_error}
        print(json.dumps({"completed_layer": layer, "signed_rank94_captured": signed["jacobian"][-1]["raw"]["captured_energy_fraction"],
                          "nonnegative_full_captured": sparse["metrics"][-1]["captured_energy_fraction"]}), flush=True)
    artifacts = {p.name: file_hash(p) for p in sorted(output.glob("*.pt"))}
    result = {"study_id": STUDY_ID, "schema_version": 1, "status": "completed",
              "conformance": "EXPLORATORY_POSTHOC", "scientific_fingerprint": plan["scientific_fingerprint"],
              "layers": outputs, "artifacts": artifacts, "elapsed_seconds": time.perf_counter()-started,
              "confirmed": False, "neural_inference_performed": False,
              "interpretation": "Geometric residual decomposition, not causal/task/Gröbner evidence",
              "not_claimed": plan["not_claimed"]}
    write_json(output/"result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        result = run(args.output)
    except BaseException as exc:
        if not existed and args.output.exists():
            write_json(args.output/"failure.json", {"status": "invalid_diagnostic", "error_type": type(exc).__name__,
                       "error": str(exc), "recorded_at": datetime.now(timezone.utc).isoformat()})
        raise
    print(json.dumps({"status": result["status"], "study_id": STUDY_ID,
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
