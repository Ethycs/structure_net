"""CPU country-decoding diagnostic on immutable J-space activation captures."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .jspace_relation_study import ROOT, _load_split, digest, file_hash, save_tensor, verify_capture, verify_plan, write_json
from .jspace_residual_expansion import LAYERS, PARENT, matrix, remainder

STUDY_ID = "JSPACE-COUNTRY-PROBE-01"
PROTOCOL = "docs/07 - Status Reports/jspace/2026-11-13_jspace-country-probe-plan.md"
EXPANSION = ROOT / "data/experiments/jspace_residual_expansion/20261113_v1"
COUNTRIES = ("China", "France", "Germany", "Russia")
ARMS = ("full", "J4", "J94", "remainder", "PCA94", "random94")
LAMBDAS = (0.0001, 0.001, 0.01, 0.1, 1., 10., 100.)
CONFIG = {"layers": list(LAYERS), "arms": list(ARMS), "lambdas": list(LAMBDAS),
          "label": "intermediate", "classes": list(COUNTRIES), "shuffle_count": 20,
          "shuffle_seed": 137, "bootstrap_count": 2000, "bootstrap_seed": 149,
          "primary_metric": "J94_minus_full_country_probe_accuracy_at_development_selected_common_layer"}


def labels(values, count):
    values = np.asarray(values)
    if values.shape != (count,) or values.dtype.kind not in "iu" or np.any((values < 0) | (values >= 4)):
        raise ValueError("Expected one integer country label in [0,4) per row")
    return values


def features(hidden, fitted, arm):
    hidden = matrix(hidden)
    if arm == "full":
        return hidden
    if arm == "remainder":
        return remainder(hidden, fitted["jacobian"])
    key, rank = {"J4": ("jacobian", 4), "J94": ("jacobian", 94),
                 "PCA94": ("pca", 94), "random94": ("random", 94)}[arm]
    return hidden @ fitted[key][:rank].T


def predict(head, values):
    scores = (matrix(values) - head["mean"]) @ head["weight"] + head["prior"]
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite probe prediction")
    return scores


class RidgeFamily:
    """Cache feature-only eigensystem; labels never determine the features."""

    def __init__(self, training, development):
        training, development = matrix(training), matrix(development)
        if training.shape[1] != development.shape[1]:
            raise ValueError("Feature width mismatch")
        self.mean = training.mean(0)
        self.training = training - self.mean
        gram = self.training @ self.training.T
        self.scale = float(np.trace(gram)/len(gram))
        if self.scale <= 1e-20:
            raise ValueError("Degenerate training features")
        singular, vectors = np.linalg.eigh(gram)
        self.singular = np.maximum(singular, 0)
        self.vectors = vectors
        self.train_eigen = self.training.T @ vectors
        self.dev_eigen = (development-self.mean) @ self.train_eigen

    def select(self, training_labels, development_labels, lambdas=LAMBDAS):
        train_y = np.eye(4)[labels(training_labels, len(self.training))]
        dev_y = np.eye(4)[labels(development_labels, len(self.dev_eigen))]
        prior = train_y.mean(0)
        projected_y = self.vectors.T @ (train_y-prior)
        candidates, best = [], None
        for penalty in lambdas:
            if not np.isfinite(penalty) or penalty <= 0:
                raise ValueError("Ridge penalties must be finite and positive")
            alpha = projected_y/(self.singular[:, None] + penalty*self.scale)
            scores = self.dev_eigen @ alpha + prior
            correct = int(np.sum(scores.argmax(1) == development_labels))
            mse = float(np.mean((scores-dev_y)**2))
            if not np.isfinite(mse):
                raise ValueError("Nonfinite development metric")
            row = {"lambda": float(penalty), "correct": correct, "total": len(dev_y),
                   "accuracy": correct/len(dev_y), "mse": mse}
            candidates.append(row)
            key = (-correct, mse, penalty)
            if best is None or key < best[0]:
                best = (key, alpha, row)
        if best is None:
            raise ValueError("No ridge candidates")
        head = {"mean": self.mean, "weight": self.train_eigen @ best[1], "prior": prior}
        return head, {"selected": best[2], "candidates": candidates}


def select_layer(development):
    return min(development, key=lambda layer: (-development[layer]["full"]["selected"]["accuracy"],
               development[layer]["full"]["selected"]["mse"], int(layer)))


def score_predictions(scores, expected):
    expected = labels(expected, len(scores))
    prediction = np.asarray(scores).argmax(1)
    correct = int(np.sum(prediction == expected))
    n, z = len(expected), 1.959963984540054
    p = correct/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    confusion = np.zeros((4, 4), dtype=int)
    np.add.at(confusion, (expected, prediction), 1)
    return {"correct": correct, "total": n, "accuracy": p,
            "wilson_95": [float(center-half), float(center+half)],
            "confusion_true_rows_predicted_columns": confusion.tolist()}


def paired_difference(left, right, truth, *, count=2000, seed=149):
    difference = (np.asarray(left) == truth).astype(float)-(np.asarray(right) == truth).astype(float)
    indices = np.random.default_rng(seed).integers(len(truth), size=(count, len(truth)))
    return {"difference": float(difference.mean()),
            "bootstrap_95": np.quantile(difference[indices].mean(1), [.025, .975]).tolist(),
            "bootstrap_count": count, "bootstrap_seed": seed, "confirmatory": False}


def pack(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, dict):
        return {key: pack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [pack(item) for item in value]
    return value


def unpack(value):
    if isinstance(value, torch.Tensor):
        return value.numpy()
    if isinstance(value, dict):
        return {key: unpack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [unpack(item) for item in value]
    return value


def run(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only study requires CUDA_VISIBLE_DEVICES='' before Python")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new append-only artifact directory")
    torch.set_num_threads(4)
    started = time.perf_counter()
    parent = verify_plan(PARENT, verify_model=False)
    verify_capture(PARENT, parent)
    expansion_result = json.loads((EXPANSION/"result.json").read_text())
    if expansion_result["status"] != "completed":
        raise ValueError("Expansion parent did not complete")
    fit_paths = [EXPANSION/f"layer_{layer}_fit.pt" for layer in LAYERS]
    for path in fit_paths:
        if file_hash(path) != expansion_result["artifacts"][path.name]:
            raise ValueError(f"Changed expansion: {path}")
    inputs = [PARENT/"plan.json", PARENT/"tasks.json", PARENT/"capture_manifest.json",
              EXPANSION/"plan.json", EXPANSION/"result.json", *fit_paths,
              *[PARENT/f"capture/{split}/hidden.pt" for split in ("train", "development", "evaluation")]]
    sources = [Path(__file__), ROOT/PROTOCOL, ROOT/"pixi.lock", ROOT/"pyproject.toml",
               ROOT/"experiments/structure_net/jspace_relation_study.py",
               ROOT/"experiments/structure_net/jspace_residual_expansion.py"]
    plan = {"schema_version": 1, "study_id": STUDY_ID, "config": CONFIG,
            "prepared_at": datetime.now(timezone.utc).isoformat(), "conformance": "EXPLORATORY_POSTHOC",
            "parent_fingerprint": parent["scientific_fingerprint"],
            "expansion_fingerprint": expansion_result["scientific_fingerprint"],
            "input_sha256": {str(path): file_hash(path) for path in inputs},
            "producer_sha256": {str(path): file_hash(path) for path in sources},
            "runtime": {"numpy": np.__version__, "torch": str(torch.__version__),
                        "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "device": "cpu"},
            "accuracy_scope": "intermediate_country_probe_not_model_answer",
            "task_accuracy_measured": False, "signed_span_not_sparse_nonnegative_J_space": True}
    plan["scientific_fingerprint"] = digest(plan)
    output.mkdir(parents=True)
    for path in sources:
        destination = output/"producer"/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write_json(output/"plan.json", plan)
    tasks = json.loads((PARENT/"tasks.json").read_text())["splits"]
    datasets = {split: tasks[split] for split in ("train", "development", "evaluation")}
    all_ids = [row["case_id"] for rows in datasets.values() for row in rows]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Case-ID overlap")
    y = {split: np.array([COUNTRIES.index(row["intermediate"]) for row in rows])
         for split, rows in datasets.items()}
    for split, expected_count in (("train", 512), ("development", 128), ("evaluation", 128)):
        if len(y[split]) != expected_count or not np.all(np.bincount(y[split], minlength=4) == expected_count//4):
            raise ValueError(f"Unexpected counts or unbalanced country labels: {split}")
    hidden = {split: _load_split(PARENT, split, datasets[split])[0].double().numpy()
              for split in ("train", "development")}
    fits = {str(layer): unpack(torch.load(path, map_location="cpu", weights_only=True))
            for layer, path in zip(LAYERS, fit_paths)}
    for fitted in fits.values():
        for key in ("jacobian", "pca", "random"):
            basis = matrix(fitted[key])
            if basis.shape != (94, 2048) or not np.allclose(basis@basis.T, np.eye(94), atol=1e-10, rtol=0):
                raise ValueError("Invalid orthonormal expansion")
    heads, development, families = {}, {}, {}
    for layer in LAYERS:
        key, index = str(layer), parent["config"]["layers"].index(layer)
        heads[key], development[key], families[key] = {}, {}, {}
        for arm in ARMS:
            family = RidgeFamily(features(hidden["train"][:, index], fits[key], arm),
                                 features(hidden["development"][:, index], fits[key], arm))
            families[key][arm] = family
            heads[key][arm], development[key][arm] = family.select(y["train"], y["development"])
        print(json.dumps({"fit_layer": layer, "full_development": development[key]["full"]["selected"],
                          "J94_development": development[key]["J94"]["selected"]}), flush=True)
    primary_layer = select_layer(development)
    rng = np.random.default_rng(CONFIG["shuffle_seed"])
    permutations = [rng.permutation(len(y["train"])) for _ in range(CONFIG["shuffle_count"])]
    shuffled_heads, shuffled_development = {}, {}
    for arm in ARMS:
        pairs = [families[primary_layer][arm].select(y["train"][permutation], y["development"])
                 for permutation in permutations]
        shuffled_heads[arm] = [pair[0] for pair in pairs]
        shuffled_development[arm] = [pair[1] for pair in pairs]
    save_tensor(output/"heads.pt", pack({"heads": heads, "shuffled_heads": shuffled_heads}))
    write_json(output/"selection.json", {"primary_layer": int(primary_layer), "development": development,
               "shuffled_development": shuffled_development, "permutations": [p.tolist() for p in permutations],
               "heads_sha256": file_hash(output/"heads.pt"), "frozen_before_evaluation_scoring": True})
    # All scoring uses the serialized heads, after every selection has finished.
    saved = unpack(torch.load(output/"heads.pt", map_location="cpu", weights_only=True))
    evaluation = _load_split(PARENT, "evaluation", datasets["evaluation"])[0].double().numpy()
    measured, predictions, shuffled = {}, {}, {}
    templates = np.array([row["template_id"] for row in datasets["evaluation"]])
    for layer in LAYERS:
        key, index = str(layer), parent["config"]["layers"].index(layer)
        measured[key], predictions[key] = {}, {}
        for arm in ARMS:
            values = features(evaluation[:, index], fits[key], arm)
            scores = predict(saved["heads"][key][arm], values)
            if not np.array_equal(scores, predict(heads[key][arm], values)):
                raise ValueError("Saved/reloaded probe differs")
            metrics = score_predictions(scores, y["evaluation"])
            metrics["by_template"] = {name: score_predictions(scores[templates == name], y["evaluation"][templates == name])
                                      for name in sorted(set(templates))}
            measured[key][arm] = metrics
            predictions[key][arm] = scores
            if key == primary_layer:
                rows = [score_predictions(predict(head, values), y["evaluation"])
                        for head in saved["shuffled_heads"][arm]]
                accuracy = [row["accuracy"] for row in rows]
                shuffled[arm] = {"replicates": rows, "mean_accuracy": float(np.mean(accuracy)),
                                 "minimum_accuracy": min(accuracy), "maximum_accuracy": max(accuracy)}
    save_tensor(output/"predictions.pt", pack({"scores": predictions, "true_labels": y["evaluation"],
                "case_ids": [row["case_id"] for row in datasets["evaluation"]]}))
    contrast = paired_difference(predictions[primary_layer]["J94"].argmax(1),
                                 predictions[primary_layer]["full"].argmax(1), y["evaluation"],
                                 count=CONFIG["bootstrap_count"], seed=CONFIG["bootstrap_seed"])
    result = {"schema_version": 1, "study_id": STUDY_ID, "status": "completed", "confirmed": False,
              "classification": "exploratory_country_decodability_measured_no_causal_claim",
              "conformance": "EXPLORATORY_POSTHOC", "scientific_fingerprint": plan["scientific_fingerprint"],
              "primary_layer": int(primary_layer), "layers": measured, "shuffle_controls": shuffled,
              "primary_comparison": contrast, "majority_chance": .25,
              "elapsed_seconds": time.perf_counter()-started, "model_inference_performed": False,
              "task_accuracy_measured": False, "accuracy_scope": plan["accuracy_scope"],
              "artifacts": {path.name: file_hash(path) for path in sorted(output.glob("*.pt"))},
              "selection_sha256": file_hash(output/"selection.json")}
    write_json(output/"result.json", result)
    campaign = {key: result[key] for key in ("schema_version", "study_id", "status", "classification", "confirmed",
                "conformance", "scientific_fingerprint", "primary_layer", "primary_comparison", "accuracy_scope",
                "task_accuracy_measured", "model_inference_performed", "elapsed_seconds")}
    campaign.update({"config": CONFIG, "runtime": plan["runtime"], "requested_runs": 1, "completed_runs": 1,
                     "failed_runs": 0, "retries": 0, "excluded_runs": 0, "primary_metric": contrast["difference"],
                     "primary_metric_scope": CONFIG["primary_metric"], "primary_layer_metrics": measured[primary_layer],
                     "shuffle_controls": shuffled, "result_path": "result.json", "result_sha256": file_hash(output/"result.json")})
    write_json(output/"campaign_results.json", campaign)
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        result = run(args.output)
    except BaseException as exc:
        if not existed and args.output.exists():
            write_json(args.output/"failure.json", {"status": "invalid_diagnostic", "error_type": type(exc).__name__,
                       "error": str(exc), "recorded_at": datetime.now(timezone.utc).isoformat()})
        raise
    print(json.dumps({key: result[key] for key in ("status", "primary_layer", "primary_comparison", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
