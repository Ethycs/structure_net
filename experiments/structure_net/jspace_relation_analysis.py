"""Train/dev-only freezing followed by untouched relation/intervention scoring.

The symbolic claims concern approximate polynomial surrogates. These exploratory
counts are neither familywise-confirmed discoveries nor exact neural proofs.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from . import jspace_relations as relations


BANK_NAMES = ("jacobian", "pca", "random")
COUNTRIES = ("China", "France", "Germany", "Russia")
PROTOCOL = {
    "maximum_degree": 2, "max_candidates_per_degree": 8,
    "max_normalized_rms": .10, "max_observed_to_shuffle": 1/3,
    "shuffle_controls": 32, "layer_relations": 3, "joint_relations": 4,
    "symbolic_max_generators": 7, "symbolic_timeout_seconds": 5,
    "rational_max_denominator": 1000, "rationalization_tolerance": .01,
    "maximum_banks": 3, "maximum_layers": 12, "maximum_edges": 6,
    "maximum_symbolic_calls": 18,
}


def _safe(value):
    """Convert NumPy metadata and reject nonfinite/non-JSON report values."""
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported report value: {type(item).__name__}")
    return json.loads(json.dumps(value, default=convert, allow_nan=False))


def _fingerprint(frozen):
    content = {key: value for key, value in frozen.items() if key != "content_sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _layout(layers, edges):
    layers, edges = list(layers), [list(edge) for edge in edges]
    if (not 1 <= len(layers) <= 12 or len(set(layers)) != len(layers)
            or any(type(layer) is not int or layer < 0 for layer in layers)):
        raise ValueError("Use one to twelve distinct nonnegative integer layers")
    if (not 1 <= len(edges) <= 6 or len({tuple(edge) for edge in edges}) != len(edges)
            or any(len(edge) != 2 or edge[0] not in layers or edge[1] not in layers
                   or edge[0] >= edge[1] for edge in edges)):
        raise ValueError("Use one to six distinct forward edges between declared layers")
    return layers, edges


def _scores(values, layers, *, allow_empty=False):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 3 or values.shape[1:] != (len(layers), 4)
            or (not allow_empty and not len(values))):
        raise ValueError("Scores must have shape [N, declared layers, 4]")
    if not np.isfinite(values).all():
        raise ValueError("All captured scores must be finite")
    return values


def _bank_names(scores):
    if not scores or not set(scores).issubset(BANK_NAMES):
        raise ValueError("Provide one to three named banks: jacobian, pca, random")
    return [name for name in BANK_NAMES if name in scores]


def _edge_key(source, destination):
    return f"{source}->{destination}"


def _accepted_listing(bank):
    return {"variable_names": bank["variable_names"], "mean": bank["mean"], "scale": bank["scale"],
            "exponents": bank["exponents"],
            "polynomials": [{key: candidate[key] for key in
                             ("id", "degree", "coefficients", "dev_rms", "dev_abs95")}
                            for candidate in bank["candidates"] if candidate["accepted"]],
            "coordinates": "train-standardized continuous scores"}


def fit_models(scores, *, layers, edges, seed=7):
    """Return a persistable freeze, touching ONLY train/development score arrays.

    ``scores[bank]['evaluation']`` may exist but is never read here. Each bank's
    train/development matrix has shape [N,len(layers),4]. Counts across banks
    must match; row identity/alignment must be provided by the capture caller.
    At most three banks and six edges imply at most eighteen symbolic attempts.
    """
    layers, edges = _layout(layers, edges)
    names = _bank_names(scores)
    positions = {layer: i for i, layer in enumerate(layers)}
    output = {"version": 1, "status": "frozen_before_evaluation", "seed": int(seed),
              "layers": layers, "edges": edges, "protocol": dict(PROTOCOL), "banks": {},
              "exact_neural_claim": False,
              "interpretation": "exploratory continuous relations and approximate rational surrogates",
              "row_alignment": "caller must align case identities across measured banks"}
    counts = None
    attempts = 0
    for bank_index, name in enumerate(names):
        train = _scores(scores[name]["train"], layers)
        dev = _scores(scores[name]["development"], layers)
        current_counts = {"train": len(train), "development": len(dev)}
        if counts is not None and current_counts != counts:
            raise ValueError("Train/development row counts must align across banks")
        counts = current_counts
        variables = list(COUNTRIES) if name == "jacobian" else [f"{name}_axis{i}" for i in range(4)]
        fitted = {"rows": current_counts, "layers": {}, "edges": {}}
        for layer in layers:
            index = positions[layer]
            bank = relations.fit_relation_bank(train[:, index], degree=2,
                                                variable_names=variables, max_candidates=8)
            bank = relations.validate_relation_bank(bank, dev[:, index],
                                                     seed=seed+bank_index*1000+layer,
                                                     controls=32, max_relations=3,
                                                     max_rms=.10, max_control_ratio=1/3)
            fitted["layers"][str(layer)] = bank
        for edge_index, (source, destination) in enumerate(edges):
            left, right = positions[source], positions[destination]
            edge_seed = seed + bank_index*1000 + 100 + edge_index
            joint_train = np.concatenate((train[:, left], train[:, right]), axis=1)
            joint_dev = np.concatenate((dev[:, left], dev[:, right]), axis=1)
            joint = relations.fit_relation_bank(
                joint_train, degree=2, groups=4, max_candidates=8,
                variable_names=[f"source_{v}" for v in variables]+[f"destination_{v}" for v in variables])
            joint = relations.validate_relation_bank(joint, joint_dev, seed=edge_seed,
                                                      controls=32, max_relations=4,
                                                      max_rms=.10, max_control_ratio=1/3)
            baselines = relations.fit_transition_baselines(train[:, left], train[:, right])
            propagation = relations.groebner_propagation(
                fitted["layers"][str(source)], joint, dev[:, left], dev[:, right],
                max_relations=7, timeout_s=5, max_denominator=1000,
                rationalization_tolerance=.01)
            attempts += 1
            propagation_dev = relations.evaluate_propagated_relations(
                propagation, dev[:, right], seed=edge_seed+10000, controls=32)
            dev_pass = [row["polynomial"] for row in propagation_dev["relations"]
                        if row["passes_frozen_test_gate"] and row["passes_dev_residual_gate"]]
            fitted["edges"][_edge_key(source, destination)] = {
                "source": source, "destination": destination, "joint_bank": joint,
                "transition_baselines": baselines, "propagation": propagation,
                "transition_baselines_development": relations.evaluate_transition_baselines(
                    baselines, dev[:, left], dev[:, right]),
                "propagation_development": propagation_dev,
                "dev_validated_polynomials": dev_pass,
                "evaluation_seed": edge_seed+20000,
            }
        output["banks"][name] = fitted
    output["symbolic_attempts"] = attempts
    output = _safe(output)
    output["content_sha256"] = _fingerprint(output)
    return output


def _evaluate_edge(edge, source, destination, *, seed):
    propagated = relations.evaluate_propagated_relations(edge["propagation"], destination,
                                                          seed=seed, controls=32)
    dev_validated = set(edge["dev_validated_polynomials"])
    for row in propagated["relations"]:
        row["passes_development_and_evaluation"] = (row["polynomial"] in dev_validated
                                                      and row["passes_frozen_test_gate"])
    return {
        "rows": len(source),
        "joint_relations": relations.evaluate_relation_bank(
            edge["joint_bank"], np.concatenate((source, destination), axis=1), seed=seed, controls=32),
        "transition_baselines": relations.evaluate_transition_baselines(
            edge["transition_baselines"], source, destination),
        "propagated_relations": propagated,
        "validated_propagation_count": sum(row["passes_development_and_evaluation"]
                                             for row in propagated["relations"]),
        "exact_neural_claim": False,
    }


def _source_compatibility(bank, source):
    accepted = [candidate for candidate in bank["candidates"] if candidate["accepted"]]
    if not accepted:
        return None, {"defined": False, "reason": "no accepted natural source relations",
                      "compatible_rows": None, "incompatible_rows": None, "relation_checks": []}
    # This evaluates frozen polynomials; it does not condition on task success.
    features = relations._features(bank, source)
    compatible = np.ones(len(source), dtype=bool)
    checks = []
    for candidate in accepted:
        absolute = np.abs(features @ np.asarray(candidate["coefficients"]))
        passes = absolute <= candidate["dev_abs95"]
        compatible &= passes
        checks.append({"id": candidate["id"], "frozen_dev_abs95": candidate["dev_abs95"],
                       "absolute_residuals": absolute.tolist(), "within_threshold": passes.tolist()})
    return compatible, {"defined": True, "compatible_rows": int(compatible.sum()),
                        "incompatible_rows": int((~compatible).sum()),
                        "criterion": "all accepted source relations within frozen development p95; no task-accuracy filter",
                        "relation_checks": checks}


def _task_scores(records):
    output = {"rows": len(records), "interpretation": "separate task endpoints; never a gate on algebra rows"}
    for key in ("original_correct", "counterfactual_correct"):
        if any(record.get(key) is not None and type(record[key]) is not bool for record in records):
            raise ValueError("Task correctness fields must be Boolean or missing")
        known = [bool(record[key]) for record in records if record.get(key) is not None]
        output[key] = {"successes": sum(known), "scored_rows": len(known),
                       "missing_rows": len(records)-len(known),
                       "accuracy": sum(known)/len(known) if known else None}
    return output


def _evaluate_interventions(frozen, scores, records):
    if scores is None and records is None:
        return {"status": "not_supplied", "groups": [], "exact_neural_claim": False}
    if scores is None or records is None:
        raise ValueError("Intervention scores and aligned records must be supplied together")
    names = _bank_names(scores)
    if not set(names).issubset(frozen["banks"]):
        raise ValueError("Intervention measured banks must have frozen natural models")
    records = _safe(list(records))
    layers = frozen["layers"]
    positions = {layer: i for i, layer in enumerate(layers)}
    identities = set()
    for record in records:
        if (not isinstance(record, dict) or not isinstance(record.get("case_id"), str)
                or type(record.get("source_site")) is not int or record["source_site"] not in positions
                or not isinstance(record.get("arm"), str)):
            raise ValueError("Intervention records need case_id, arm and a declared source_site")
        identity = (record["case_id"], record["arm"], record["source_site"])
        if identity in identities:
            raise ValueError("Duplicate intervention case/arm/site would duplicate evidence")
        identities.add(identity)
    output = {"status": "evaluated_separately_from_natural", "rows": len(records),
              "comparison": "same measured bank as intervention arm only", "groups": [],
              "records": records, "task_scores": _task_scores(records),
              "unmatched_arm_row_indices": [i for i, record in enumerate(records) if record["arm"] not in names],
              "exact_neural_claim": False}
    for name in names:
        values = _scores(scores[name], layers, allow_empty=True)
        if len(values) != len(records):
            raise ValueError("Intervention score rows and record rows must align")
        for site in sorted({record["source_site"] for record in records if record["arm"] == name}):
            indices = [i for i, record in enumerate(records) if record["arm"] == name and record["source_site"] == site]
            selected = values[indices]
            group_records = [records[i] for i in indices]
            group = {"measured_bank": name, "intervention_arm": name, "source_site": site,
                     "rows": len(indices), "row_indices": indices,
                     "case_ids": [record["case_id"] for record in group_records],
                     "task_scores": _task_scores(group_records), "edges": {}}
            for source, destination in frozen["edges"]:
                if source < site:
                    continue
                key = _edge_key(source, destination)
                edge = frozen["banks"][name]["edges"][key]
                left, right = selected[:, positions[source]], selected[:, positions[destination]]
                seed = edge["evaluation_seed"] + 30000 + site
                all_rows = _evaluate_edge(edge, left, right, seed=seed)
                compatible, compatibility = _source_compatibility(
                    frozen["banks"][name]["layers"][str(source)], left)
                if compatible is None:
                    subset = {"status": "undefined", "reason": compatibility["reason"], "evaluation": None}
                elif not compatible.any():
                    subset = {"status": "empty", "rows": 0, "row_indices": [], "evaluation": None}
                else:
                    subset_indices = [indices[i] for i in np.flatnonzero(compatible)]
                    subset = {"status": "evaluated", "rows": len(subset_indices), "row_indices": subset_indices,
                              "evaluation": _evaluate_edge(edge, left[compatible], right[compatible], seed=seed)}
                group["edges"][key] = {"all_rows": all_rows, "source_compatibility": compatibility,
                                      "source_compatible_subset": subset,
                                      "interpretation": "conditional subset is secondary; all intervention rows retained above"}
            output["groups"].append(group)
    return output


def evaluate_models(frozen, evaluation_scores, intervention_scores=None, intervention_records=None):
    """Score a previously frozen bundle; ``evaluation_scores[bank]`` is [N,L,4]."""
    if frozen.get("status") != "frozen_before_evaluation" or frozen.get("content_sha256") != _fingerprint(frozen):
        raise ValueError("Frozen model content or fingerprint is invalid")
    names = _bank_names(evaluation_scores)
    if set(names) != set(frozen["banks"]):
        raise ValueError("Evaluation banks must exactly match frozen banks")
    layers = frozen["layers"]
    positions = {layer: i for i, layer in enumerate(layers)}
    output = {"version": 1, "frozen_content_sha256": frozen["content_sha256"], "banks": {},
              "exact_neural_claim": False,
              "interpretation": "exploratory heldout counts; no familywise confirmation or exact neural proof"}
    expected_rows = None
    for bank_index, name in enumerate(names):
        values = _scores(evaluation_scores[name], layers)
        if expected_rows is not None and len(values) != expected_rows:
            raise ValueError("Evaluation row counts must align across banks")
        expected_rows = len(values)
        model = frozen["banks"][name]
        results = {"rows": len(values), "layers": {}, "edges": {}}
        summary = {"layer_candidates_accepted_on_dev": 0, "layer_relations_passing_test": 0,
                   "joint_candidates_accepted_on_dev": 0, "joint_relations_passing_test": 0,
                   "eliminated_polynomials": 0, "propagation_passing_dev": 0,
                   "propagation_passing_dev_and_test": 0, "symbolic_status_counts": {}}
        for layer in layers:
            bank = model["layers"][str(layer)]
            evaluated = relations.evaluate_relation_bank(bank, values[:, positions[layer]],
                                                          seed=frozen["seed"]+20000+bank_index*1000+layer,
                                                          controls=32)
            results["layers"][str(layer)] = {"polynomial_listing": _accepted_listing(bank), "evaluation": evaluated}
            summary["layer_candidates_accepted_on_dev"] += evaluated["accepted_count"]
            summary["layer_relations_passing_test"] += sum(row["passes_frozen_test_gate"] for row in evaluated["relations"])
        for source, destination in frozen["edges"]:
            key = _edge_key(source, destination)
            edge = model["edges"][key]
            evaluated = _evaluate_edge(edge, values[:, positions[source]], values[:, positions[destination]],
                                        seed=edge["evaluation_seed"])
            results["edges"][key] = {"joint_polynomial_listing": _accepted_listing(edge["joint_bank"]),
                                     "symbolic_propagation": edge["propagation"],
                                     "transition_baselines_development": edge["transition_baselines_development"],
                                     "propagation_development": edge["propagation_development"],
                                     "evaluation": evaluated}
            summary["joint_candidates_accepted_on_dev"] += evaluated["joint_relations"]["accepted_count"]
            summary["joint_relations_passing_test"] += sum(row["passes_frozen_test_gate"] for row in evaluated["joint_relations"]["relations"])
            summary["eliminated_polynomials"] += len(edge["propagation"]["consequences"])
            summary["propagation_passing_dev"] += len(edge["dev_validated_polynomials"])
            summary["propagation_passing_dev_and_test"] += evaluated["validated_propagation_count"]
            status = edge["propagation"]["status"]
            summary["symbolic_status_counts"][status] = summary["symbolic_status_counts"].get(status, 0) + 1
        results["summary"] = summary
        output["banks"][name] = results
    output["interventions"] = _evaluate_interventions(frozen, intervention_scores, intervention_records)
    if _fingerprint(frozen) != frozen["content_sha256"]:
        raise RuntimeError("Evaluation unexpectedly modified the frozen models")
    return _safe(output)


def analyze_scores(scores, intervention_scores=None, intervention_records=None, *, layers, edges, seed=7):
    """Convenience wrapper: complete ALL model freezing before reading evaluation."""
    frozen = fit_models(scores, layers=layers, edges=edges, seed=seed)
    evaluation_scores = {name: scores[name]["evaluation"] for name in frozen["banks"]}
    evaluation = evaluate_models(frozen, evaluation_scores, intervention_scores, intervention_records)
    return {"frozen_models": frozen, "evaluation": evaluation}
