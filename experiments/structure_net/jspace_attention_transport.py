"""CPU fitting and evaluation of transported J observables; no vanishing-ideal gate."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from .jspace_attention_runtime import residual_nodes
from .jspace_relation_tasks import COUNTRIES
from .jspace_relation_study import file_hash, save_tensor, write_json
from .jspace_causal_components import cpu_only, paired_test


PENALTIES = (1e-4, .001, .01, .1, 1., 10.)


def polynomial(x, quadratic):
    if not quadratic:
        return x
    return np.concatenate((x, np.stack([x[:, i]*x[:, j] for i in range(x.shape[1]) for j in range(i, x.shape[1])], 1)), 1)


def regress(x, y, xd, yd, classification=False):
    mean, prior = x.mean(0), y.mean(0)
    xc = x-mean
    gram = xc.T@xc
    scale = max(float(np.trace(gram)/len(gram)), 1e-12)
    s, v = np.linalg.eigh(gram)
    rhs = v.T@xc.T@(y-prior)
    best = None
    for penalty in PENALTIES:
        weight = v@(rhs/(np.maximum(s, 0)[:, None]+penalty*scale))
        predicted = (xd-mean)@weight+prior
        mse = float(np.mean((predicted-yd)**2))
        correct = int((predicted.argmax(1) == yd.argmax(1)).sum()) if classification else None
        key = (-correct, mse, penalty) if classification else (mse, penalty)
        if best is None or key < best[0]:
            best = (key, {"mean": torch.from_numpy(mean), "prior": torch.from_numpy(prior),
                          "weight": torch.from_numpy(weight)}, {"lambda": penalty, "development_mse": mse,
                                                                 "development_correct": correct, "development_total": len(yd)})
    return best[1], best[2]


def predict(model, x):
    return (x-model["mean"].numpy())@model["weight"].numpy()+model["prior"].numpy()


def gather(capture, pairs, sides=("source", "donor")):
    values = torch.cat([capture[side] for side in sides], 0).double().numpy()
    country = np.array([COUNTRIES.index(p[side]["intermediate"]) for side in sides for p in pairs])
    attribute = np.array([float(p[side]["attribute"] == "nationality") for side in sides for p in pairs])[:, None]
    return values, country, attribute


def fit(root):
    from .jspace_attention_study import verify, checked, STUDY_ID
    cpu_only()
    torch.set_num_threads(4)
    plan = verify(root)
    checked(root, "discovery")
    checked(root, "training")
    started = time.perf_counter()
    tasks = json.loads((root/"tasks.json").read_text())
    selection = json.loads((root/"selection.json").read_text())
    train = torch.load(root/"training/features.pt", map_location="cpu", weights_only=True)
    dev = torch.load(root/"discovery/features.pt", map_location="cpu", weights_only=True)
    all_nodes = train["nodes"]
    if all_nodes != dev["nodes"]:
        raise ValueError("Capture node alignment changed")
    nodes = residual_nodes(selection["layer"])
    indices = [all_nodes.index(n) for n in nodes]
    x, y, u = gather(train, tasks["pairs"]["train"])
    d, yd, ud = gather(dev, tasks["pairs"]["development"])
    origin_index = all_nodes.index(f"{selection['layer']}.attention")
    ox, od = x[:, origin_index].reshape(len(x), -1), d[:, origin_index].reshape(len(d), -1)
    om, os = ox.mean(0), np.maximum(ox.std(0), 1e-4)
    origin_probe, origin_metric = regress((ox-om)/os, np.eye(4)[y], (od-om)/os, np.eye(4)[yd], True)
    x, d = x[:, indices].reshape(len(x), len(nodes), -1), d[:, indices].reshape(len(d), len(nodes), -1)
    mean = x.mean(0)
    scale = np.maximum(x.std(0), 1e-4)
    z, zd = (x-mean)/scale, (d-mean)/scale
    models = {"nodes": nodes, "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale), "steps": {}, "probes": []}
    models["origin"] = {"node": all_nodes[origin_index], "probe": origin_probe, "mean": torch.from_numpy(om), "scale": torch.from_numpy(os)}
    metrics = {"steps": {}, "probes": []}
    for index, node in enumerate(nodes):
        probe, metric = regress(z[:, index], np.eye(4)[y], zd[:, index], np.eye(4)[yd], True)
        models["probes"].append(probe)
        metrics["probes"].append({"node": node, **metric})
    for name in ("linear", "quadratic"):
        models["steps"][name], metrics["steps"][name] = [], []
        for i in range(len(nodes)-1):
            tx = polynomial(np.concatenate((z[:, i], u), 1), name == "quadratic")
            dx = polynomial(np.concatenate((zd[:, i], ud), 1), name == "quadratic")
            fitted, metric = regress(tx, z[:, i+1], dx, zd[:, i+1])
            models["steps"][name].append(fitted)
            metrics["steps"][name].append({"source": nodes[i], "target": nodes[i+1], **metric})
    save_tensor(root/"transport.pt", models)
    restored = torch.load(root/"transport.pt", map_location="cpu", weights_only=True)
    if not np.array_equal(predict(models["probes"][0], zd[:, 0]), predict(restored["probes"][0], zd[:, 0])):
        raise ValueError("Transport head roundtrip changed")
    summary = {"study_id": STUDY_ID, "scientific_fingerprint": plan["scientific_fingerprint"], "selection": selection,
               "frozen_before_evaluation": not (root/"evaluation").exists(), "nodes": nodes,
               "training_observations": len(x), "training_independent_worlds": len(x)//2,
               "development_observations": len(d), "context": "known binary requested attribute; no future activations or labels supplied",
               "metrics": metrics, "origin_probe": origin_metric, "elapsed_seconds": time.perf_counter()-started,
               "transport_sha256": file_hash(root/"transport.pt")}
    if not summary["frozen_before_evaluation"]:
        raise ValueError("Evaluation already exists")
    write_json(root/"fit.json", summary)
    return {"nodes": len(nodes), "fit_seconds": summary["elapsed_seconds"]}


def rollout(model, initial, attribute, kind):
    values = [initial]
    alive = np.ones(len(initial), dtype=bool)
    masks = [alive.copy()]
    for step in model["steps"][kind]:
        with np.errstate(over="ignore", invalid="ignore"):
            x = polynomial(np.concatenate((values[-1], attribute), 1), kind == "quadratic")
            value = predict(step, x)
        alive &= np.isfinite(value).all(1) & (np.abs(value).max(1) < 1e4)
        value[~alive] = 0.  # Placeholder only; masked failures cannot count as predictions.
        values.append(value)
        masks.append(alive.copy())
    return np.stack(values, 1), np.stack(masks, 1)


def error_summary(predicted, expected, valid=None):
    if valid is None:
        valid = np.ones(len(predicted), dtype=bool)
    if not valid.all():
        return {"status": "diverged", "failed": int((~valid).sum()), "total": len(valid), "nrmse": None}
    denominator = float(np.sum(expected**2))
    if denominator < 1e-16:
        return {"status": "zero_reference", "failed": 0, "total": len(valid), "nrmse": None}
    return {"status": "measured", "failed": 0, "total": len(valid),
            "nrmse": float(np.sqrt(np.sum((predicted-expected)**2)/denominator))}


def analyze(root):
    from .jspace_attention_study import verify, checked, STUDY_ID
    cpu_only()
    plan = verify(root)
    checked(root, "evaluation")
    fit = json.loads((root/"fit.json").read_text())
    if fit["transport_sha256"] != file_hash(root/"transport.pt"):
        raise ValueError("Frozen transport changed")
    model = torch.load(root/"transport.pt", map_location="cpu", weights_only=True)
    natural = torch.load(root/"evaluation/features.pt", map_location="cpu", weights_only=True)
    changed = torch.load(root/"evaluation/edited_features.pt", map_location="cpu", weights_only=True)
    records = json.loads((root/"evaluation/records.json").read_text())
    tasks = json.loads((root/"tasks.json").read_text())["pairs"]["evaluation"]
    selection = fit["selection"]
    indices = [natural["nodes"].index(n) for n in model["nodes"]]
    x, country, attribute = gather(natural, tasks, ("source",))
    origin = model["origin"]
    oi = natural["nodes"].index(origin["node"])
    ox = x[:, oi].reshape(len(x), -1)
    op = changed["scores"][:, oi].double().numpy().reshape(ox.shape)
    origin_scores = {"node": origin["node"], "total": len(x),
                     "country_correct": int((predict(origin["probe"], (ox-origin["mean"].numpy())/origin["scale"].numpy()).argmax(1) == country).sum()),
                     "patched_target_correct": int((predict(origin["probe"], (op-origin["mean"].numpy())/origin["scale"].numpy()).argmax(1) == np.array([COUNTRIES.index(p["source"]["swap_to"]) for p in tasks])).sum())}
    x = x[:, indices].reshape(len(x), len(indices), -1)
    patch = changed["scores"][:, indices].double().numpy().reshape(x.shape)
    mean, scale = model["mean"].numpy(), model["scale"].numpy()
    z, zp = (x-mean)/scale, (patch-mean)/scale
    delta = zp-z
    eligible = np.array([r["source"]["original_correct"] and r["donor"]["original_correct"] for r in records])
    arm_scores = {}
    for arm in records[0]["edits"]:
        target = [r["edits"][arm]["target_correct"] for r in records]
        retained = [r["edits"][arm]["original_correct"] for r in records]
        total = int(eligible.sum())
        correct = sum(bool(ok and e) for ok, e in zip(target, eligible))
        arm_scores[arm] = {"eligible_total": total, "target_correct": correct,
                           "target_success": correct/total if total else None,
                           "original_retained": sum(bool(ok and e) for ok, e in zip(retained, eligible)),
                           "all_case_target_correct": sum(target), "all_case_total": len(target)}
    comparison = paired_test([r["edits"][selection["method"]] for r,e in zip(records,eligible) if e],
                             [r["edits"]["random_"+selection["method"]] for r,e in zip(records,eligible) if e]) if eligible.any() else None
    clean_correct = sum(r["source"]["original_correct"] for r in records)
    donor_correct = sum(r["donor"]["original_correct"] for r in records)
    source_gate = (clean_correct/len(records) >= .9 and donor_correct/len(records) >= .9 and eligible.mean() >= .8)
    j = arm_scores[selection["method"]]
    causal = bool(source_gate and arm_scores["dense"]["target_success"] >= .5 and j["target_success"] >= .2
                  and comparison["difference"] >= .1 and comparison["two_sided_exact_p"] <= .05)
    predictions = {}
    for kind in ("linear", "quadratic"):
        clean, cmask = rollout(model, z[:, 0], attribute, kind)
        edited, pmask = rollout(model, zp[:, 0], attribute, kind)
        predictions[kind] = (clean, edited, cmask & pmask)
    curve = []
    for index, node in enumerate(model["nodes"]):
        probe = model["probes"][index]
        gold = country
        pred = predict(probe, z[:, index]).argmax(1)
        ppred = predict(probe, zp[:, index]).argmax(1)
        target_country = np.array([COUNTRIES.index(p["source"]["swap_to"]) for p in tasks])
        row = {"node": node, "country_correct": int((pred == gold).sum()), "total": len(gold),
               "patched_country_target_correct": int((ppred == target_country).sum()),
               "actual_delta_rms": float(np.sqrt(np.mean(delta[:, index]**2))),
               "carry_delta": error_summary(delta[:, 0], delta[:, index]), "methods": {}}
        for kind, (clean, edited, valid) in predictions.items():
            row["methods"][kind] = {"natural": error_summary(clean[:, index], z[:, index], valid[:, index]),
                                      "intervention_delta": error_summary(edited[:, index]-clean[:, index], delta[:, index], valid[:, index]),
                                      "country_correct": int(((predict(probe, clean[:, index]).argmax(1) == gold) & valid[:, index]).sum()),
                                      "patched_country_target_correct": int(((predict(probe, edited[:, index]).argmax(1) == target_country) & valid[:, index]).sum())}
        curve.append(row)
    primary = curve[7]  # after three further complete blocks: selected L+3 block output.
    q = primary["methods"]["quadratic"]["intervention_delta"]["nrmse"]
    l = primary["methods"]["linear"]["intervention_delta"]["nrmse"]
    c = primary["carry_delta"]["nrmse"]
    gain = bool(q is not None and l is not None and c is not None and q < .75 and q <= .9*l and q <= .9*c)
    classification = ("compact_causal_attention_with_polynomial_transport_signal" if causal and gain else
                      "compact_causal_attention_without_polynomial_transport_advantage" if causal else
                      "attention_concept_origin_not_causally_established")
    # The source-only holdout groups remain descriptive; no re-selection occurs.
    groups = {}
    for name, mask in (("seen_combinations", np.array([not p["source"]["composition_heldout"] for p in tasks])),
                       ("heldout_combinations", np.array([p["source"]["composition_heldout"] for p in tasks]))):
        groups[name] = {"total": int(mask.sum()), "teacher_correct": int(sum(r["source"]["original_correct"] for r,m in zip(records,mask) if m)),
                        "country_probe_correct_by_node": [int((predict(model["probes"][i],z[:,i]).argmax(1)[mask] == country[mask]).sum()) for i in range(len(indices))]}
    write_json(root/"trace.json", {"curve": curve, "groups": groups})
    result = {"study_id": STUDY_ID, "status": "completed", "confirmed": False, "classification": classification,
               "scientific_fingerprint": plan["scientific_fingerprint"], "selected_origin": selection,
               "teacher": {"correct": clean_correct, "total": len(records), "accuracy": clean_correct/len(records), "donor_correct": donor_correct},
               "source_competence_gate": bool(source_gate), "causal_origin_gate": causal,
               "polynomial_transport_signal_gate": gain, "arms": arm_scores, "J_minus_random": comparison,
               "origin_probe": origin_scores, "primary_transport": primary, "trace_sha256": file_hash(root/"trace.json"), "fit_sha256": file_hash(root/"fit.json"),
               "evaluation_manifest_sha256": file_hash(root/"evaluation_manifest.json"),
               "scope": "one frozen Qwen; eight selected J-lens directions; attention edits and contextual observable transport; no exact Groebner, full J-space or compression claim",
               "limitations": ["single_checkpoint_pilot", "source_and_donor_correlated", "development_selected_origin", "learned_transport_context_contains_attribute", "no_model_weight_training", "no_PCA_transport_comparison"]}
    write_json(root/"result.json", result)
    write_json(root/"campaign_results.json", {**result, "result_path": "result.json", "result_sha256": file_hash(root/"result.json")})
    return {"classification": classification, "teacher": result["teacher"], "arms": arm_scores, "primary_transport": primary}
