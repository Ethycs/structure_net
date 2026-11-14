"""Append-only CPU closeout for a COMPLETED all-block Qwen campaign.

Analyze only after all GPU timing is complete: it reads the retained calibration
banks to verify hashes. No stage trains, invokes a GPU, or changes a lease.
The producer's campaign_results.json remains unchanged.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from neural_architecture_lab.jspace_allblocks_hypothesis import (
    JSpaceAllBlocksRecord, STUDY_ID, SCOPE, required_stages, require, close,
)
from . import finalize_jspace_qwen_affine as common

ROOT, LOG = common.ROOT, common.LOG
DEFAULT_ROOT = ROOT / "data/experiments/jspace_qwen_parallel/20261114_v2"
DEFAULT_REPORT = ROOT / "docs/08 - Analysis/jspace/2026-11-14_jspace-allblocks-parallel.md"
DEFAULT_REGISTRY = ROOT / "data/hypothesis_registry/jspace_allblocks_20261114.json"
TYPED = ROOT / "src/neural_architecture_lab/jspace_allblocks_hypothesis.py"
TEST = ROOT / "tests/neural_architecture_lab/test_jspace_allblocks_hypothesis.py"
POSTPROCESSING = (Path(__file__), TYPED, TEST, Path(common.__file__), common.TYPED)
sha, read, write = common.sha, common.read, common.write


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def quality_from_rows(native, compact):
    require(len(native) == len(compact) == 512, "Expected 512 paired rows")
    require(len({r["identity"] for r in native}) == 512, "Duplicate evaluation identity")
    require(all(all(a[k] == b[k] for k in ("identity", "family", "answer")) for a, b in zip(native, compact)), "Unpaired evaluation")

    def summary(pairs):
        first, second = [a for a, _ in pairs], [b for _, b in pairs]
        p = common.paired(first, second); n = len(pairs)
        ac = sum(r["prediction"] == r["answer"] for r in first)
        bc = sum(r["prediction"] == r["answer"] for r in second)
        return dict(count=n, native_correct=ac, compact_correct=bc, native_accuracy=ac/n, compact_accuracy=bc/n,
                    accuracy_drop=(ac-bc)/n, net_lost_correct=ac-bc, wins=p["gains"], losses=p["losses"],
                    agreement=1-p["changed_answers"]/n, exact_binomial_p=p["exact_mcnemar_two_sided_p"])

    pairs = list(zip(native, compact))
    return {"overall": summary(pairs), "by_family": {f: summary([(a, b) for a, b in pairs if a["family"] == f])
            for f in sorted({r["family"] for r in native})}}


def verify_runtime(runtime, prompt_ids):
    """Recompute throughput from wall samples; graph-only latency cannot substitute."""
    for arm, repeats in (("cuda_graph", 3), ("eager_static", 1), ("hf_eager", 1)):
        value = runtime["timings"][arm]; rows = value["samples"]
        require(len(rows) == 8 * repeats, "Missing runtime samples")
        for repeat in range(repeats):
            selected = [r for r in rows if r["repeat"] == repeat]
            require(sorted(r["identity"] for r in selected) == sorted(prompt_ids), "Wrong runtime prompt population")
        require(all(r["arm"] == arm and r["tokens"] == 32 and math.isfinite(r["seconds"]) and r["seconds"] > 0 for r in rows), "Invalid wall-clock timing")
        seconds = sum(r["seconds"] for r in rows)
        require(close(value["total_seconds"], seconds) and close(value["tokens_per_second"], sum(r["tokens"] for r in rows)/seconds), "Runtime rate disagrees with wall samples")
    generated = runtime["generated_tokens"]
    parity = generated["eager_static"] == generated["cuda_graph"]
    require(runtime["graph_static_exact"] is parity, "Graph/static parity flag disagrees with tokens")
    require(all(len(rows) == 8 and all(len(tokens) == 32 for tokens in rows) for rows in generated.values()), "Wrong generated output lengths")


def redundancy_summary(relations):
    """Coordinates are layer-qualified; this is task reuse within each layer."""
    by_layer, totals = {}, {}
    for layer, report in relations.items():
        width = str(report["selected_anchor_width"])
        interfaces = report["metrics"]["candidates"][width]["interfaces"]
        by_layer[layer] = {}
        for name, interface in interfaces.items():
            if "redundancy" not in interface:
                continue
            r = interface["redundancy"]
            require(r["task_uses"] >= r["unique_elements"] >= 0 and r["redundant_occurrences"] == r["task_uses"]-r["unique_elements"], "Invalid task redundancy count")
            by_layer[layer][name] = r
            total = totals.setdefault(name, dict(layers=0, task_uses=0, unique_layer_qualified_elements=0,
                redundant_occurrences=0, shared_all_tasks=0, selected_not_eligible_all_tasks=0))
            total["layers"] += 1
            for key in ("task_uses", "redundant_occurrences", "shared_all_tasks", "selected_not_eligible_all_tasks"):
                total[key] += r[key]
            total["unique_layer_qualified_elements"] += r["unique_elements"]
    for total in totals.values():
        total["redundant_fraction"] = total["redundant_occurrences"] / total["task_uses"] if total["task_uses"] else 0.
    return {"by_layer": by_layer, "totals": totals,
            "scope": "Approximate common fitted relations meeting each interface's diagnostic threshold; layer-qualified coordinates, not exact neural identities or demonstrated cross-layer shared bases"}


def packed_parameter_count(run, plan, result):
    import torch
    state = torch.load(run / "evaluation/packed.pt", map_location="cpu", weights_only=True, mmap=True)
    extra = state["stack"]["_extra_state"]
    require(state["fingerprint"] == plan["fingerprint"] and extra["packed"] and extra["stripped"] and
            tuple(extra["layer_indices"]) == tuple(range(36)), "Export is not the complete packed 36-block stack")
    cfg = read(Path(plan["model"]["path"]) / "config.json")
    require(cfg["tie_word_embeddings"] is True, "Frozen tied vocabulary required")
    index = read(Path(plan["model"]["path"]) / "model.safetensors.index.json")["weight_map"]
    names = ("qkv_weight", "qkv_bias", "attention_output", "gate_up_weight", "mlp_output", "polynomial_bias")
    count = cfg["vocab_size"] * cfg["hidden_size"] + sum(t.numel() for t in state["norm"].values())
    for layer in range(36):
        count += sum(state["stack"][f"blocks.{layer}.{name}"].numel() for name in names)
        if f"model.layers.{layer}.self_attn.o_proj.bias" in index:
            count += state["stack"][f"blocks.{layer}.attention_output_bias"].numel()
    exported = sum(v.numel()*v.element_size() for group in (state["stack"], state["norm"]) for v in group.values() if isinstance(v, torch.Tensor))
    external_readout = 2 * cfg["vocab_size"] * cfg["hidden_size"]
    difference = result["results"]["compact"]["tensor_bytes"] - exported - external_readout
    require(difference >= 0, "Reported deployed size omits exported tensors or tied vocabulary")
    return {"model_parameters": count, "exported_tensor_bytes": exported, "external_tied_readout_bytes": external_readout,
            "other_resident_buffer_bytes": difference, "scope": "Packed learned tensors, original tied vocabulary once, final RMS gain; buffers excluded from parameter count"}


def reused_stage_provenance(run, plan, name, manifest):
    """Reused v1 bytes keep their production identity under a v2 wrapper."""
    if "reused_from" not in manifest:
        return {}
    require(plan["schema"] == "jspace.qwen-parallel.v2" and manifest.get("provenance_kind") == "verified_immutable_reuse",
            "Undeclared immutable artifact reuse")
    reuse, reference = plan["reuse"], manifest["reused_from"]
    old_root = Path(reuse["root"]); relative = name + "/manifest.json"
    require(reference == {"root": str(old_root), "source_fingerprint": reuse["source_fingerprint"],
            "manifest": relative, "manifest_sha256": reuse["manifest_sha256"][relative]}, "Reused stage lineage changed")
    require(sha(old_root / relative) == reference["manifest_sha256"], "Original reused manifest changed")
    old_manifest, old_plan = read(old_root / relative), read(old_root / "plan.json")
    require(old_plan["fingerprint"] == reuse["source_fingerprint"] == digest({k: v for k, v in old_plan.items() if k != "fingerprint"}),
            "Original reused producer fingerprint changed")
    require(old_manifest["status"] == "completed" and old_manifest["fingerprint"] == old_plan["fingerprint"] and
            old_manifest["artifacts"] == manifest["artifacts"] and
            manifest["original_stage_metadata"] == {k: v for k, v in old_manifest.items() if k != "artifacts"},
            "Reused artifact production metadata changed")
    for artifact in manifest["artifacts"]:
        require((run / artifact).samefile(old_root / artifact), "Reused immutable artifact is no longer the original hardlink")
    hashes = {str(old_root / relative): reference["manifest_sha256"]}
    for source, checksum in old_plan["source_sha256"].items():
        hashes[source] = checksum
        hashes[str(old_root / "producer" / Path(source).relative_to(ROOT))] = checksum
    hashes.update(old_plan["input_sha256"])
    return hashes


def analyze(run):
    plan, result = read(run / "plan.json"), read(run / "campaign_results.json")
    require(plan["fingerprint"] == digest({k: v for k, v in plan.items() if k != "fingerprint"}), "Changed frozen plan")
    require(result == read(run / "evaluation/result.json"), "Producer aggregate and evaluation result disagree")
    manifests, hashes, verification = {}, {}, {}
    for name in required_stages():
        path = run / name / "manifest.json"; manifest = read(path)
        require(manifest["status"] == "completed" and manifest["fingerprint"] == plan["fingerprint"], "Incomplete/mismatched stage: " + name)
        manifests[name] = manifest; verification[str(path)] = sha(path)
        hashes.update(reused_stage_provenance(run, plan, name, manifest))
        for relative, value in manifest["artifacts"].items():
            target = str(run / relative)
            require(target not in hashes or hashes[target] == value, "Conflicting artifact hash")
            hashes[target] = value
    if plan["schema"] == "jspace.qwen-parallel.v2":
        expected_reuse = set(plan["reuse"]["manifest_sha256"])
        require({name + "/manifest.json" for name, manifest in manifests.items() if "reused_from" in manifest} == expected_reuse,
                "Declared reused stages differ from manifests")
    for path, value in plan["source_sha256"].items():
        hashes[path] = value
        hashes[str(run / "producer" / Path(path).relative_to(ROOT))] = value
    hashes.update(plan["input_sha256"])
    hashes.update({str(Path(plan["model"]["path"]) / name): value for name, value in plan["model"]["files"].items()})
    for path, value in hashes.items():
        require(sha(path) == value, "Changed artifact/input/producer: " + path)
    inputs = read(run / "evaluation/inputs.json")
    require(inputs["fingerprint"] == plan["fingerprint"] and inputs["task_sha256"] == sha(run / "tasks.json"), "Evaluation input fingerprint changed")
    for relative, value in inputs["artifacts"].items():
        require(sha(run / relative) == value, "Evaluation dependency changed: " + relative)
    tasks = read(run / "tasks.json"); cases = tasks["splits"]["evaluation"]
    require(digest(tasks) == plan["task_digest"] and len(cases) == 512, "Changed fresh task manifest")
    excluded = set(tasks["previous_identities"])
    excluded.update(c["identity"] for s, rows in tasks["splits"].items() if s != "evaluation" for c in rows)
    require(len({c["identity"] for c in cases}) == 512 and not excluded.intersection(c["identity"] for c in cases), "Fresh split leakage")
    require(len(tasks["splits"]["train"]) == 512 and len(tasks["splits"]["development"]) == 64, "Training/development budget changed")
    native, compact = (read(run / "evaluation" / (arm + "_fresh.json")) for arm in ("native_before", "compact"))
    require(all(all(a[k] == b[k] for k in ("identity", "family", "answer")) for a, b in zip(native, cases)), "Scored identities/labels differ from frozen tasks")
    require(quality_from_rows(native, compact) == result["quality"], "Quality summary differs from raw records")
    require((read(run / "evaluation/reloaded_fresh.json") == compact) is result["packed_reload_exact"], "Reload claim differs from records")
    require(sha(run / "evaluation/packed.pt") == result["packed_sha256"] and (run / "evaluation/packed.pt").stat().st_size == result["packed_file_bytes"], "Export changed")
    prompts = [c for family in sorted(result["quality"]["by_family"]) for c in
               [r for r in tasks["splits"]["development"] if r["family"] == family][:2]]
    require(len(prompts) == 8 and all(len(c["input_ids"]) + 32 <= 416 for c in prompts), "Runtime prompt support changed")
    for arm, values in result["results"].items():
        require(values["runtime"] == read(run / "evaluation" / (arm + "_timing.json")), "Runtime artifact differs")
        verify_runtime(values["runtime"], [c["identity"] for c in prompts])
    training, relations, teachers = {}, {}, {}
    for group in range(6):
        prefix = run / "groups" / f"group_{group:02d}"
        path = prefix / "recovery/training.json"; training[str(group)] = read(path); verification[str(path)] = sha(path)
        require(sha(prefix / "recovered_unpacked.pt") == sha(prefix / "recovery/recovered_unpacked.pt"), "Accepted group export differs")
        teacher = read(prefix / "capture/metadata.json")["teacher_fingerprint"]
        expected_teacher = digest(plan["model"] if group == 0 else sha(run / "groups" / f"group_{group-1:02d}" / "recovered_unpacked.pt"))
        require(teacher == expected_teacher, "Broken sequential accepted-teacher chain")
        teachers[str(group)] = teacher
        for layer in range(30-group*6, 36-group*6):
            path = prefix / f"block_{layer:02d}/relations.json"; relation = read(path)
            require(relation["teacher_fingerprint"] == teacher and relation["layer"] == layer, "Relation teacher/layer provenance changed")
            require(relation["fixed_anchor_width"] == plan["config"]["anchors"] and relation["removed_channels"] == plan["config"]["removed_channels"], "Polynomial compilation budget changed")
            relations[str(layer)] = relation; verification[str(path)] = sha(path)
    value = {"schema_version": "jspace.allblocks.closeout.v2", "created_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan, "result": result, "training": training, "manifests": manifests,
        "teacher_chain": teachers, "relations": relations, "redundancy": redundancy_summary(relations),
        "parameter_count": packed_parameter_count(run, plan, result), "artifact_sha256": hashes,
        "analysis_source_sha256": {str(p): sha(p) for p in POSTPROCESSING},
        "verification_sha256": {**verification, **{str(run / p): sha(run / p) for p in
            ("plan.json", "tasks.json", "campaign_results.json", "evaluation/result.json", "evaluation/inputs.json")}},
        "integrity": {"completed_stages": len(manifests), "artifact_hashes_verified": len(hashes),
            "producer_files_verified_live_and_archived": len(plan["source_sha256"]), "independent_confirmations": 0,
            "fresh_split_disjoint": True, "reload_scope": "Fresh model construction and strict packed load in the evaluator process; not a fresh-process inference test"}}
    record = JSpaceAllBlocksRecord.from_evidence(value)
    value["record"] = record.to_dict()
    archive = run / "closeout_source"; archive.mkdir(exist_ok=False)
    for path in POSTPROCESSING:
        target = archive / path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    write(run / "closeout_analysis.json", value)
    return {"analysis": str(run / "closeout_analysis.json"), "sha256": sha(run / "closeout_analysis.json"), "classification": record.classification}


def expected(run):
    value = read(run / "closeout_analysis.json")
    for path, checksum in {**value["verification_sha256"], **value["analysis_source_sha256"]}.items():
        require(sha(path) == checksum, "Closeout provenance changed: " + path)
    require(value["result"] == read(run / "campaign_results.json"), "Analyzed result changed")
    record = JSpaceAllBlocksRecord.from_evidence(value)
    require(record.to_dict() == value["record"], "Typed record changed")
    return value, record


def report(run, path):
    value, r = expected(run); result = value["result"]
    lines = ["# All-block Qwen compression", "", f"Status: completed. Study: {STUDY_ID}. Classification: **{r.classification}**. `confirmed=false`.", "",
        f"Compact accuracy is **{r.compact_correct}/512 ({100*r.compact_correct/512:.3f}%)**, versus native **{r.native_correct}/512**. "
        f"Deployed tensor reduction is **{100*r.size_reduction:.3f}%**; whole-model throughput is **{r.compact_graph_tokens_per_second:.3f} tokens/s**, "
        f"or **{r.whole_model_speedup:.4f}×** the mean of the native before/after controls. "
        f"The combined target {'passed on this restricted benchmark' if r.targets_met else 'was not met'}.", "",
        f"Producer root: `{run}`. Scientific fingerprint: `{r.scientific_fingerprint}`. "
        f"Producer aggregate SHA-256: `{sha(run/'campaign_results.json')}`. Closeout evidence SHA-256: `{sha(run/'closeout_analysis.json')}`.", "",
        "| Joint gate | Required | Result |", "| --- | --- | --- |"]
    labels = {"overall_quality": f"≤{100*r.quality_thresholds['maximum_accuracy_drop']:g}pp aggregate accuracy loss", "every_family_quality": "≤4/128 net loss per family",
              "deployed_size": "≥38% fewer tensor bytes", "whole_model_speed": "≥1.4× whole-model rate",
              "packed_reload_exact": "exact packed reload records", "graph_static_tokens_exact": "graph/static token parity"}
    lines.extend(f"| {name} | {labels[name]} | {'pass' if passed else 'fail'} |" for name, passed in r.gates.items())
    if r.protocol_schema.endswith(".v2"):
        reuse = value["plan"]["reuse"]
        lines += ["", "The user authorized a five-percentage-point aggregate accuracy budget before v2 recovery and held-out scoring. "
            "Per-family results and the original one-point budget remain diagnostics, with no per-family veto of the aggregate gate. "
            f"Original 1pp diagnostic: {'pass' if r.quality_diagnostics['original_one_percentage_point_quality'] else 'fail'}; "
            f"all-family 4/128 diagnostic: {'pass' if r.quality_diagnostics['every_family_quality'] else 'fail'}.", "",
            f"V2 reuses group-0 capture, basis and fits {reuse['completed_relation_layers']} from `{reuse['root']}` "
            f"(original fingerprint `{reuse['source_fingerprint']}`). Wrapper manifests retain original producer identity and verify immutable hardlinks. "
            "The stopped v1 had no recovery training or held-out scoring. The numerical-rank amendment uses the actual sketch-SVD dimensions; "
            "its diagnostic, prior failure and protocol changes are frozen in the v2 plan."]
    lines += ["", f"Native deployed tensors: {r.native_tensor_bytes:,} bytes; compact: {r.compact_tensor_bytes:,} bytes. "
              "These include all parameters/buffers and the tied vocabulary once, excluding serving caches and allocator overhead. "
              "The packed export is a stack-plus-final-norm delta requiring the pinned native embedding/readout.", "",
              "| Full model | Graph tokens/s | Uncaptured static-cache tokens/s | HF eager tokens/s |", "| --- | ---: | ---: | ---: |"]
    for arm, data in result["results"].items():
        t = data["runtime"]["timings"]
        lines.append(f"| {arm} | {t['cuda_graph']['tokens_per_second']:.4f} | {t['eager_static']['tokens_per_second']:.4f} | {t['hf_eager']['tokens_per_second']:.4f} |")
    lines += ["", "The primary wall-clock rate includes dynamic prefill, static-cache transfer and 32 generated tokens. Only decode is CUDA-graph captured; no graph-prefill claim is made. "
              "The denominator is the arithmetic mean of the native before/after token rates. Capture setup is excluded. Local kernel timing cannot satisfy the whole-model gate.", "",
              "| Task | Native correct /128 | Compact correct /128 | Corrected | Newly incorrect |", "| --- | ---: | ---: | ---: | ---: |"]
    for family, q in result["quality"]["by_family"].items():
        lines.append(f"| {family} | {q['native_correct']} | {q['compact_correct']} | {q['wins']} | {q['losses']} |")
    q = result["quality"]["overall"]
    lines += ["", f"Overall paired gains/losses: {q['wins']}/{q['losses']}; agreement {100*q['agreement']:.3f}%; exact paired p={q['exact_binomial_p']:.6g}. "
              "These are descriptive results, not noninferiority or equivalence proof. The 512 final identities are disjoint from all excluded prior pools and current training/development.", "",
              "All six groups used 512 training and 64 development cases, with output-only .7*T²*KL+.3*CE at T=2, persistent FP32 CPU Adam masters, declared BF16 model parameters/gradients, and block checkpointing. "
              "Every selected checkpoint is the minimum packed-development CE, including epoch zero. The accepted teacher advances between groups; original-teacher drift remains diagnostic.", "",
              "| Group | Added layers | Selected epoch | Initial CE | Selected CE | Final CE | Original-teacher KL at selection |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for group in range(6):
        t = value["training"][str(group)]
        lines.append(f"| {group} | {30-6*group}–{35-6*group} | {t['selected_epoch']} | {t['initial']['ce']:.6f} | {t['selected']['ce']:.6f} | {t['history'][-1]['development']['ce']:.6f} | {t['selected'].get('original_kl')} |")
    lines += ["", "Complete packed-development CE trajectory:", "", "| Epoch | Group 0 | Group 1 | Group 2 | Group 3 | Group 4 | Group 5 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for epoch in range(13):
        lines.append("| " + str(epoch) + " | " + " | ".join(f"{r.development_ce_by_group[str(g)][epoch]:.6f}" for g in range(6)) + " |")
    lines += ["", "All 36 layer relation reports and full task-overlap matrices are retained in closeout_analysis.json. Redundancy totals use layer-qualified elements; matching a channel index across layers is not treated as the same concept.", "",
              "| Interface | Layers | Task uses | Unique layer-qualified elements | Repeated uses | Repeated fraction | Shared across all tasks |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, total in value["redundancy"]["totals"].items():
        lines.append(f"| {name} | {total['layers']} | {total['task_uses']} | {total['unique_layer_qualified_elements']} | {total['redundant_occurrences']} | {100*total['redundant_fraction']:.3f}% | {total['shared_all_tasks']} |")
    lines += ["", "Eligibility is a diagnostic threshold on approximate fitted relations. An exact graph-ideal certificate does not establish an identity of the original neural network.", "",
              f"Integrity: {value['integrity']['completed_stages']} completed stage manifests and {value['integrity']['artifact_hashes_verified']} verified artifact/input/source paths. "
              + value["integrity"]["reload_scope"] + ".", "", SCOPE,
              "Accuracy inputs are capped at 384 tokens; the compiled position tables support 416 positions, with 32 generated tokens in the benchmark. No unrestricted long-context serving claim is made."]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    return {"report": str(path), "sha256": sha(path)}


def metadata(run, record):
    return {"hypothesis_id": STUDY_ID, "tested": True, "confirmed": False, "confirmation_status": record.classification,
        "accuracy": record.compact_correct/512, "primary_metric": record.compact_correct/512,
        "correct": record.compact_correct, "count": 512, "targets_met": record.targets_met,
        "whole_model_speedup": record.whole_model_speedup, "tensor_reduction": record.size_reduction,
        "graph_tokens_per_second": record.compact_graph_tokens_per_second,
        "protocol_schema": record.protocol_schema,
        "maximum_accuracy_drop": record.quality_thresholds["maximum_accuracy_drop"],
        "family_quality_is_gate": record.quality_thresholds.get("family_quality_is_gate", True),
        "accuracy_scope": SCOPE,
        "scientific_fingerprint": record.scientific_fingerprint, "analysis_sha256": sha(run / "closeout_analysis.json")}


def readback(run, registry, key=None):
    _, record = expected(run)
    if key is None:
        stored = read(registry); key = stored["storage"]["result_hash"]
        require(stored["analysis_sha256"] == sha(run / "closeout_analysis.json") and stored["report_sha256"] == sha(stored["report_path"]), "Registry/report provenance changed")
    db = common.logger(); rows = db.experiments_collection.get(ids=[key], include=["metadatas"])
    require(rows["ids"] == [key] and all(rows["metadatas"][0].get(k) == v for k, v in metadata(run, record).items()), "Experiment readback mismatch")
    rows = db.hypotheses_collection.get(ids=[STUDY_ID], include=["metadatas"])
    require(rows["ids"] == [STUDY_ID], "Hypothesis missing")
    h = rows["metadatas"][0]
    require(h.get("tested") is True and h.get("confirmed") is False and h.get("confirmation_status") == record.classification and
            h.get("evidence_count") == 1 and json.loads(h["direct_result_hashes_json"]) == [key], "Hypothesis status/evidence link mismatch")
    return {"readback_verified": True, "result_hash": key, **metadata(run, record)}


def store(run, registry, report_path):
    require(not registry.exists() and report_path.is_file(), "New registry and stabilized report required")
    value, record = expected(run)
    from neural_architecture_lab.core import ExperimentResult
    hypothesis = dict(id=STUDY_ID, name="All-block benchmark-derived Qwen compression with staged joint recovery", category="compression",
        tested=True, confirmed=False, evidence_count=1, direct_experiment_count=1, confirmation_status=record.classification,
        tested_scope=SCOPE, description="Joint 38% tensor reduction, 1.4x whole-model speed, and quality gates; six sequential groups are one campaign.")
    metrics = {k: v for k, v in metadata(run, record).items() if k in ("accuracy", "whole_model_speedup", "tensor_reduction", "graph_tokens_per_second")}
    experiment = ExperimentResult(experiment_id="jspace-allblocks-" + record.scientific_fingerprint[:16], hypothesis_id=STUDY_ID,
        metrics=metrics, primary_metric=record.compact_correct/512, model_architecture=[value["plan"]["config"]["rank"]]*36,
        model_parameters=value["parameter_count"]["model_parameters"],
        training_time=sum(t["elapsed_seconds"] for t in value["training"].values()),
        model_checkpoint=str(run / "evaluation/packed.pt"), observations=[SCOPE, record.classification, str(report_path),
            "Packed delta requires the pinned native tied embedding/readout; exact graph certificate is for the fitted surrogate."])
    db = common.logger(); prior = db.hypotheses_collection.get(ids=[STUDY_ID], include=["metadatas"])
    require(not prior["ids"] or prior["metadatas"][0].get("tested") is not True, "Refusing to overwrite tested evidence")
    db.log_hypothesis(hypothesis); key = db.log_experiment_result(experiment)
    db.experiments_collection.update(ids=[key], metadatas=[metadata(run, record)])
    db.hypotheses_collection.update(ids=[STUDY_ID], metadatas=[{"direct_result_hashes_json": json.dumps([key]),
        "analysis_sha256": sha(run / "closeout_analysis.json"), "report_path": str(report_path)}])
    receipt = readback(run, registry, key)
    write(registry, {"schema_version": "nal.meta-hypothesis.v1", "hypothesis": hypothesis, "record": record.to_dict(),
        "storage": receipt, "analysis_sha256": sha(run / "closeout_analysis.json"), "report_path": str(report_path), "report_sha256": sha(report_path)})
    return receipt


def closeout(run, registry):
    value, record = expected(run); receipt = read(run / "allblocks_registry_readback.json")
    require(receipt == readback(run, registry), "Missing independent readback")
    stored = read(registry)
    event = {"schema_version": "jspace.research-log.v1", "id": "jspace-allblocks-" + record.scientific_fingerprint,
        "event": "allblocks_campaign_finalized", "study_id": STUDY_ID, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "confirmed": False, "classification": record.classification, "targets_met": record.targets_met,
        "record": record.to_dict(), "new_experimental_evidence": True, "independent_confirmations": 0,
        "registry_result_id": receipt["result_hash"], "registry_sha256": sha(registry),
        "analysis_sha256": sha(run / "closeout_analysis.json"), "report_path": stored["report_path"], "report_sha256": stored["report_sha256"]}
    with LOG.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX); stream.seek(0)
        require(not any(json.loads(line).get("id") == event["id"] for line in stream if line.strip()), "Research event already logged")
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
    write(run / "allblocks_finalization.json", {"status": "completed", "confirmed": False, "record": record.to_dict(),
        "registry_readback": receipt, "research_event": event, "analysis_source_sha256": value["analysis_source_sha256"]})
    return {"finalized": True, "registry": str(registry), "finalization_sha256": sha(run / "allblocks_finalization.json")}


if __name__ == "__main__":
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Start with CUDA_VISIBLE_DEVICES='' for CPU-only closeout")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("analyze", "report", "store", "readback", "closeout"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args(); run = args.root.resolve(); registry = args.registry.resolve()
    if args.stage == "analyze": output = analyze(run)
    elif args.stage == "report": output = report(run, args.report.resolve())
    elif args.stage == "store": output = store(run, registry, args.report.resolve())
    elif args.stage == "readback":
        output = readback(run, registry); write(run / "allblocks_registry_readback.json", output)
    else: output = closeout(run, registry)
    print(json.dumps(output, allow_nan=False))
