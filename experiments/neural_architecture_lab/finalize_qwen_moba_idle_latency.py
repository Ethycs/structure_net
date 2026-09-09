#!/usr/bin/env python3
"""Audit and report the idle latency diagnostic, retaining the earlier evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from experiments.neural_architecture_lab.audit_qwen_moba_feasibility import sha256, strict_read
from neural_architecture_lab.qwen_moba_idle_latency_hypothesis import IdleLatencyEvidence

REPORT = Path("docs/08 - Analysis/2026-09-09_qwen8b-moba-idle-latency.md")


def save(path, data):
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError("preserve existing different artifact")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def audit(path):
    result = strict_read(path)
    producer = strict_read(path.parent / "producer.json")
    identity = {key: value for key, value in producer.items() if key != "fingerprint"}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if fingerprint != producer["fingerprint"] or fingerprint != result["producer_fingerprint"]:
        raise ValueError("producer fingerprint mismatch")
    snapshots = {p.name.split("_", 1)[1]: p for p in (path.parent / "producer_source").iterdir()}
    for source, digest in producer["sources"].items():
        if sha256(snapshots[Path(source).name]) != digest:
            raise ValueError("source snapshot changed")
    if result["status"] != "completed" or result["classification"] != "idle_gpu_latency_diagnostic" or result["summary"] != {"requested": 10, "completed": 10}:
        raise ValueError("requires complete uncontended diagnostic")
    if not result["validity"]["passed"] or result["frozen_weights_before"] != result["frozen_weights_after"]:
        raise ValueError("graph or frozen weight check failed")
    rows = {(c["length"], c["variant"]): c for c in result["cells"]}
    if set(rows) != {(n, v) for n in producer["lengths"] for v in producer["variants"]}:
        raise ValueError("incomplete or duplicate cells")
    for (length, variant), cell in rows.items():
        if strict_read(path.parent / "cells" / f"n{length}_{variant}.json") != cell:
            raise ValueError("cell artifact mismatch")
        if not cell["no_external_contention_detected"] or len(cell["samples_ms"]) != 5:
            raise ValueError("contention or sample count violation")
        if cell["median_ms"] != statistics.median(cell["samples_ms"]):
            raise ValueError("timing median mismatch")
        parity = cell["scheduling_parity"]
        if min(parity["heads_cosine_mean"], parity["ov_cosine_mean"]) < .9999:
            raise ValueError("scheduling changed representations materially")
    return result, producer, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("data/experiments/qwen_moba_prefix/20260909_idle_latency_v1/campaign_results.json"))
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    result, producer, rows = audit(args.campaign)
    earlier = Path("data/experiments/qwen_moba_prefix")
    prior = {stage: strict_read(earlier / stage / "producer.json")["gpu_before"] for stage in ("20260909_shared_screen_v1", "20260909_shared_final_v1")}
    lines = ["# Qwen MoBA: idle latency and correction", "", "Date: 2026-09-09. Systems diagnostic, reused validation input.", "",
             "**The approximately two-second latency was real; blaming it on another GPU job was incorrect.** Both earlier screen/final producer files show about 11,511 MiB free and 1% GPU utilization before model loading. The job that had blocked earlier pilots was already absent. The `shared_gpu` names describe the conservative memory-limited protocol, not established contention during those final measurements. The original numerical results remain unchanged.", "",
             "The new controlled run starts with 11,618 MiB free and 0% utilization and records external GPU processes before/after every cell. No external contention was detected. All weights are frozen; the architecture and FP16/FP32 readout contract are unchanged. MoBA remains the official implementation with block1024/top2 across six attention layers.", "",
             "## Measured latency", "", "Five rotated/alternating CUDA-event samples after one warmup per path. Medians below exclude loading, disk I/O and component-profiling hooks. Complete-sequence MLP execution changes batching, not which layers or weights are used.", "",
             "| Execution | 8K tokens | 16K tokens |", "| --- | ---: | ---: |"]
    labels = {"full_mlp2048": "Full attention, MLP chunks of 2048", "full_mlpwhole": "Full attention, complete-sequence MLP",
              "moba_heads4_mlp2048": "MoBA, four heads per call, MLP chunks of 2048",
              "moba_heads32_mlp2048": "MoBA, all 32 heads, MLP chunks of 2048",
              "moba_heads32_mlpwhole": "MoBA, all 32 heads, complete-sequence MLP"}
    for variant, label in labels.items():
        lines.append(f"| {label} | {rows[8192, variant]['median_ms']:.2f} ms | {rows[16384, variant]['median_ms']:.2f} ms |")
    full, sparse = rows[16384, "full_mlpwhole"]["median_ms"], rows[16384, "moba_heads32_mlpwhole"]["median_ms"]
    ratio = full / sparse
    lines += ["", f"With matched complete-sequence MLP batching, MoBA is **{ratio:.3f}× faster at 16K**, reducing latency by **{100 * (1-sparse/full):.1f}%**. At 8K it remains slower than matched full attention. Removing four-head chunking saves about {rows[16384, 'moba_heads4_mlp2048']['median_ms']-rows[16384, 'moba_heads32_mlp2048']['median_ms']:.1f} ms at 16K; changing MLP batching saves another {rows[16384, 'moba_heads32_mlp2048']['median_ms']-sparse:.1f} ms. These changes do not turn this into a low-latency prefix.", "", "## Where time goes", "",
              "Separate CUDA-event instrumentation at 16K; these are component estimates, not the primary timing samples. Ranges include CPU launch gaps within the measured component. The MLP column includes only the first five MLPs; layer six's MLP is omitted.", "",
              "| Execution | Attention | Five MLPs | Q/K/V/output projections | Other | Total |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for variant in ("full_mlpwhole", "moba_heads32_mlpwhole"):
        p = result["instrumented_profiles"][variant]
        lines.append(f"| {labels[variant]} | {p['attention_ms']:.1f} ms | {p['mlp_ms']:.1f} ms | {p['projections_ms']:.1f} ms | {p['other_ms']:.1f} ms | {p['total_ms']:.1f} ms |")
    p = result["instrumented_profiles"]["moba_heads32_mlpwhole"]
    lines += ["", f"Attention accounts for about {100*p['attention_ms']/p['total_ms']:.1f}% of the MoBA prefix time. The MLPs, projections and remaining work take about {(p['total_ms']-p['attention_ms'])/1000:.2f} seconds. Keeping those costs fixed, even eliminating the remaining attention time would yield at most {p['total_ms']/(p['total_ms']-p['attention_ms']):.2f}× additional speedup. This bound is an arithmetic inference from the instrumented run, not a measured alternative implementation.", "",
              "The benchmark recomputes a complete 16K-token prefix. It is not a measurement of one new summary or a cached incremental update, nor of an isolated final-layer attention call. MoBA alone provides a modest improvement for this prefix workload.", "", "## Validity and limits", "",
              "All 10 execution/context cells completed. Source snapshots and fingerprints, exact input window, model hashes, paired samples, memory usage and GPU process state are retained. The real-prefix graph check passed; weight hashes match before/after. Scheduling variants retain mean head/output cosine similarity above 0.9999 against their corresponding chunked execution. This checks scheduling equivalence, not application accuracy or exact numerical identity.", "",
              "Only the already-used Wikipedia validation window was timed. This is not another ten-window quality confirmation. No weights were trained and no 95% application-accuracy claim follows. The earlier readout-preservation result remains separate from this latency correction.", "", "## Provenance", "",
              f"- Campaign: `{args.campaign}`", f"- SHA-256: `{sha256(args.campaign)}`", f"- Producer fingerprint: `{producer['fingerprint']}`",
              "- Entry point: `experiments/structure_net/qwen_moba_idle_latency.py`", "- Protocol: `docs/07 - Status Reports/2026-09-09_qwen8b-moba-idle-latency.md`",
              "- Two focused contention/profiling tests passed; the existing prefix numerical controls are reused.", "", "Earlier startup telemetry (preserved evidence):", "", "```json", json.dumps(prior, indent=2), "```", ""]
    report = "\n".join(lines)
    if REPORT.exists() and REPORT.read_text() != report:
        raise ValueError("preserve existing different report")
    if not REPORT.exists():
        REPORT.write_text(report)
    evidence = IdleLatencyEvidence(str(args.campaign), sha256(args.campaign), str(REPORT), sha256(REPORT), 10)
    if args.register:
        from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
        logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses", queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected", enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
        old_path = Path("data/hypothesis_registry/qwen3-8b-prefix6-moba-feasibility-v1.json")
        parent = strict_read(old_path)["hypotheses"][0]
        parent["description"] = "Official MoBA on frozen Qwen3-8B prefix. Correction: the shared_gpu protocol used memory limits, but screen/final startup telemetry was already idle. Idle follow-up measures about 2.01s full versus 1.79s MoBA at 16K; MLPs dominate. Application accuracy remains unmeasured."
        parent["latency_correction_report"] = str(REPORT)
        parent["latency_correction_report_sha256"] = sha256(REPORT)
        records = [parent, evidence.record()]
        for record in records:
            logger.log_hypothesis(record)
            readback = logger.hypotheses_collection.get(ids=[record["id"]], include=["metadatas"])
            if readback["ids"] != [record["id"]] or any(readback["metadatas"][0].get(k) != record[k] for k in ("tested", "confirmed", "confirmation_status", "description")):
                raise RuntimeError("exact registry readback failed")
        found = logger.hypotheses_collection.query(query_texts=["Qwen MoBA idle latency correction MLP projections two seconds"], n_results=10)
        if evidence.record()["id"] not in found["ids"][0]:
            raise RuntimeError("semantic registry readback failed")
        registry = {"schema": "nal.qwen-moba-latency-correction.v1", "supersedes_description_of": str(old_path),
                    "original_registry_sha256": sha256(old_path), "hypotheses": records,
                    "exact_readback_verified": True, "semantic_readback_verified": True}
        save(Path("data/hypothesis_registry/qwen-moba-20260909-latency-correction.json"), registry)
        save(args.campaign.parent / "registry_correction_readback.json", registry)
    print(json.dumps({"report": str(REPORT), "campaign_sha256": evidence.campaign_sha256, "report_sha256": evidence.report_sha256, "matched_16k_speedup": ratio, "registered": args.register}))


if __name__ == "__main__":
    main()
