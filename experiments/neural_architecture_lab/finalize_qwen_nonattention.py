#!/usr/bin/env python3
"""Audit frozen-prefix acceleration and publish a conservative local record."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from experiments.neural_architecture_lab.audit_qwen_moba_feasibility import sha256, strict_read
from experiments.neural_architecture_lab.finalize_qwen_moba_idle_latency import save
from neural_architecture_lab.qwen_nonattention_hypothesis import NonattentionEvidence

ROOT = Path("data/experiments/qwen_moba_prefix")
REPORT = Path("docs/08 - Analysis/2026-09-09_qwen8b-nonattention-acceleration.md")
NAMES = ("fused_fp16", "int8_mlp6", "int8_mlp3", "int8_gate_up6")


def gate(metrics, approximate):
    cosine, relative, pearson = (.995, .05, .995) if approximate else (.9999, .005, .999)
    return (min(metrics["heads_cosine_mean"], metrics["ov_cosine_mean"]) >= cosine
            and max(metrics["heads_relative_l2"], metrics["ov_relative_l2"]) <= relative
            and metrics["change_pearson"] is not None and metrics["change_pearson"] >= pearson
            and metrics["top_quarter_change_f1"] >= .95)


def audit(path):
    result = strict_read(path)
    producer = strict_read(path.parent / "producer.json")
    identity = {k: v for k, v in producer.items() if k != "fingerprint"}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if digest != producer["fingerprint"] or digest != result["producer_fingerprint"]:
        raise ValueError("producer fingerprint mismatch")
    snapshots = {p.name.split("_", 1)[1]: p for p in (path.parent / "producer_source").iterdir()}
    for source, expected in producer["sources"].items():
        if sha256(snapshots[Path(source).name]) != expected:
            raise ValueError("source snapshot mismatch")
    if result["status"] != "completed" or not result["validity"]["passed"] or result["frozen_weights_before"] != result["frozen_weights_after"]:
        raise ValueError("invalid completion or frozen-prefix control")
    if result["application_accuracy_tested"]:
        raise ValueError("this experiment does not measure application accuracy")
    regression = result["stage"] == "regression"
    expected_cells = ({(s, seed, 16384, producer["selection_artifact"]["contents"]["selection"]["execution"]["name"])
                       for s in ("wikipedia", "bnc_spoken") for seed in (7, 17, 29, 41, 53)} if regression else
                      {(s, 7, n, v) for s in ("wikipedia", "bnc_spoken") for n in (8192, 16384) for v in NAMES})
    actual_cells = {(r["source"], r["seed"], r["length"], r["variant"]) for r in result["cells"]}
    if actual_cells != expected_cells or len(result["cells"]) != len(expected_cells):
        raise ValueError("incomplete or duplicate grid")
    for row in result["cells"]:
        name = f"{row['source']}_s{row['seed']}_n{row['length']}_{row['variant']}.json"
        if strict_read(path.parent / "cells" / name) != row:
            raise ValueError("cell mismatch")
        schedule = producer["schedules"][row["source"]]
        start = schedule["final_starts"][str(row["seed"])] if regression else schedule["validation_start"]
        if row["start"] != start:
            raise ValueError("input schedule mismatch")
        expected_gate = False
        if row["status"] == "completed":
            for arm, values in row["samples_ms"].items():
                if len(values) != 5 or min(values) <= 0 or statistics.median(values) != row["medians_ms"][arm]:
                    raise ValueError("timing sample mismatch")
            if row["speedup_vs_moba"] != row["medians_ms"]["moba"] / row["medians_ms"]["candidate"] or row["speedup_vs_full"] != row["medians_ms"]["full"] / row["medians_ms"]["candidate"]:
                raise ValueError("speedup mismatch")
            fidelity = gate(row["metrics_vs_moba"], row["execution"]["int8_threshold"] is not None)
            if row["fidelity_pass"] != fidelity:
                raise ValueError("fidelity gate mismatch")
            expected_gate = fidelity and row["speedup_vs_moba"] >= 1.05 and row["no_external_contention_detected"]
        if row["gate"] != expected_gate:
            raise ValueError("joint gate mismatch")
    if result["summary"] != {"requested": len(expected_cells), "completed": sum(r["status"] == "completed" for r in result["cells"]),
                             "joint_passes": sum(r["gate"] for r in result["cells"])}:
        raise ValueError("summary mismatch")
    if regression:
        expected = "execution_regression_supported" if all(r["gate"] for r in result["cells"]) else "execution_regression_not_confirmed"
        if result["classification"] != expected or not result["reused_regression_population"]:
            raise ValueError("regression classification mismatch")
    else:
        choices = []
        for index, variant in enumerate(NAMES):
            rows = [r for r in result["cells"] if r["length"] == 16384 and r["variant"] == variant]
            if all(r["gate"] for r in rows):
                choices.append((statistics.mean(r["speedup_vs_moba"] for r in rows),
                                statistics.mean(r["metrics_vs_moba"]["change_pearson"] for r in rows), -index, rows[0]["execution"]))
        selected = {"execution": max(choices, key=lambda r: r[:3])[3], "length": 16384} if choices else None
        if selected != result["selection"] or result["classification"] != ("execution_candidate_selected" if selected else "no_execution_candidate_passed"):
            raise ValueError("selection mismatch")
    return result, producer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, default=ROOT / "20260909_nonattention_screen_v1/campaign_results.json")
    parser.add_argument("--regression", type=Path)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    screen, screen_producer = audit(args.screen)
    regression, regression_producer = audit(args.regression) if args.regression else (None, None)
    if bool(screen["selection"]) != bool(regression):
        raise ValueError("selected candidates require the complete conditional regression")
    if regression:
        selected_artifact = regression_producer["selection_artifact"]
        if selected_artifact["contents"] != screen or selected_artifact["sha256"] != sha256(args.screen):
            raise ValueError("sealed selection changed")
        for key in ("sources", "inputs", "schedules", "model", "execution", "settings", "variants", "environment", "upstream_commit"):
            if screen_producer[key] != regression_producer[key]:
                raise ValueError("screen/regression producing configuration changed: " + key)
    final = regression or screen
    campaign = args.regression or args.screen
    selected = screen["selection"]["execution"]["name"] if screen["selection"] else None
    micro_path = ROOT / "20260909_nonattention_pilot_v1/campaign_results.json"
    micro = strict_read(micro_path)
    lines = ["# Qwen3-8B prefix: accelerating work outside MoBA", "", "Date: 2026-09-09. Exploratory frozen-weight execution study.", "",
             f"Hypothesis: `qwen3-8b-prefix6-nonattention-acceleration-v1`. Classification: **`{final['classification']}`**.", "",
             "The baseline and candidates use the same six attention layers and first five complete MLPs of the pinned public Qwen3-8B checkpoint. Layer six's MLP is omitted in every arm. Official MoBA block1024/top2 is unchanged. No optimizer, training update or layer removal is used. FP16 variants retain the original weight values; INT8 variants use approximate copies and retain the original checkpoint for controls.", "",
             "## Full-prefix validation screen", "",
             "RTX 3060, one request, complete 8K/16K prefill, both corpora. Five rotated/alternating CUDA-event samples per arm; table cells are means of the two per-source medians. Loading, compilation and initialization are excluded. Fidelity is compared with original FP16 MoBA, separately from the full-attention reference.", "",
             "| Execution | 8K latency | 16K latency | 16K speed vs MoBA | 16K joint passes | Max head L2 error, 16K |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm, label in (("full", "Original FP16, full attention"), ("moba", "Original FP16, MoBA")):
        rows = [r for r in screen["cells"] if r["variant"] == "fused_fp16" and r["status"] == "completed"]
        values = [statistics.mean(r["medians_ms"][arm] for r in rows if r["length"] == n) for n in (8192, 16384)]
        lines.append(f"| {label} | {values[0]:.1f} ms | {values[1]:.1f} ms | — | — | — |")
    for variant in NAMES:
        rows = [r for r in screen["cells"] if r["variant"] == variant]
        if not all(r["status"] == "completed" for r in rows):
            lines.append(f"| {variant} | runtime failure retained | — | — | 0/2 | — |")
            continue
        long = [r for r in rows if r["length"] == 16384]
        values = [statistics.mean(r["medians_ms"]["candidate"] for r in rows if r["length"] == n) for n in (8192, 16384)]
        lines.append(f"| {variant} | {values[0]:.1f} ms | {values[1]:.1f} ms | {statistics.mean(r['speedup_vs_moba'] for r in long):.3f}× | {sum(r['gate'] for r in long)}/2 | {100*max(r['metrics_vs_moba']['heads_relative_l2'] for r in long):.3f}% |")
    lines += ["", "FP16 fusion packs Q/K/V, fuses pointwise normalization/rotary/residual work and the MLP activation, and computes the final output projection on segment-pooled heads. The projection commutes with pooling in exact arithmetic; FP16 rounding changes are measured. INT8 candidates additionally quantize the first five MLPs: threshold 6, threshold 3, or threshold 6 gate/up only with FP16 down projections. The compiler used pointwise fusion and vendor GEMMs; it reported insufficient GPU SMs for its GEMM max-autotuning path.", "",
              "FP16 gates: head/output mean cosine ≥0.9999, both relative L2 errors ≤0.005, change Pearson ≥0.999, and strongest-change-quartile F1 ≥0.95. INT8 gates: cosine ≥0.995, L2 ≤0.05, Pearson ≥0.995, and F1 ≥0.95. Every passing cell must also be ≥1.05× faster than original MoBA with no external contention detected. Selection uses both 16K validation sources; 8K is a crossover diagnostic.", "",
              f"Completed {screen['summary']['completed']}/16 screen cells; {screen['summary']['joint_passes']} joint passes across both lengths. Selected execution: `{selected}`.", ""]
    if regression:
        rows = regression["cells"]
        lines += ["## Reused-window regression", "", "The selected execution was fixed before checking the five existing final input windows per corpus. These are ten reused numerical/runtime regression windows, not ten newly trained models or a newly sealed application test.", "",
                  "| Corpus / seed | Original full | Original MoBA | Selected | Speed vs MoBA | Head L2 | Change Pearson | Change F1 | Joint pass |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for r in rows:
            if r["status"] != "completed":
                lines.append(f"| {r['source']} / {r['seed']} | failed | — | — | — | — | — | — | no |")
                continue
            m, t = r["metrics_vs_moba"], r["medians_ms"]
            lines.append(f"| {r['source']} / {r['seed']} | {t['full']:.1f} ms | {t['moba']:.1f} ms | {t['candidate']:.1f} ms | {r['speedup_vs_moba']:.3f}× | {100*m['heads_relative_l2']:.3f}% | {m['change_pearson']:.6f} | {m['top_quarter_change_f1']:.4f} | {r['gate']} |")
        completed = [r for r in rows if r["status"] == "completed"]
        means = {arm: statistics.mean(r["medians_ms"][arm] for r in completed) for arm in ("full", "moba", "candidate")}
        lines += ["", f"Joint passes: **{regression['summary']['joint_passes']}/10**. Mean of window medians: original full **{means['full']:.1f} ms**, original MoBA **{means['moba']:.1f} ms**, selected **{means['candidate']:.1f} ms**. Ratio of mean latencies: **{means['moba']/means['candidate']:.3f}×** versus original MoBA ({100*(1-means['candidate']/means['moba']):.1f}% less latency), **{means['full']/means['candidate']:.3f}×** versus original full attention ({100*(1-means['candidate']/means['full']):.1f}% less latency).", "",
                  f"Against unchanged MoBA, worst head/output relative L2: {100*max(r['metrics_vs_moba']['heads_relative_l2'] for r in completed):.3f}% / {100*max(r['metrics_vs_moba']['ov_relative_l2'] for r in completed):.3f}%. Minimum change Pearson: {min(r['metrics_vs_moba']['change_pearson'] for r in completed):.6f}; minimum change F1: {min(r['metrics_vs_moba']['top_quarter_change_f1'] for r in completed):.4f}.", "",
                  f"Against original full attention, mean head cosine ranges {min(r['metrics_vs_full']['heads_cosine_mean'] for r in completed):.6f}–{max(r['metrics_vs_full']['heads_cosine_mean'] for r in completed):.6f}, and change Pearson {min(r['metrics_vs_full']['change_pearson'] for r in completed):.6f}–{max(r['metrics_vs_full']['change_pearson'] for r in completed):.6f}. Those include MoBA's approximation and are not accuracy measurements.", ""]
    lines += ["## MLP pilot and practical limits", "", "The preliminary microbenchmark used the actual third-layer MLP input from the 8K Wikipedia validation window. Its results did not determine full-prefix support:", "",
              "| One MLP variant | Latency | Speed | Relative L2 |", "| --- | ---: | ---: | ---: |"]
    for r in micro["cells"]:
        if r["status"] == "completed":
            lines.append(f"| {r['variant']} | {r['candidate_ms']:.2f} ms | {r['speedup']:.3f}× | {100*r['relative_l2']:.3f}% |")
    all_rows = [r for r in screen["cells"] + (regression["cells"] if regression else []) if r["status"] == "completed"]
    lines += ["", f"Numerical first forwards, including compilation when needed, ranged {min(r['first_forward_seconds_including_compile'] for r in all_rows):.2f}–{max(r['first_forward_seconds_including_compile'] for r in all_rows):.2f} seconds in the screen/regression. The earlier 8K lifecycle pilot incurred a separate 14.34-second first forward. Disk compiler caches were reused across stages; setup numbers are not universal cold-cache costs. Peak allocated CUDA memory, including retained originals and any approximate copies, was {max(r['peak_cuda_allocated_bytes'] for r in all_rows)/1e9:.2f} GB.", "",
              "Original checkpoint hashes match before/after in each stage. GPU process snapshots accompany every cell. This is complete-context batch-one execution on one RTX 3060; no claim is made for other GPUs, short contexts, decoding, or the gateway retrieval model. INT8 changes arithmetic precision; it does not remove the MLPs. FP16 execution reduces overhead and moves one linear projection after pooling; it does not make all five MLP matrix multiplications disappear.", "",
              "**Application accuracy and the user's 95% final-accuracy target remain unmeasured.** The output checks assess preservation of attention readouts and change rankings on representative public text. No hierarchy labels, retrieval outcomes, or answer accuracy were supplied. No MLP or attention weight was retrained. Incremental caching was not benchmarked; append-only workload clarification was not available when this full-pass study ran.", "", "## Provenance and reproduction", "",
              f"- Screen: `{args.screen}`; SHA-256 `{sha256(args.screen)}`.",
              f"- Decisive campaign: `{campaign}`; SHA-256 `{sha256(campaign)}`.",
              f"- MLP pilot: `{micro_path}`; SHA-256 `{sha256(micro_path)}`.",
              f"- Screen fingerprint: `{screen_producer['fingerprint']}`.",
              "- Protocol: `docs/07 - Status Reports/2026-09-09_qwen8b-nonattention-acceleration.md`.",
              "- Seven focused tests cover packed MLP order, full-prefix graph equivalence, numerical gates, two-source selection, and conservative registry status.", "",
              "```bash", "env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH= \\",
              " PYTHONPATH=.:src TORCHINDUCTOR_COMPILE_THREADS=2 \\",
              " TORCHINDUCTOR_CACHE_DIR=$PWD/data/experiments/moba_paper/inductor_cache \\",
              " TRITON_CACHE_DIR=$PWD/data/experiments/moba_paper/triton_cache \\",
              " data/experiments/moba_paper/environment/bin/python \\",
              " -m experiments.structure_net.run_qwen_nonattention --stage screen \\",
              " --output data/experiments/qwen_moba_prefix/NEW_NONATTENTION_SCREEN_ROOT", "```", "",
              "Use `--stage regression --selection <new-screen>/campaign_results.json` and a new output root only if the fixed screen selects a candidate. The isolated environment and pinned model/upstream hashes are recorded in each producer manifest.", ""]
    report = "\n".join(lines)
    if REPORT.exists() and REPORT.read_text() != report:
        raise ValueError("preserve existing different report")
    if not REPORT.exists():
        REPORT.write_text(report)
    evidence = NonattentionEvidence(str(campaign), sha256(campaign), str(REPORT), sha256(REPORT), final["classification"],
                                    16, len(regression["cells"]) if regression else 0,
                                    regression["summary"]["joint_passes"] if regression else 0, selected)
    if args.register:
        from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
        logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses", queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected", enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
        record = evidence.record()
        logger.log_hypothesis(record)
        exact = logger.hypotheses_collection.get(ids=[record["id"]], include=["metadatas"])
        if exact["ids"] != [record["id"]] or any(exact["metadatas"][0].get(k) != record[k] for k in ("tested", "confirmed", "confirmation_status", "description")):
            raise RuntimeError("exact registry readback failed")
        semantic = logger.hypotheses_collection.query(query_texts=["Qwen frozen prefix nonattention acceleration FP16 fusion INT8 MLP"], n_results=10)
        if record["id"] not in semantic["ids"][0]:
            raise RuntimeError("semantic registry readback failed")
        saved = {"schema": "nal.qwen-nonattention-registry.v1", "hypotheses": [record], "screen_sha256": sha256(args.screen),
                 "exact_readback_verified": True, "semantic_readback_verified": True}
        save(Path("data/hypothesis_registry") / (record["id"] + ".json"), saved)
        save(campaign.parent / "registry_readback.json", saved)
    print(json.dumps({"report": str(REPORT), "campaign_sha256": sha256(campaign), "report_sha256": sha256(REPORT),
                      "classification": final["classification"], "registered": args.register}))


if __name__ == "__main__":
    main()
