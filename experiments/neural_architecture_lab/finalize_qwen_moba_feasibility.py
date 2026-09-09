#!/usr/bin/env python3
"""Build an audited, conservative report from Qwen MoBA feasibility stages."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics

from experiments.neural_architecture_lab.audit_qwen_moba_feasibility import audit, sha256, strict_read, quality_pass
from neural_architecture_lab.qwen_moba_feasibility_hypothesis import QwenMoBAEvidence, store_evidence

ROOT = Path("data/experiments/qwen_moba_prefix")
REPORT = Path("docs/08 - Analysis/2026-09-09_qwen8b-moba-feasibility.md")


def save(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError(f"preserve existing different artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text)


def attention_pair_fraction(length, block, topk):
    if length % block:
        raise ValueError("this analytic count requires complete blocks")
    chunks = length // block
    retained = chunks * block * (block + 1) // 2 + block * block * sum(min(i, topk - 1) for i in range(chunks))
    return retained / (length * (length + 1) / 2)


def kernel_audit(path):
    result = strict_read(path)
    if result["status"] != "completed" or result["classification"] != "shared_gpu_systems_diagnostic":
        raise ValueError("requires completed explicitly shared-GPU diagnostic")
    producer = result["producer"]
    if strict_read(path.parent / "producer.json") != producer or result["uncontended_speed_claim_allowed"]:
        raise ValueError("kernel producer or interpretation mismatch")
    snapshots = {p.name.split("_", 1)[1]: p for p in (path.parent / "producer_source").iterdir()}
    for source, digest in producer["sources"].items():
        if sha256(snapshots[Path(source).name]) != digest:
            raise ValueError("kernel source snapshot changed")
    for c in result["cells"]:
        cell_path = path.parent / "cells" / f"h{c['query_heads']}_n{c['length']}_b{c['block_size']}_k{c['top_k']}.json"
        if strict_read(cell_path) != c:
            raise ValueError("kernel cell changed")
        if c["status"] == "completed":
            for t in c["timing"].values():
                if len(t["samples_ms"]) != 5 or statistics.median(t["samples_ms"]) != t["median_ms"]:
                    raise ValueError("kernel timing samples inconsistent")
            if c["attention_speedup"] != c["timing"]["full"]["median_ms"] / c["timing"]["moba"]["median_ms"]:
                raise ValueError("kernel timing ratio mismatch")
    if result["summary"] != {"requested": len(result["cells"]), "completed": sum(c["status"] == "completed" for c in result["cells"])}:
        raise ValueError("kernel counts inconsistent")
    return result


def figure(screen, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    names = list(dict.fromkeys(c["name"] for c in screen["cells"]))
    for index, name in enumerate(names):
        rows = [c for c in screen["cells"] if c["name"] == name and c["status"] == "completed"]
        for ax, metric in zip(axes, ("heads_cosine_mean", "prefix_speedup")):
            for source in ("wikipedia", "bnc_spoken"):
                selected = sorted((c for c in rows if c["source"] == source), key=lambda c: c["length"])
                values = [c["prefix_speedup"] if metric == "prefix_speedup" else c["metrics"][metric] for c in selected]
                ax.plot([c["length"] for c in selected], values, color=f"C{index}", marker="o", linestyle="-" if source == "wikipedia" else ":", label=name if source == "wikipedia" else None)
    axes[0].axhline(.95, color="black", linestyle="--", linewidth=.8)
    axes[1].axhline(1.05, color="black", linestyle="--", linewidth=.8)
    axes[0].set(ylabel="Head-readout cosine similarity", title="Representation proxy; not application accuracy")
    axes[1].set(ylabel="Full-prefix time / MoBA-prefix time", title="Shared-GPU observed prefix speed")
    for ax in axes:
        ax.set(xlabel="Context tokens", xticks=[8192, 16384])
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=7)
    fig.suptitle("Frozen Qwen3-8B six-layer prefix — official MoBA")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170)
    fig.savefig(destination.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def render(aggregate, readouts, screen, kernel, final, campaign_path):
    lines = ["# Qwen3-8B prefix: measured MoBA feasibility", "", "Date: 2026-09-09.", "",
             f"Hypothesis: `{aggregate['hypothesis_id']}`. Classification: **`{aggregate['classification']}`**.", "",
             "The official MoBA implementation executes on the real frozen Qwen3-8B six-layer prefix. Several sparse settings preserve the defined readout proxies without training any weights. Runtime conclusions below apply to the shared RTX 3060 used here. Uncontended speed and the application's final accuracy remain unconfirmed.", "",
             f"Audited aggregate: `{campaign_path}`. SHA-256: `{sha256(campaign_path)}`.", "",
             "## What was executed", "",
             "Public `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`: embedding plus the first six layers, 1,780,008,448 parameters loaded. Layers 1–5 execute complete transformer blocks; layer 6 stops after attention and its output projection. All 67 loaded tensors stayed frozen, verified by before/after hashes. No MLP, attention projection, or adapter was trained.", "",
             "FP16 weights and forward computation; FP32 routing logits and softmax accumulation in official MoBA, and FP32 pooled readouts. Full attention uses FlashAttention 2.6.3 with native grouped-query attention. MoBA is upstream commit `b5d58363311d3ca946f1ec444182727c15e338b5`; its required KV-head expansion, routing, rearrangement and output combining are included in timing. The auxiliary custom Triton kernel was not used for these Qwen results.", "",
             "Inputs are Qwen-tokenized Wikipedia and BNC spoken text, with fixed 60/20/20 partitions. They are representative language inputs, not the user's Terra-summary distribution. Readouts pool 128-token segments of sixth-layer attention values and output-projected values. Adjacent-segment change correlation and top-quarter change F1 measure preservation of the full-attention reference; none measures hierarchy or retrieval accuracy.", "",
             "## Completion and controls", "",
             f"- Readout-only grid: {readouts['summary']['completed']}/120 completed; 24 baseline allocation failures at 32K under the 45% GPU-memory cap. {sum(quality_pass(c) for c in readouts['cells'])} sparse cells passed all numerical readout gates. No runtime or joint selection was inferred from that stage.",
             f"- Shared-GPU prefix screen: {screen['summary']['completed']}/24 completed; {screen['summary']['joint_passes']} individual source/configuration cells passed the joint readout/runtime gate.",
             f"- Synthetic official-kernel diagnostic: {kernel['summary']['completed']}/{kernel['summary']['requested']} completed; six 32K Qwen-head-shape cases exceeded its 40% memory cap.",
             "- Actual-prefix parity against the Hugging Face graph passed; sixth-MLP exclusion and token/head chunking have focused contract tests. Official forward/backward pilot output and gradient errors were within the stated tolerance; this feasibility campaign itself has zero optimizer updates.",
             "- The shared RTX 3060 had an unrelated workload occupying about 5 GiB and reporting 100% utilization before these measurements. No unrelated process was stopped. Raw per-cell GPU state and all timing samples are retained.", "",
             "## Whole-prefix observations", "",
             "Both paths include embedding, Q/K/V and output projections, five MLPs, and identical pooled readouts. MoBA processes four query heads at a time to fit beside the other workload; full attention retains native GQA. MLP computation uses identical 2048-token chunks in both paths. Times use five alternating paired CUDA-event samples after one warmup per path. Table ranges span the two source windows, not confidence intervals.", "",
             "| Tokens | MoBA layers | Block / top-k | Head cosine, minimum source mean | Change correlation, minimum | Change F1, minimum | Prefix speedup range | Both sources pass? |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: | :---: |"]
    for length in (8192, 16384):
        for name in dict.fromkeys(c["name"] for c in screen["cells"]):
            rows = [c for c in screen["cells"] if c["length"] == length and c["name"] == name]
            if any(c["status"] != "completed" for c in rows):
                lines.append(f"| {length} | {name} | — | capacity failure | — | — | — | No |")
                continue
            setting = rows[0]["setting"]
            minimum = lambda metric: min(c["metrics"][metric] for c in rows)
            speeds = [c["prefix_speedup"] for c in rows]
            lines.append(f"| {length} | {setting['layers']} | {setting['block_size']} / {setting['top_k']} | {minimum('heads_cosine_mean'):.4f} | {minimum('change_pearson'):.4f} | {minimum('top_quarter_change_f1'):.4f} | {min(speeds):.3f}–{max(speeds):.3f}× | {'Yes' if all(c['gate'] for c in rows) else 'No'} |")
    lines += ["", "The joint gate also requires mean output-projected cosine ≥0.95, head cosine ≥0.95, change correlation ≥0.95, change F1 ≥0.90 and observed prefix speedup ≥1.05 on each source. Full endpoint values, timing samples, minima and maxima are in the cell JSON.", "",
              "![Readout preservation and shared-GPU prefix speed](figures/2026-09-09_qwen8b-moba-feasibility.png)", "", "## Attention-only diagnostic", "",
              "The following uses synthetic Q/K/V with Qwen's 32-query/8-KV-head shape and processes all heads together. It measures only attention and has no model weights or quality endpoint. It shares the GPU, so these numbers cannot be used as an uncontended benchmark or a whole-prefix speedup.", "",
              "| Tokens | Block / top-k | Full attention ms | MoBA ms | Attention speedup |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for c in kernel["cells"]:
        if c["status"] == "completed" and c["query_heads"] == 32 and c["block_size"] in (256, 512, 1024):
            lines.append(f"| {c['length']} | {c['block_size']} / {c['top_k']} | {c['timing']['full']['median_ms']:.2f} | {c['timing']['moba']['median_ms']:.2f} | {c['attention_speedup']:.3f}× |")
    lines += ["", f"At 16K, block256/top4 evaluates approximately {100 * attention_pair_fraction(16384, 256, 4):.2f}% of the full causal token pairs per sparse layer. This counts QK/AV attention pairs only; routing, movement, projections and MLP work remain. Fewer pairs therefore do not establish proportionally lower total computation or latency.", "", "## Held-out decision", ""]
    if final is None:
        lines += ["No setting passed the shared-GPU selection rule on both sources. The ten final windows remain sealed. Adapter training was not launched: the tested execution path did not establish the joint readout/runtime result needed to justify it. The uncontended primary screen also remains deferred."]
    else:
        lines += [f"Frozen selection: `{json.dumps(screen['selection'], sort_keys=True)}`. Fresh final windows: {final['summary']['completed']}/10 completed, {final['summary']['joint_passes']}/10 joint passes. Classification: `{final['classification']}`.", "",
                  "| Source | Seed | Head cosine | Change correlation | Change F1 | Full prefix ms | MoBA prefix ms | Speedup | Joint pass |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |"]
        for c in final["cells"]:
            if c["status"] != "completed":
                lines.append(f"| {c['source']} | {c['seed']} | capacity failure | — | — | — | — | — | No |")
            else:
                m = c["metrics"]
                lines.append(f"| {c['source']} | {c['seed']} | {m['heads_cosine_mean']:.4f} | {m['change_pearson']:.4f} | {m['top_quarter_change_f1']:.4f} | {c['timing']['full']['median_ms']:.2f} | {c['timing']['moba']['median_ms']:.2f} | {c['prefix_speedup']:.3f}× | {'Yes' if c['gate'] else 'No'} |")
        completed = [c for c in final["cells"] if c["status"] == "completed"]
        if completed:
            head_errors = [c["metrics"]["heads_relative_l2"] for c in completed]
            ov_errors = [c["metrics"]["ov_relative_l2"] for c in completed]
            lines += ["", f"Head-readout relative L2 error ranges from {min(head_errors):.3f} to {max(head_errors):.3f}; output-projected relative L2 error ranges from {min(ov_errors):.3f} to {max(ov_errors):.3f}. The cosine gates measure directional similarity; the recorded L2 errors quantify the remaining numerical difference. Peak allocated CUDA memory reaches {max(c['peak_cuda_allocated_bytes'] for c in completed) / 2**30:.3f} GiB for this memory-bounded execution."]
        lines += ["", "These are input-window replicates from one checkpoint, not independently trained models. Shared-GPU passage remains an exploratory deployment observation, with uncontended performance unconfirmed."]
    lines += ["", "## Interpretation and limits", "",
              "MoBA is technically applicable to this Qwen prefix while leaving all fully connected weights frozen. The measured readout preservation supports further application testing. It does not establish 95% final accuracy, improved retrieval, or generally better model quality. The reference is the unchanged full-attention readout, not a labeled ground truth.", "",
              "The [paper](https://arxiv.org/pdf/2502.13189) and [official implementation](https://github.com/MoonshotAI/MoBA) concern sparse attention and long-context behavior. The paper's 95.31% figure is sparsity, not accuracy. Its large-context throughput results do not imply the same speedup for this shorter, consumer-GPU prefix. This experiment did not reproduce its continued-training budget, million-token regime, decoding, or the full 8B network.", "",
              "Next evidence needed for an application decision: an uncontended GPU run using the existing frozen protocol, followed by actual summary inputs and labeled hierarchy/retrieval evaluation. The original TinyLLM distillation study and the stopped all-weight BLiMP search remain separate; neither supplies Qwen application evidence.", "", "## Reproduction and provenance", "",
              "Use the isolated `data/experiments/moba_paper/environment/bin/python` (PyTorch 2.4.1+cu121, FlashAttention 2.6.3) with `LD_LIBRARY_PATH=` and repository `PYTHONPATH`. The [reproduction guide](../02%20-%20Implementation/qwen-moba-feasibility-reproduction.md) includes environment and pinned-weight setup. Model, tokenizer, source corpus, token windows, producer sources and source snapshots are identified in each `producer.json`. Raw completed directories are immutable; rerun into a new output root.", "",
              "```bash", "env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH= \\", "  PYTHONPATH=/home/rabbit/structure_net \\", "  data/experiments/moba_paper/environment/bin/python -m \\", "  experiments.structure_net.qwen_moba_shared_assessment \\", "  --output data/experiments/qwen_moba_prefix/NEW_shared_screen", "```", "",
              "| Evidence | SHA-256 |", "| --- | --- |"]
    for artifact in aggregate["artifacts"]:
        lines.append(f"| `{artifact['path']}` | `{artifact['sha256']}` |")
    lines += ["", "Source/tests/protocol/report are durable repository artifacts. Large checkpoint and campaign data stay under ignored `data/`. The registry retains `confirmed: false`, explicitly distinguishing completed exploratory measurements from unconfirmed uncontended performance and unmeasured application accuracy.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readouts", type=Path, default=ROOT / "20260909_readouts_v1/campaign_results.json")
    parser.add_argument("--screen", type=Path, default=ROOT / "20260909_shared_screen_v1/campaign_results.json")
    parser.add_argument("--kernel", type=Path, default=Path("data/experiments/moba_head_benchmark/20260909_paper_shared_diagnostic_v1/campaign_results.json"))
    parser.add_argument("--final", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "20260909_registered_shared_v1")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    stages = [args.readouts, args.screen] + ([args.final] if args.final else [])
    audits = [audit(path, sha256(path)) for path in stages]
    readouts, screen = strict_read(args.readouts), strict_read(args.screen)
    final = strict_read(args.final) if args.final else None
    if bool(screen["selection"]) != bool(final):
        raise ValueError("selected candidate requires final evaluation; no selection must leave final sealed")
    kernel = kernel_audit(args.kernel)
    artifacts = [{"path": str(path), "sha256": sha256(path)} for path in stages + [args.kernel]]
    classification = final["classification"] if final else screen["classification"]
    aggregate = {"schema": "nal.qwen-moba-feasibility-registered.v1", "status": "completed_exploratory",
                 "hypothesis_id": screen["hypothesis_id"], "classification": classification, "confirmed": False,
                 "application_accuracy_tested": False, "uncontended_speed_confirmed": False,
                 "artifacts": artifacts, "audits": audits, "source_sha256": sha256(__file__),
                 "completed_prefix_cells": sum(a["summary"]["completed"] for a in audits),
                 "final_cells": final["summary"]["completed"] if final else 0,
                 "final_joint_passes": final["summary"]["joint_passes"] if final else 0,
                 "limitations": ["shared_gpu_timing", "generic_text_not_application_summaries", "no_application_accuracy", "32k_capacity_failures", "one_pretrained_checkpoint"]}
    campaign_path = args.output / "campaign_results.json"
    save(campaign_path, aggregate)
    text = render(aggregate, readouts, screen, kernel, final, campaign_path)
    if REPORT.exists() and REPORT.read_text() != text:
        raise ValueError("preserve existing different report")
    if not REPORT.exists():
        REPORT.write_text(text)
    destination = REPORT.parent / "figures/2026-09-09_qwen8b-moba-feasibility.png"
    if not destination.exists():
        figure(screen, destination)
    evidence = QwenMoBAEvidence(classification, aggregate["completed_prefix_cells"], aggregate["final_joint_passes"], aggregate["final_cells"], str(campaign_path), sha256(campaign_path), str(REPORT), sha256(REPORT))
    save(args.output / "evidence.json", asdict(evidence))
    if args.register:
        import numpy as np
        if "NaN" not in np.__dict__:
            np.NaN = np.nan
        from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
        logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses", queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected", enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
        receipt = store_evidence(evidence, logger, Path("data/hypothesis_registry/qwen3-8b-prefix6-moba-feasibility-v1.json"))
        save(args.output / "registry_readback.json", receipt)
        print(json.dumps(receipt))
    print(json.dumps(asdict(evidence)))


if __name__ == "__main__":
    main()
