"""Independently audit speed iteration, including every rejected candidate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import torch

from experiments.neural_architecture_lab.finalize_llama_cotenancy import audited
from experiments.structure_net.llama_cuda_graph import bitwise_tree
from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json
from neural_architecture_lab.llama_exact_speed_hypothesis import ExactSpeedEvidence, ExactBlasEvidence, SpeedCell, CANDIDATES, BLAS_CANDIDATES, GRID

REPORT = Path("docs/08 - Analysis/2026-10-10_llama32-exact-speed.md")
BLAS_REPORT = Path("docs/08 - Analysis/2026-10-10_llama32-exact-blas.md")


def read_controls(row):
    path = Path(row["control_artifact"])
    if sha256(path) != row["control_sha256"]:
        raise ValueError("changed control artifact")
    values = torch.load(path, map_location="cpu", weights_only=True)
    if len(values) != 2 or torch.equal(values[0]["ids"], values[1]["ids"]):
        raise ValueError("two distinct inputs required")
    results = []
    for value in values:
        if not bitwise_tree(value["expected"], value["baseline_graph"]) or not bitwise_tree(value["expected"], value["baseline_eager"]):
            raise ValueError("baseline graph itself is invalid")
        valid = all(bitwise_tree(value["expected"], value[k]) for k in ("candidate_eager", "actual"))
        if valid != value["bitwise_equal"]:
            raise ValueError("claimed control equality disagrees with tensor bits")
        results.append(valid)
    valid = all(results) and row["owned_outputs"]
    if valid != row["all_bitwise"]:
        raise ValueError("exactness gate mismatch")
    return valid


def validate(campaign, root, pilot=False, candidate_names=CANDIDATES):
    if campaign["status"] != ("systems_lifecycle_only_not_quality_evidence" if pilot else "completed") or not campaign["uncontended"]:
        raise ValueError("completed uncontended run required")
    if campaign["parameter_bytes"] != 6425499648:
        raise ValueError("changed parameter bytes")
    expected = {(n, length, steps) for n in candidate_names for length, steps in GRID}
    if len(campaign["workloads"]) != len(expected) or {(r["candidate"], r["length"], r["decode_steps"]) for r in campaign["workloads"]} != expected:
        raise ValueError("complete fixed grid required")
    cells = []
    anchor_path = Path(campaign["anchor_artifact"])
    if sha256(anchor_path) != campaign["anchor_sha256"]:
        raise ValueError("changed pre-packing anchors")
    anchors = torch.load(anchor_path, map_location="cpu", weights_only=True)
    for row in campaign["workloads"]:
        path = root / f"workload_{row['candidate']}_{row['length']}_{row['decode_steps']}.json"
        if json.loads(path.read_text()) != row:
            raise ValueError("changed cell artifact")
        exact = read_controls(row)
        controls = torch.load(row["control_artifact"], map_location="cpu", weights_only=True)
        original = anchors[f"{row['length']}_{row['decode_steps']}"]
        if len(original) != 2 or any(not bitwise_tree(v["ids"], a["ids"]) or not bitwise_tree(v["expected"], a["expected"]) for v, a in zip(controls, original)):
            raise ValueError("reference target moved after packing")
        del controls
        if not row["packing_verified"] or row["parameter_bytes"] != campaign["parameter_bytes"]:
            raise ValueError("changed parameter packing")
        if exact:
            count = 1 if pilot else 10
            if row["status"] != "exact_measured" or row["samples"] != count:
                raise ValueError("exact cell missing measurements")
            names = ["baseline_graph", "candidate_graph"]
            if row["orders"] != [names if i % 2 == 0 else names[::-1] for i in range(count)]:
                raise ValueError("non-alternating timing order")
            if set(row["timings"]) != set(names):
                raise ValueError("wrong timing arms")
            for arm in row["timings"].values():
                if len(arm["samples"]) != count:
                    raise ValueError("wrong sample count")
                for kind in ("wall", "cuda"):
                    numbers = [s[kind + "_ms"] for s in arm["samples"]]
                    if any(not math.isfinite(x) or x <= 0 for x in numbers) or statistics.median(numbers) != arm[kind + "_median_ms"]:
                        raise ValueError("invalid timing median")
            ratio = row["timings"][names[0]]["wall_median_ms"] / row["timings"][names[1]]["wall_median_ms"]
            if row["speedup"] != ratio or row["speed_gate"] != (ratio >= 1.10):
                raise ValueError("invalid speed gate")
            if row["decode_steps"]:
                token_path = Path(row["token_artifact"])
                if sha256(token_path) != row["token_sha256"]:
                    raise ValueError("changed token artifact")
                tokens = torch.load(token_path, map_location="cpu", weights_only=True)
                if len(tokens) != 2 or torch.equal(tokens[0]["ids"], tokens[1]["ids"]):
                    raise ValueError("two distinct token controls required")
                if not all(bitwise_tree(v["expected"], v[k]) for v in tokens for k in ("baseline", "actual")):
                    raise ValueError("generation graph differs from full trace")
        elif row["status"] != "rejected_bit_mismatch" or row["timings"] is not None or row["speedup"] is not None or row["speed_gate"]:
            raise ValueError("rejected cell supplied speed evidence")
        cells.append(SpeedCell(row["candidate"], row["length"], row["decode_steps"], exact, row["speedup"]))
    return tuple(cells)


def report(c, aggregate):
    lines = ["# Llama 3.2 3B: exact speed iteration", "", "Date: 2026-10-10. Profile: EXPLORATORY.",
        f"Classification: **`{c['classification']}`**. One checkpoint and device; confirmation remains false.", "",
        f"Aggregate: `{aggregate}`, SHA-256 `{sha256(aggregate)}`.",
        "[Prospective plan](../07%20-%20Status%20Reports/2026-10-10_llama32-exact-speed-plan.md).", "",
        "This iteration compares five projection implementations against the already optimized CUDA graph baseline. Every arm uses the full 28-layer original FP16 model on GPU 0 (RTX 3060), with unchanged attention, MLPs and parameters. `mm` uses single-model 2D GEMM, `packed_bmm` combines Q/K/V and gate/up output rows, and `packed_mm` does both. `parallel2` and `parallel3` overlap independent, unchanged projections using two or three CUDA streams.", "",
        "| Candidate | Prompt/generated | Exact controls | Baseline graph ms | Candidate graph ms | Speed ratio |",
        "| --- | ---: | --- | ---: | ---: | ---: |"]
    for r in c["workloads"]:
        numbers = [f"{r['timings'][k]['wall_median_ms']:.3f}" for k in ("baseline_graph", "candidate_graph")] if r["timings"] else ["unmeasured", "rejected"]
        ratio = f"{r['speedup']:.3f}×" if r["speedup"] else "—"
        lines.append(f"| {r['candidate']} | {r['length']}/{r['decode_steps']} | {'pass' if r['all_bitwise'] else 'FAIL'} | {numbers[0]} | {numbers[1]} | {ratio} |")
    lines += ["", f"Candidates passing all exactness cells: `{c['eligible_candidates']}`.",
        f"Promoted workload configurations (>=1.10×): `{c['promoted']}`.", "",
        "Each exact cell uses ten alternating-order timing pairs after warmup, with device synchronization and wall/CUDA-event samples retained. Latency includes graph input copying and owned output cloning. Historical eager timings are not the baseline. A rejected candidate cannot be rescued by a faster or exact result on another shape.", "",
        "Two unequal inputs per cell were checked. Prefill controls retain every output-logit and KV bit. Generation controls retain all 32 steps' logits, every token and the final KV cache, with candidate eager and graph output compared to the original reference. The separately timed token-only generation graphs were checked against those traces. Previously returned outputs were checked after a replay on changed input. Finalization independently reloads and compares the saved tensors.", "",
        "The original checkpoint SHA-256 is `f01cc0653ec6c9bd7026b8676864612b91b6c79d2dc4df42599cbe4cf99eb9a9`. Every packed parameter slice was checked bitwise against its source. Parameter storage remains **6,425,499,648 bytes**. Reference projections become views into the packed allocation with their original shape and strides. Before packing, all six original reference outputs were saved; every later reference eager/graph output was checked against those immutable anchors. Both execution paths share physical weights, so paired peaks below are not the memory of two independent models.", "",
        "| Candidate | Packing/setup s | Paired peak allocated GB (maximum exact cell) | Maximum candidate timing capture/warmup s |",
        "| --- | ---: | ---: | ---: |"]
    for name, setup in c["setup"].items():
        rows = [r for r in c["workloads"] if r["candidate"] == name and r["all_bitwise"]]
        peak = f"{max(r['paired_memory']['peak_allocated_bytes'] for r in rows)/1e9:.3f}" if rows else "unmeasured"
        capture = f"{max(r['timing_capture_seconds']['candidate'] for r in rows):.3f}" if rows else "unmeasured"
        lines.append(f"| {name} | {setup['packing_seconds']:.3f} | {peak} | {capture} |")
    lines += ["", "Capture/packing cost is excluded from steady-state timings and requires reuse to amortize. Graphs have fixed prompt and generation lengths. Each candidate/workload pair is released before the next. GPU process/memory snapshots passed the contention check; this does not measure transient external kernel activity. No concurrent experiment or heavy audit was scheduled during timing.", "",
        f"All fifteen primary cells reached a terminal result: {c['exact_cells']} exact measurements and {c['rejected_cells']} numerical rejections. No retried or pooled primary cells. Pilots v1 and v2 stopped on out-of-memory failures at packed prefill 512, after observing packed-projection bit mismatches. The [first amendment](../07%20-%20Status%20Reports/2026-10-10_llama32-exact-speed-amendment.md) reduced audit buffer lifetime and added scheduling candidates. The [storage amendment](../07%20-%20Status%20Reports/2026-10-10_llama32-exact-speed-storage-amendment.md) shares packed buffers with the reference and anchors all comparisons to saved pre-packing outputs. Both preceded primary timing. Failed pilots are preserved; corrected pilot v3 contains systems evidence only. This report uses primary v3 only.", "",
        "No training, tensor rank reduction, weight compression or application-accuracy improvement is claimed. Exactness is established on the tested inputs, not proved over every possible input/kernel dispatch. The prior 95% application target remains unmeasured. Two independent models within one model's storage/latency budget remain unsupported.", "",
        "Reproduce using `experiments.structure_net.campaign_llama_cotenancy --exact-speed pilot` and then `--exact-speed full`, with separate new `--output` roots and the original `--control` root. Audit and register with `experiments.neural_architecture_lab.finalize_llama_exact_speed --campaign <full-root>/speed --register`.", ""]
    return "\n".join(lines)


def blas_report(c, aggregate):
    lines = ["# Llama 3.2 3B: exact BLAS backend iteration", "",
        "Date: 2026-10-10. Profile: EXPLORATORY. One checkpoint/device; confirmation remains false.",
        f"Classification: **`{c['classification']}`**.", "",
        f"Aggregate `{aggregate}`; SHA-256 `{sha256(aggregate)}`.",
        "[Prospective plan](../07%20-%20Status%20Reports/2026-10-10_llama32-exact-blas-plan.md).", "",
        "This follow-up prefers cuBLASLt for original dense projections using PyTorch's installed backend API, with batched GEMM or 2D GEMM. The control explicitly uses cuBLAS. This is a library preference that can fall back for unsupported operations; it does not establish which kernel algorithm the library selected. Backend state is restored after every projection, including on exceptions.", "",
        "| Candidate | Prompt/generated | Bitwise controls | Baseline graph ms | Candidate graph ms | Speed ratio |",
        "| --- | ---: | --- | ---: | ---: | ---: |"]
    for r in c["workloads"]:
        timing = [f"{r['timings'][n]['wall_median_ms']:.3f}" for n in ("baseline_graph", "candidate_graph")] if r["timings"] else ["unmeasured", "rejected"]
        ratio = f"{r['speedup']:.3f}×" if r["speedup"] else "—"
        lines.append(f"| {r['candidate']} | {r['length']}/{r['decode_steps']} | {'pass' if r['all_bitwise'] else 'FAIL'} | {timing[0]} | {timing[1]} | {ratio} |")
    lines += ["", f"Eligible candidates: `{c['eligible_candidates']}`. Promoted workloads: `{c['promoted']}`.", "",
        f"All six fixed cells completed: {c['exact_cells']} exact measurements, {c['rejected_cells']} numerical rejections, no runtime failures or reused primary cells. A failed exactness cell vetoes the entire candidate. Only exact cells receive ten alternating timing pairs; a workload requires >=1.10× to pass the speed gate. The baseline is the current CUDA graph, not the slower eager implementation.", "",
        "Complete original 28-layer FP16 Llama 3.2 3B, RTX 3060 physical GPU 0, two CPU threads, TF32 disabled, unchanged Flash Attention and all MLP layers. Checkpoint SHA-256 `f01cc0653ec6c9bd7026b8676864612b91b6c79d2dc4df42599cbe4cf99eb9a9`. Both paths share the same **6,425,499,648 parameter bytes**; no weight packing, compression or training.", "",
        "Original eager anchors were saved before the intervention. Two unequal inputs per workload check every prefill logit/KV bit and all 32 generation-step logits, generated tokens and final KV cache. Candidate eager/graph and baseline eager/graph were compared to those anchors. Changed-input replay and output ownership were checked. All saved tensor controls, hashes, producer snapshots and timing medians were independently audited. The separate tiny/full-model pilot contains systems evidence only.", "",
        "Timings include graph input copying and owned outputs, after warmup and with device synchronization. All wall/CUDA samples, capture/setup times and allocator peaks are in the aggregate. Setup is excluded from steady-state latency and requires reuse. Graphs have fixed shapes and generation lengths. GPU process/memory snapshots passed the contention check; transient external kernel activity is not measured.", "",
        "Numerical differences are rejected even if generated tokens happen to agree. Bitwise checks on tested inputs do not prove equality for every possible input or library dispatch. No 95% application-accuracy result, two-for-one independent-model deployment, or universal speedup is claimed.", "",
        "Reproduce with `experiments.structure_net.campaign_llama_cotenancy --exact-blas pilot` and `--exact-blas full`, separate new output roots and the original control root. Audit with `experiments.neural_architecture_lab.finalize_llama_exact_speed --backend --campaign <root>/blas --pilot-audit` for the pilot; use `--register` for the primary run.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--pilot-audit", action="store_true", help="verify pilot artifacts without generating primary claims")
    parser.add_argument("--backend", action="store_true", help="audit the separate BLAS backend follow-up")
    args = parser.parse_args()
    names = BLAS_CANDIDATES if args.backend else CANDIDATES
    evidence_type = ExactBlasEvidence if args.backend else ExactSpeedEvidence
    report_path = BLAS_REPORT if args.backend else REPORT
    if args.pilot_audit:
        if args.register:
            raise ValueError("pilot cannot be registered as primary evidence")
        result = json.loads((args.campaign / "result.json").read_text())
        producer = json.loads((args.campaign / "producer.json").read_text())
        fingerprint = hashlib.sha256(json.dumps({k: v for k, v in producer.items() if k != "fingerprint"}, sort_keys=True).encode()).hexdigest()
        if fingerprint != producer["fingerprint"] or fingerprint != result["producer_fingerprint"]:
            raise ValueError("pilot fingerprint mismatch")
        snapshots = {p.name.split("_", 1)[1]: p for p in (args.campaign / "producer_source").iterdir()}
        if any(sha256(snapshots[Path(path).name]) != digest for path, digest in producer["sources"].items()):
            raise ValueError("pilot source changed")
        if not producer["configuration"]["pilot"] or producer["configuration"]["samples"] != 1:
            raise ValueError("wrong pilot configuration")
        cells = validate(result, args.campaign, pilot=True, candidate_names=names)
        for row in result["tiny_lifecycle"]:
            read_controls(row)
        receipt = {"status": "systems_lifecycle_audit_passed_not_quality_evidence", "result_sha256": sha256(args.campaign / "result.json"),
                   "source_fingerprint": fingerprint, "cells": len(cells), "exact_cells": sum(v.exact for v in cells),
                   "tiny_checks": len(result["tiny_lifecycle"])}
        write_json(args.campaign / "pilot_audit.json", receipt)
        print(json.dumps(receipt))
        return
    result, producer = audited(args.campaign / "result.json")
    if producer["configuration"]["pilot"] or producer["configuration"]["samples"] != 10:
        raise ValueError("pilot cannot provide measured evidence")
    cells = validate(result, args.campaign, candidate_names=names)
    draft = evidence_type("pending", "0"*64, "pending", "0"*64, cells).record()
    if result["hypothesis_id"] != draft["id"]:
        raise ValueError("wrong hypothesis identity")
    if result["eligible_candidates"] != draft["eligible_candidates"] or result["promoted"] != draft["promoted"]:
        raise ValueError("promotion disagrees with independently audited cells")
    if any(result[k] for k in ("application_accuracy_tested", "weight_compression_used", "two_for_one_supported")):
        raise ValueError("unsupported scope claim")
    c = {**result, "classification": draft["confirmation_status"], "profile": "EXPLORATORY",
         "requested": len(cells), "completed": len(cells), "failed_runtime": 0, "reused": 0,
         "exact_cells": sum(v.exact for v in cells), "rejected_cells": sum(not v.exact for v in cells),
         "source_result": str(args.campaign / "result.json"), "source_result_sha256": sha256(args.campaign / "result.json")}
    aggregate = args.campaign / "campaign_results.json"
    if aggregate.exists() and json.loads(aggregate.read_text()) != c:
        raise ValueError("preserve completed aggregate")
    write_json(aggregate, c)
    prose = (blas_report if args.backend else report)(c, aggregate)
    if report_path.exists() and report_path.read_text() != prose:
        raise ValueError("preserve completed report")
    report_path.write_text(prose)
    evidence = evidence_type(str(aggregate), sha256(aggregate), str(report_path), sha256(report_path), cells)
    if args.register:
        from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
        logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
            queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
            enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
        record = evidence.record()
        logger.log_hypothesis(record)
        exact = logger.hypotheses_collection.get(ids=[record["id"]], include=["metadatas"])
        keys = ("tested", "confirmed", "confirmation_status", "direct_experiment_count")
        if exact["ids"] != [record["id"]] or any(exact["metadatas"][0].get(k) != record[k] for k in keys):
            raise RuntimeError("exact registry readback failed")
        semantic = logger.hypotheses_collection.query(query_texts=[record["description"]], n_results=10)
        if record["id"] not in semantic["ids"][0]:
            raise RuntimeError("semantic registry readback failed")
        saved = {"hypotheses": [record], "exact_readback_verified": True, "semantic_readback_verified": True}
        write_json(Path("data/hypothesis_registry") / (record["id"] + ".json"), saved)
        write_json(args.campaign / "registry_readback.json", saved)
        if json.loads((args.campaign / "registry_readback.json").read_text())["hypotheses"][0] != record:
            raise RuntimeError("typed record readback failed")
    print(json.dumps({"classification": c["classification"], "report": str(report_path), "registered": args.register}))


if __name__ == "__main__":
    main()
