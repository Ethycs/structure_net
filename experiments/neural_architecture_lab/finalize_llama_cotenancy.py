"""Hash-check complete physical-model evidence, apply gates, report and register."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import statistics

from experiments.structure_net.campaign_llama_cotenancy import SETTINGS
from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json
from experiments.structure_net.run_llama_cotenancy import aggregate_quality, promotion
from experiments.structure_net.llama_cotenancy_data import score
from neural_architecture_lab.llama_cotenancy_hypothesis import LlamaCotenancyEvidence

REPORT = Path("docs/08 - Analysis/2026-10-10_llama32-tensorized-cotenancy.md")


def read(path):
    def reject(value):
        raise ValueError("non-finite JSON: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


@lru_cache(None)
def digest(path):
    return sha256(path)


def audited(path, expected=None):
    path = Path(path)
    if expected is not None and digest(path) != expected:
        raise ValueError("changed result: " + str(path))
    result = read(path)
    producer = read(path.parent/"producer.json")
    fingerprint = hashlib.sha256(json.dumps({k: v for k, v in producer.items() if k != "fingerprint"}, sort_keys=True).encode()).hexdigest()
    if producer["fingerprint"] != fingerprint or result["producer_fingerprint"] != fingerprint:
        raise ValueError("producer fingerprint mismatch")
    snapshots = {p.name.split("_", 1)[1]: p for p in (path.parent/"producer_source").iterdir()}
    for source, expected_source in producer["sources"].items():
        if digest(snapshots[Path(source).name]) != expected_source:
            raise ValueError("source snapshot changed")
    if result["status"] not in ("completed", "capacity_failure"):
        raise ValueError("unresolved runtime failure")
    if result.get("checkpoint") and digest(result["checkpoint"]) != result.get("sha256", result.get("checkpoint_sha256")):
        raise ValueError("changed checkpoint")
    return result, producer


def packing_pass(result):
    if result["status"] != "completed":
        return False
    checks = [p["pass"] for row in result["benchmarks"] for p in row["parity"].values() if isinstance(p, dict)]
    return bool(checks) and all(checks) and result["uncontended"]


def build(root):
    index = read(root/"campaign_index.json")
    if index["status"] != "completed":
        raise ValueError("campaign has unfinished required work")
    control, _ = audited(index["control"], index["control_sha256"])
    if not control["control_pass"]:
        raise ValueError("full-model numerical control failed")
    cells, producers = {}, {}
    for entry in index["cells"]:
        if entry["name"] in cells:
            raise ValueError("duplicate cell")
        result, producer = audited(entry["result"], entry["sha256"])
        cells[entry["name"]], producers[entry["name"]] = result, producer
    required = {"projection_screen", "gguf_pilot_m1", "gguf_pilot_m2", "gguf_validation_m1", "gguf_benchmark_m2", "gguf_benchmark_m4", "dense_fp16_validation"}
    required |= {s+"_"+suffix for s in SETTINGS for suffix in ("conversion", "pilot_m2", "validation", "benchmark_m1", "benchmark_m2", "benchmark_m4")}
    if not required <= cells.keys():
        raise ValueError("missing fixed campaign cells")
    fixtures = read(next((root/"gguf_validation_m1/producer_source").glob("*_fixtures.json")))
    by_id = {r["id"]: r for r in fixtures["examples"]}
    expected_ids = {r["id"] for r in fixtures["examples"] if r["split"] == "validation"}
    for name in ("gguf_validation_m1", "dense_fp16_validation", *[s+"_validation" for s in SETTINGS]):
        result = cells[name]
        if result["status"] != "completed" or {r["id"] for r in result["quality"]} != expected_ids or len(result["quality"]) != 4:
            raise ValueError("incomplete validation quality")
        if result["quality_summary"] != aggregate_quality(result["quality"]):
            raise ValueError("quality aggregate mismatch")
        for row in result["quality"]:
            if row["metrics"] != score(row["text"], by_id[row["id"]]):
                raise ValueError("per-fragment or source-support score mismatch")
    for name, result in cells.items():
        if "_benchmark_m" in name and result["status"] == "completed" and not name.startswith("gguf"):
            if {(r["length"], r["decode_steps"]) for r in result["benchmarks"]} != {(128, 0), (512, 0), (128, 32)}:
                raise ValueError("benchmark grid incomplete")
            for row in result["benchmarks"]:
                for arm in row["timings"].values():
                    if len(arm["samples"]) != 5 or any(arm[k+"_median_ms"] != statistics.median(s[k+"_ms"] for s in arm["samples"]) for k in ("wall", "cuda")):
                        raise ValueError("incorrect timing median/sample count")
                for p in row["parity"].values():
                    if isinstance(p, dict) and p["pass"] != (p["relative_l2"] <= .002 and p["max_abs"] <= .05 and p["top_token_agreement"] >= .99):
                        raise ValueError("packing numerical gate mismatch")
    baseline = cells["gguf_validation_m1"]
    baseline_summary = baseline["quality_summary"]
    competent = baseline_summary["fragment_coverage"] >= .9 and baseline_summary["supported_line_rate"] >= .95
    candidates = {}
    runtime_rows = []
    for setting in SETTINGS:
        validation = cells[setting+"_validation"]
        gate = promotion(validation["quality"], baseline["quality"])
        gate["packing_pass"] = all(packing_pass(cells[f"{setting}_benchmark_m{count}"]) for count in (1, 2, 4)
                                   if cells[f"{setting}_benchmark_m{count}"]["status"] == "completed")
        gate["physical_m2_executed"] = cells[f"{setting}_benchmark_m2"]["status"] == "completed"
        conversion = cells[setting+"_conversion"]
        candidates[setting] = {"gate": gate, "quality": validation["quality_summary"], "parameter_bytes": conversion["parameter_bytes"],
                               "two_models_to_one_q4_weight_ratio": 2*conversion["parameter_bytes"]/baseline["model_weight_bytes"],
                               "conversion_seconds": conversion["seconds"], "checkpoint_sha256": conversion["sha256"]}
        for count in (1, 2, 4):
            result = cells[f"{setting}_benchmark_m{count}"]
            q4 = cells["gguf_validation_m1" if count == 1 else f"gguf_benchmark_m{count}"]
            if result["status"] != "completed":
                runtime_rows.append({"setting": setting, "replicas": count, "status": result["status"]})
                continue
            for row in result["benchmarks"]:
                q4row = next((r for r in q4.get("timings", []) if (r["length"], r["decode_steps"]) == (row["length"], row["decode_steps"])), None)
                q4ms = min(statistics.median(v) for v in q4row["samples_wall_ms"].values()) if q4row else None
                ms = row["timings"]["packed"]["wall_median_ms"]
                runtime_rows.append({"setting": setting, "replicas": count, "length": row["length"], "decode_steps": row["decode_steps"],
                                     "status": "completed", "packed_wall_ms": ms, "separate_wall_ms": row["timings"]["separate"]["wall_median_ms"],
                                     "packing_speedup": row["packing_wall_speedup"], "q4_best_wall_ms": q4ms, "speedup_vs_q4": q4ms/ms if q4ms else None,
                                     "peak_allocated_bytes": row["memory"]["packed"]["peak_allocated_bytes"], "parity": row["parity"]})
    promoted = [s for s, r in candidates.items() if r["gate"]["quality_pass"] and r["gate"]["packing_pass"]]
    if promoted != index["promoted"]:
        raise ValueError("promotion disagrees with producer")
    if not competent:
        classification = "quality_assay_invalid"
    elif not any(r["gate"]["quality_pass"] for r in candidates.values()):
        classification = "tensorization_quality_failed"
    elif not promoted:
        classification = "packing_execution_invalid"
    else:
        if "gguf_test" not in cells:
            raise ValueError("conditional test missing")
        successes = []
        for setting in promoted:
            result = cells[setting+"_test"]
            final = {}
            for family in ("work", "personal"):
                final[family] = promotion([r for r in result["quality"] if r["family"] == family],
                                          [r for r in cells["gguf_test"]["quality"] if r["family"] == family])
            candidates[setting]["final_family_gates"] = final
            faster = any(r["setting"] == setting and r["status"] == "completed" and r["replicas"] >= 2
                         and r["packing_speedup"] >= 1.1 and (r["speedup_vs_q4"] or 0) >= 1.1 for r in runtime_rows)
            successes.append(all(g["quality_pass"] for g in final.values()) and faster)
        classification = "practical_cotenancy_supported" if any(successes) else "quality_retained_no_practical_win"
    expected = required | ({"gguf_test", *[s+"_test" for s in promoted]} if promoted else set())
    if cells.keys() != expected:
        raise ValueError("unexpected campaign cells")
    return {"hypothesis_id": "llama32-3b-tensorized-cotenancy-v1", "schema": "nal.llama-cotenancy-aggregate.v1",
            "status": "completed", "classification": classification, "profile": "EXPLORATORY", "baseline_competent": competent,
            "candidates": candidates, "runtime_rows": runtime_rows, "projection": cells["projection_screen"]["rows"],
            "baseline_quality": baseline_summary, "dense_fp16_quality": cells["dense_fp16_validation"]["quality_summary"],
            "q4_runtime": {str(n): cells["gguf_validation_m1" if n == 1 else f"gguf_benchmark_m{n}"] for n in (1, 2, 4)},
            "summary": {"requested_cells": len(expected), "completed_cells": sum(r["status"] == "completed" for r in cells.values()),
                        "capacity_failures": sum(r["status"] == "capacity_failure" for r in cells.values()), "runtime_failures": 0,
                        "validation_examples_per_model": 4, "promoted": promoted, "conditional_test": index["conditional_test"]},
            "cells": index["cells"], "campaign_index_sha256": digest(root/"campaign_index.json"), "control": control,
            "limits": ["one model checkpoint and one RTX 3060", "four synthetic validation summaries; seeds are inputs, not training replicas",
                       "Q4 timing uses a different runtime; both sequential and independent concurrent baselines are measured",
                       "GGUF memory peaks are boundary samples and may miss transient allocations",
                       "Only 1, 2, and 4 physical models were tested; maximum capacity is not established",
                       "No retraining or calibration; this does not disprove learned tensor networks or other ranks"],
            "application_accuracy_tested": False, "trained_parameters": 0}


def report(campaign, artifact):
    c = campaign
    lines = ["# Llama 3.2 3B tensorization and physical-model packing", "", "Date: 2026-10-10. Exploratory feasibility study.", "",
             f"Hypothesis: `{c['hypothesis_id']}`. Status: **completed**. Classification: **`{c['classification']}`**.", "",
             f"Aggregate: `{artifact}`. SHA-256: `{sha256(artifact)}`.", "",
             "Complete 28-layer replicas retain all attention, MLPs, residuals, normalization, RoPE and the tied embedding/output head. Corresponding independent factor tensors are stacked on a leading model axis and contracted with dense batched GEMMs. No model weights are shared across replicas. Separate execution uses disjoint slices of exactly the same factors. There was no training.", "",
             "## Summary quality", "", "Greedy generation on four preregistered synthetic exact-source extraction tasks, three final-decision fragments each. Output lines are checked against exact source sentences. These measurements are not application accuracy or evidence for a 95% production target. Compression includes the embedding/output head.", "",
             "| Model | Covered fragments | Supported lines | Complete summaries | Reference NLL | Weight bytes/model |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, quality, size in [("Deployed Q4_K_M", c["baseline_quality"], c["q4_runtime"]["1"]["model_weight_bytes"]),
                                 ("Original FP16", c["dense_fp16_quality"], c["control"]["parameter_bytes"]),
                                 *[(s, r["quality"], r["parameter_bytes"]) for s, r in c["candidates"].items()]]:
        lines.append(f"| {name} | {quality['covered_fragments']}/{quality['total_fragments']} | {quality['supported_lines']}/{quality['output_lines']} | {quality['complete_summaries']}/{quality['examples']} | {quality['reference_nll_mean']:.3f} | {size:,} |")
    lines += ["", f"Q4 baseline competence passed: **{c['baseline_competent']}**. Conditional ten-example final evaluation: `{c['summary']['conditional_test']}`. Complete generations are stored in every validation `result.json`.", "",
              "## Full-model timing", "", "RTX 3060, FP16 factors/inference, FP32 decomposition and normalization. Each entry is a synchronized wall-time median of five samples after one warmup. Q4 uses the faster median of sequential and concurrent execution of independent llama.cpp models. Lengths are common token inputs; decode rows force 32 steps after a 128-token prefill. Conversion/loading are excluded. Each physical replica reserves a 1024-token KV capacity budget.", "",
              "| Factorization | Models | Prefill / decode | Packed ms | Separate ms | Packing speed | Best Q4 ms | Speed vs Q4 | Peak allocated GB |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in c["runtime_rows"]:
        if r["status"] != "completed":
            lines.append(f"| {r['setting']} | {r['replicas']} | {r['status']} | — | — | — | — | — | — |")
            continue
        q4ms = f"{r['q4_best_wall_ms']:.2f}" if r['q4_best_wall_ms'] else "unavailable"
        speed = f"{r['speedup_vs_q4']:.3f}×" if r['speedup_vs_q4'] else "unavailable"
        lines.append(f"| {r['setting']} | {r['replicas']} | {r['length']} / {r['decode_steps']} | {r['packed_wall_ms']:.2f} | {r['separate_wall_ms']:.2f} | {r['packing_speedup']:.3f}× | {q4ms} | {speed} | {r['peak_allocated_bytes']/1e9:.3f} |")
    lines += ["", "The packing speed compares the same compressed weights in the same Python/FlashAttention runtime. The Q4 comparison includes runtime differences. Faster execution cannot rescue failed summary quality. A peak allocated measurement includes parameter storage, reserved KV capacity, live activations and workspace; it is not the entire CUDA process footprint.", "",
              "## Compression, memory and numerical controls", "", "| Factorization | Conversion seconds | Two models / one Q4 weight storage | Packing numerical gate |", "| --- | ---: | ---: | --- |"]
    for name, r in c["candidates"].items():
        lines.append(f"| {name} | {r['conversion_seconds']:.1f} | {r['two_models_to_one_q4_weight_ratio']:.3f}× | {r['gate']['packing_pass']} |")
    lines += ["", "| Q4 physical models | Weight GB | Resident GPU increment GB | Observed peak increment GB |", "| --- | ---: | ---: | ---: |"]
    for n, r in c["q4_runtime"].items():
        if r["status"] == "completed":
            lines.append(f"| {n} | {r['model_weight_bytes']/1e9:.3f} | {r['gpu_resident_increment_bytes']/1e9:.3f} | {r['observed_peak_gpu_increment_bytes']/1e9:.3f} |")
    lines += ["", "Q4 memory peaks were sampled after allocation and synchronized inference; transient peaks may be higher. Raw records include allocator reserved/allocated peaks, process/device snapshots, KV bytes, parity and individual timings. The unmodified FP16 28-layer implementation passed Hugging Face prefill and cached-decode controls after checkpoint save/reload.", "",
              "Packing is mathematically equivalent in exact arithmetic, but FP16 batched kernels need not be bitwise equal. The fixed logit gates are relative L2 ≤0.002, maximum absolute error ≤0.05 and top-token agreement ≥0.99. Failed numerical cells remain failures. KV differences are separately recorded.", "",
              "## Projection diagnostic", "", "This uses the deployed GGUF's actual first query projection, 3072×3072, dequantized before exact CUDA SVD. Two independent FP16 factor copies process 128 unequal token probes. This is a projection result only.", "",
              "| Rank | Two-copy factor bytes | One Q4 bytes | Retained energy | Relative weight error | Packing CUDA speed |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in c["projection"]:
        lines.append(f"| {r['rank']} | {r['factor_storage_bytes_two_models']:,} | {r['one_q4_projection_bytes']:,} | {100*r['retained_energy']:.2f}% | {100*r['reconstruction_relative_l2']:.2f}% | {r['packing_speedup']:.3f}× |")
    lines += ["", "## Boundaries and reproduction", ""] + ["- "+s+"." for s in c["limits"]]
    lines += ["", "The Q4 artifact matches the deployed file supplied over PBX by SHA-256 and byte count. Source snapshots, fixture/model hashes, pinned library versions and commands are retained in each producer manifest. The source checkpoint is [Meta Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct); the deployed artifact is [bartowski Q4_K_M](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF).", "",
              "```bash", "data/experiments/moba_paper/environment/bin/python -m experiments.structure_net.campaign_llama_cotenancy \\",
              "  --control data/experiments/llama_tensorized_cotenancy/20261010_control_v1 \\",
              "  --output data/experiments/llama_tensorized_cotenancy/NEW_CAMPAIGN_ROOT", "```", "",
              "The launcher selects physical GPU 0 before importing CUDA. It runs cells sequentially and preserves failed artifacts. Use `run_llama_cotenancy --help` for individual stages. Run the three focused `test_*llama*`/`test_model_major_tensor.py` suites before GPU work.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    campaign = build(args.campaign)
    artifact = args.campaign/"campaign_results.json"
    if artifact.exists() and read(artifact) != campaign:
        raise ValueError("preserve different completed aggregate")
    write_json(artifact, campaign)
    text = report(campaign, artifact)
    if REPORT.exists() and REPORT.read_text() != text:
        raise ValueError("preserve different report")
    REPORT.write_text(text)
    evidence = LlamaCotenancyEvidence(str(artifact), sha256(artifact), str(REPORT), sha256(REPORT), campaign["classification"], 4, len(campaign["summary"]["promoted"]))
    if args.register:
        from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
        logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses", queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected", enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
        record = evidence.record()
        logger.log_hypothesis(record)
        exact = logger.hypotheses_collection.get(ids=[record["id"]], include=["metadatas"])
        if exact["ids"] != [record["id"]] or any(exact["metadatas"][0].get(k) != record[k] for k in ("tested", "confirmed", "confirmation_status", "description")):
            raise RuntimeError("exact registry readback failed")
        semantic = logger.hypotheses_collection.query(query_texts=["Llama tensorized complete physical models SVD tensor train packing co-tenancy"], n_results=10)
        if record["id"] not in semantic["ids"][0]:
            raise RuntimeError("semantic registry readback failed")
        saved = {"schema": "nal.llama-cotenancy-registry.v1", "hypotheses": [record], "exact_readback_verified": True, "semantic_readback_verified": True}
        write_json(Path("data/hypothesis_registry")/(record["id"]+".json"), saved)
        write_json(args.campaign/"registry_readback.json", saved)
    print(json.dumps({"report": str(REPORT), "classification": campaign["classification"], "campaign_sha256": sha256(artifact), "registered": args.register}))


if __name__ == "__main__":
    main()
