"""Audit bitwise controls and timing, then report the actual deployment scope."""
import argparse
import json
from pathlib import Path
import statistics
import torch

from experiments.neural_architecture_lab.finalize_llama_cotenancy import audited
from experiments.structure_net.llama_cuda_graph import bitwise_tree
from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json
from neural_architecture_lab.llama_exact_deployment_hypothesis import ExactDeploymentEvidence

REPORT = Path("docs/08 - Analysis/2026-10-10_llama32-exact-deployment.md")


def validate_workloads(c, root):
    if c["status"] != "completed" or not c["uncontended"] or len(c["workloads"]) != 3:
        raise ValueError("completed uncontended full grid required")
    if {(r["length"], r["decode_steps"]) for r in c["workloads"]} != {(128, 0), (512, 0), (128, 32)}:
        raise ValueError("wrong workload grid")
    for row in c["workloads"]:
        if json.loads((root / f"workload_{row['length']}_{row['decode_steps']}.json").read_text()) != row:
            raise ValueError("workload artifact changed")
        path = Path(row["control_artifact"])
        if sha256(path) != row["control_sha256"]:
            raise ValueError("control artifact changed")
        controls = torch.load(path, map_location="cpu", weights_only=True)
        if len(controls) != 2 or torch.equal(controls[0]["ids"], controls[1]["ids"]):
            raise ValueError("two distinct correctness inputs required")
        if not all(v["bitwise_equal"] and bitwise_tree(v["actual"], v["expected"]) for v in controls) or not row["all_bitwise"]:
            raise ValueError("bitwise controls failed")
        expected_orders = [["eager", "cuda_graph"] if i % 2 == 0 else ["cuda_graph", "eager"] for i in range(10)]
        if row["orders"] != expected_orders or row["samples"] != 10:
            raise ValueError("timing was not ten alternating pairs")
        for arm in row["timings"].values():
            if len(arm["samples"]) != 10 or any(arm[k + "_median_ms"] != statistics.median(s[k + "_ms"] for s in arm["samples"]) for k in ("wall", "cuda")):
                raise ValueError("timing median or sample count changed")
        ratio = row["timings"]["eager"]["wall_median_ms"] / row["timings"]["cuda_graph"]["wall_median_ms"]
        if ratio != row["speedup"] or row["faster_single_model_gate"] != (ratio >= 1.10):
            raise ValueError("speed gate mismatch")
    if not c["all_bitwise"] or c["parameter_bytes"] != 6425499648 or any(r["parameter_bytes"] != c["parameter_bytes"] for r in c["workloads"]):
        raise ValueError("weight or exactness gate mismatch")
    passed = any(r["faster_single_model_gate"] for r in c["workloads"])
    if c["faster_single_model_supported"] != passed or c["two_for_one_supported"] or c["weight_compression_used"] or c["inference_arithmetic_changed"]:
        raise ValueError("unsupported deployment claim")


def report(c, aggregate):
    lines = ["# Llama 3.2 3B: exact deployment efficiency", "",
        f"Date: 2026-10-10. Status: completed. Profile: **EXPLORATORY**. Hypothesis `{c['hypothesis_id']}`.",
        f"Classification: **`{c['classification']}`**. Confirmation remains false for this one-checkpoint exploratory comparison.", "",
        f"Aggregate `{aggregate}`; SHA-256 `{sha256(aggregate)}`.",
        "[Fixed deployment plan](../07%20-%20Status%20Reports/2026-10-10_llama32-lossless-deployment-plan.md).", "",
        "CUDA graph replay speeds up some workloads of the complete unchanged 28-layer FP16 model. All measured paired outputs match bit for bit. The gain comes from reducing execution dispatch overhead; it does not establish low-rank tensor compression, a smaller residual network, fewer matrix operations, or two independent models in one model's memory budget.", "",
        "## Paired latency", "",
        "Ten alternating-order samples after warmup, one RTX 3060, two CPU threads. Wall medians include input copying and owned output cloning for graph replay. Eager and graph share the same one physical model as alternative execution paths.", "",
        "| Workload | Eager ms | Graph ms | Speed ratio | ≥1.10× gate |",
        "| --- | ---: | ---: | ---: | --- |"]
    for r in c["workloads"]:
        name = f"Prefill {r['length']}" + (f" + {r['decode_steps']} generated tokens" if r["decode_steps"] else "")
        lines.append(f"| {name} | {r['timings']['eager']['wall_median_ms']:.2f} | {r['timings']['cuda_graph']['wall_median_ms']:.2f} | {r['speedup']:.3f}× | {'pass' if r['faster_single_model_gate'] else 'fail'} |")
    lines += ["", "All individual wall/CUDA-event samples and their alternating execution orders are retained in the aggregate. The 512-token workload does not pass the speed gate. Generation produces 32 greedy tokens with EOS early termination disabled, exactly as in the matched eager arm.", "",
        "## Resident memory and setup", "",
        "Parameter storage is unchanged at **6.425 GB**. Peaks below describe one workload graph at a time; they do not measure retaining every possible sequence-length graph simultaneously.", "",
        "| Workload (prompt/generated) | Eager peak allocated GB | Graph peak allocated GB | Eager peak reserved GB | Graph peak reserved GB | Capture + warmup s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in c["workloads"]:
        a, b = r["eager_memory"], r["graph_memory"]
        lines.append(f"| {r['length']}/{r['decode_steps']} | {a['peak_allocated_bytes']/1e9:.3f} | {b['peak_allocated_bytes']/1e9:.3f} | {a['peak_reserved_bytes']/1e9:.3f} | {b['peak_reserved_bytes']/1e9:.3f} | {r['capture_and_warmup_seconds']:.3f} |")
    lines += ["", "Eager peak allocation was measured before graph capture. Graph figures include its retained pool and the returned output copy; capture-time peaks are also retained in the raw result. These are PyTorch allocator measurements, not whole-device memory accounting. Device/process snapshots were recorded and the external-process memory contention check passed. Existing unrelated GPU services were left unchanged.", "",
        "Capture/warmup is a startup cost excluded from steady-state replay latency. The graph accepts new token IDs of the captured shape; a different shape or generation length needs another capture. The steady-state speedup requires reuse. This comparison is against the repository's eager FP16 reference, not against every optimized inference engine or the deployed Q4 llama.cpp runtime.", "",
        "## Exactness and integrity", "",
        "The CUDA pilot checked prefill, cached decoding, generation, changed inputs, and output ownership on a tiny model; it then passed representative full-model memory and exactness checks. Pilot results are systems evidence only. The measured full run completed all 3 workloads, with 10 timing pairs each, zero failed or reused cells.", "",
        "Two unequal fixed input sequences were checked for each workload. Prefill controls compare every output-logit and KV-cache bit; generation controls compare every generated token. Both expected and actual tensors are retained in hashed `.pt` control artifacts, and the finalizer independently rechecked their bitwise equality. No training or floating-point reassociation was introduced.", "",
        "The exact reference checkpoint hash is `f01cc0653ec6c9bd7026b8676864612b91b6c79d2dc4df42599cbe4cf99eb9a9`. The producer includes source snapshots, plan/fixture hashes and dependency/GPU settings. That FP16 implementation was previously compared with Hugging Face using numerical tolerances; this is not a claim of bitwise agreement across different runtimes.", "",
        "The first full run (`20261010_v1`) was rejected by the finalizer: its nominally different control prompts shared the same first 128 tokens. Its timing results remain diagnostics and are not pooled here. The [input-diversity amendment](../07%20-%20Status%20Reports/2026-10-10_llama32-deployment-input-amendment.md) prospectively fixed the second control to use the other sequence's tail, added a per-workload inequality assertion and regression test, and required a new pilot/full run. This report uses only corrected `20261010_v2` evidence.", "",
        "## Requested deployment target", "",
        "- **Faster single model:** supported for the passing workloads above, relative to this eager reference, with the same parameter storage and unchanged arithmetic.",
        "- **Two regular independent models in one model's memory and latency:** not achieved or directly timed here. No parameter sharing is counted as independent-model compression.",
        "- **Tensorization-specific compute/compression advantage:** not established. Graph replay can also optimize regular dense execution.", "",
        "The separate [partial archive study](2026-10-10_llama32-lossless-compression.md) reduced original checkpoint storage to 4.588 GB with exact reconstruction. Two such archives already exceed one original model's 6.425 GB raw-weight budget, and decoding restores dense weights. Archive bytes are not treated as resident inference memory. The factor archive stage failed and no complete factors-plus-residuals archive total is claimed.", "",
        "No new application-accuracy measurement was made. Exact execution retains the prior model's answers and errors; the previous 10/12 synthetic validation result and unmeasured 95% application target remain unchanged. One model, one GPU and fixed workloads do not establish universal speedups.", "",
        "## Reproduction", "", "```bash",
        "data/experiments/moba_paper/environment/bin/python \\",
        "  -m experiments.structure_net.campaign_llama_cotenancy \\",
        "  --exact-deployment pilot \\",
        "  --control data/experiments/llama_tensorized_cotenancy/20261010_control_v1 \\",
        "  --output data/experiments/llama_lossless_deployment/NEW_PILOT_ROOT", "```", "",
        "After the pilot passes, run the same launcher with `--exact-deployment full` and a new output root. It selects physical GPU 0 before CUDA initialization. Finalize with `experiments.neural_architecture_lab.finalize_llama_exact_deployment --campaign <full-root>/deployment --register`. The reusable graph wrapper is `experiments/structure_net/llama_cuda_graph.py`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    result, producer = audited(args.campaign / "result.json")
    if producer["configuration"]["pilot"] or producer["configuration"]["samples"] != 10:
        raise ValueError("pilot cannot supply measured evidence")
    validate_workloads(result, args.campaign)
    classification = "exact_execution_speedup_supported_no_two_for_one" if result["faster_single_model_supported"] else "exact_execution_speedup_not_supported"
    c = {**result, "classification": classification, "profile": "EXPLORATORY", "requested": 3, "completed": 3, "failed": 0, "reused": 0,
         "source_result": str(args.campaign / "result.json"), "source_result_sha256": sha256(args.campaign / "result.json")}
    aggregate = args.campaign / "campaign_results.json"
    if aggregate.exists() and json.loads(aggregate.read_text()) != c:
        raise ValueError("preserve completed aggregate")
    write_json(aggregate, c)
    text = report(c, aggregate)
    if REPORT.exists() and REPORT.read_text() != text:
        raise ValueError("preserve completed report")
    REPORT.write_text(text)
    evidence = ExactDeploymentEvidence(str(aggregate), sha256(aggregate), str(REPORT), sha256(REPORT), c["all_bitwise"],
                                      tuple(r["speedup"] for r in c["workloads"]), c["parameter_bytes"])
    if args.register:
        from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
        logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
            queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
            enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
        record = evidence.record()
        logger.log_hypothesis(record)
        exact = logger.hypotheses_collection.get(ids=[record["id"]], include=["metadatas"])
        # Chroma indexes the logger's canonical research metadata; the complete
        # typed record, including branch-specific booleans, is saved below.
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
            raise RuntimeError("complete typed JSON registry readback failed")
    print(json.dumps({"classification": classification, "report": str(REPORT), "registered": args.register}))


if __name__ == "__main__":
    main()
