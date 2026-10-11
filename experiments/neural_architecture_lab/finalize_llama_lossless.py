"""Audit exact residual artifacts, then report and read back the registry."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import torch

from experiments.neural_architecture_lab.finalize_llama_cotenancy import audited, read, digest
from experiments.structure_net.llama_cotenancy_data import score
from experiments.structure_net.run_llama_cotenancy import aggregate_quality
from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json
from neural_architecture_lab.llama_lossless_residual_hypothesis import LosslessResidualEvidence
from experiments.structure_net.llama_lossless_residual import bits

REPORT = Path("docs/08 - Analysis/2026-10-10_llama32-lossless-residual.md")


def aggregate(root,timing_path):
    index=read(root/"campaign_index.json")
    if index["status"] != "completed" or {r["name"] for r in index["cells"]} != {"projection","weights","original_pilot","residual_pilot","original","exact_residual"} or len(index["cells"]) != 6:
        raise ValueError("incomplete lossless campaign")
    cells={row["name"]:audited(row["result"],row["sha256"])[0] for row in index["cells"]}
    weights=cells["weights"]
    original_manifest=read(root/"original/producer.json")
    source_path=original_manifest["configuration"]["checkpoint"]
    if digest(source_path)!=original_manifest["configuration"]["checkpoint_sha256"]:
        raise ValueError("original reference checkpoint changed")
    source_payload=torch.load(source_path,map_location="cpu",weights_only=False,mmap=True)
    decoded_payload=torch.load(weights["checkpoint"],map_location="cpu",weights_only=False,mmap=True)
    if source_payload["config"]!=decoded_payload["config"] or source_payload["vectors"].keys()!=decoded_payload["vectors"].keys() or source_payload["matrices"].keys()!=decoded_payload["matrices"].keys():
        raise ValueError("model architecture or parameter names changed during decoding")
    vector_checks={name:torch.equal(bits(value),bits(decoded_payload["vectors"][name])) for name,value in source_payload["vectors"].items()}
    if len(vector_checks)!=57 or not all(vector_checks.values()):
        raise ValueError("normalization vector changed during lossless checkpoint conversion")
    del source_payload,decoded_payload
    if weights["matrix_count"] != 197 or len(weights["weights"]) != 197 or len({r["name"] for r in weights["weights"]}) != 197:
        raise ValueError("matrix audit incomplete")
    for row in weights["weights"]:
        if not row["bitwise_reconstruction"] or row["source_weight_sha256"] != row["reconstructed_sha256"] or digest(row["residual_path"]) != row["residual_sha256"]:
            raise ValueError("lossless residual weight artifact changed")
    for name in ("original_pilot","residual_pilot","original","exact_residual"):
        if digest(root/name/"logit_controls.pt") != cells[name]["logit_controls_sha256"]:
            raise ValueError("logit control changed")
    for reference, corrected in (("original_pilot","residual_pilot"),("original","exact_residual")):
        a=torch.load(root/reference/"logit_controls.pt",map_location="cpu",weights_only=True)
        b=torch.load(root/corrected/"logit_controls.pt",map_location="cpu",weights_only=True)
        checks={length:{kind:torch.equal(bits(value),bits(a[length][kind])) for kind,value in row.items()} for length,row in b.items()}
        if cells[corrected]["logit_checks"] != checks:
            raise ValueError("bitwise logit gate does not match stored tensors")
    original,residual=cells["original"],cells["exact_residual"]
    fixtures=read(next((root/"original/producer_source").glob("*_fixtures.json")))
    by_id={r["id"]:r for r in fixtures["examples"]}
    for result in (original,residual):
        for split in ("validation","test"):
            if split not in result:
                continue
            expected={r["id"] for r in fixtures["examples"] if r["split"]==split}
            if {r["id"] for r in result[split]} != expected or len(result[split]) != len(expected):
                raise ValueError("quality split incomplete")
            for row in result[split]:
                if row["metrics"] != score(row["text"],by_id[row["id"]]):
                    raise ValueError("source fragment scoring mismatch")
            if result[split+"_summary"] != aggregate_quality(result[split]):
                raise ValueError("quality summary mismatch")
    generation_checks={split:{r["id"]:r["generated_ids"]==next(e["generated_ids"] for e in original[split] if e["id"]==r["id"]) for r in residual[split]}
                       for split in ("validation","test") if split in residual}
    if generation_checks != residual["generation_checks"]:
        raise ValueError("paired generation identity mismatch")
    all_outputs=residual["all_control_logits_bitwise"] and all(v for rows in generation_checks.values() for v in rows.values())
    timing,timing_producer=audited(timing_path)
    if timing_producer["configuration"]["checkpoint_sha256"]!=weights["checkpoint_sha256"] or not timing["uncontended"]:
        raise ValueError("corrected timing changed checkpoint or encountered contention")
    timings=timing["paired_timings"]
    if len(timings)!=3 or {(r["length"],r["decode_steps"]) for r in timings}!={(128,0),(512,0),(128,32)}:
        raise ValueError("paired timing grid incomplete")
    for row in timings:
        expected_orders=[["original","exact_residual"] if i%2==0 else ["exact_residual","original"] for i in range(5)]
        if row["orders"]!=expected_orders:
            raise ValueError("two-arm timing order did not alternate")
        for arm in row["timings"].values():
            if len(arm["samples"]) != 5 or any(arm[k+"_median_ms"] != statistics.median(s[k+"_ms"] for s in arm["samples"]) for k in ("wall","cuda")):
                raise ValueError("paired timing mismatch")
        if row["speedup"] != row["timings"]["original"]["wall_median_ms"]/row["timings"]["exact_residual"]["wall_median_ms"]:
            raise ValueError("speed ratio mismatch")
    exact=weights["all_weights_bitwise"] and all_outputs and cells["residual_pilot"]["all_control_logits_bitwise"] and cells["projection"]["bitwise_exact_control"]
    efficient=any(r["speedup"] >= 1.1 for r in timings) or residual["parameter_bytes"] < original["parameter_bytes"]
    if not exact:
        classification="lossless_not_preserved"
    elif not original["validation_competent"]:
        classification="lossless_accuracy_assay_invalid"
    else:
        classification="lossless_efficiency_supported" if efficient else "lossless_preserved_no_efficiency_gain"
    if not all(r["uncontended"] for r in cells.values()):
        raise ValueError("external contention requires a separately reported timing qualification")
    return {"schema":"nal.llama-lossless-residual-aggregate.v1","hypothesis_id":"llama32-3b-lossless-residual-cotenancy-v1",
            "status":"completed","classification":classification,"all_weights_bitwise":weights["all_weights_bitwise"],"all_outputs_bitwise":all_outputs,
            "exactness_pass":exact,"efficiency_pass":efficient,"accuracy_gate":original["validation_competent"],
            "matrix_count":197,"repair_entries":weights["repair_entries"],"storage_bytes_per_model":weights["storage_bytes_per_model"],
            "normalization_vector_checks":vector_checks,
            "capacity_storage_lower_bound":weights["capacity_storage_lower_bound"],"gpu_total_bytes":weights["gpu_total_bytes"],"gpu_free_bytes_at_audit":weights["gpu_free_bytes"],
            "validation_original":original["validation_summary"],"validation_residual":residual["validation_summary"],
            "test_original":original.get("test_summary"),"test_residual":residual.get("test_summary"),"generation_checks":generation_checks,
            "paired_timings":timings,"timing_artifact":str(timing_path),"timing_artifact_sha256":sha256(timing_path),
            "fixed_order_timing_diagnostic":residual["paired_original_vs_residual"],
            "memory":{name:{"parameter_bytes":r["parameter_bytes"],"benchmarks":r["benchmarks"]} for name,r in (("original",original),("exact_residual",residual))},
            "projection":cells["projection"],"weight_audit_seconds":weights["seconds"],"cells":index["cells"],"campaign_index_sha256":sha256(root/"campaign_index.json"),
            "source_checkpoint_sha256":read(root/"original/producer.json")["configuration"]["original_sha256"],
            "decoded_checkpoint":weights["checkpoint"],"decoded_checkpoint_sha256":weights["checkpoint_sha256"],
            "trained_parameters":0,"application_accuracy_tested":False,
            "limits":["The exact residual execution control retains and evaluates the full dense operator; it is not a compressed correction network.",
                      "The stored FP32 matrix residual construction exceeds device capacity before activations; capacity is a byte lower bound, not an attempted OOM.",
                      "Four validation and ten final synthetic extraction examples are not production summary accuracy.",
                      "One RTX 3060 and one Llama checkpoint; these constructions do not rule out other lossless representations."]}


def report(c,path):
    storage=c["storage_bytes_per_model"]
    lines=["# Llama 3.2 3B: lossless tensorization with exact residual correction","",
           f"Date: 2026-10-10. Hypothesis `{c['hypothesis_id']}`. Status **completed**. Classification **`{c['classification']}`**.","",
           f"Aggregate `{path}`. SHA-256 `{sha256(path)}`.","",
           "The exact correction preserved the source weights and the measured model behavior. It retained the original dense operator to compute the required output residual, so it did not eliminate that operator's storage or work. This is an exact execution control, not evidence that a smaller exact residual network was discovered.","",
           f"All **{c['matrix_count']}/197 matrices** reconstructed bit for bit after serialization and decoding; all 57 normalization vectors also matched. The codec needed **{c['repair_entries']:,} exceptional-entry repairs** in addition to FP32 residuals and FP16 factors. The measured prefill/cached-decode logits matched the original FP16 reference bit for bit: **{c['all_outputs_bitwise']}**. Generated token sequences were also checked pairwise.","",
           "## Accuracy","", "The task copies three exact final-decision sentences from notes containing superseded proposals. A fixed worked example demonstrates the format. These are synthetic fixtures, not application data.","",
           "| Split / model | Exact fragments | Supported lines | Complete summaries | Reference NLL |","| --- | ---: | ---: | ---: | ---: |"]
    for split in ("validation","test"):
        for arm in ("original","residual"):
            q=c[split+"_"+arm]
            if q:
                lines.append(f"| {split} / {arm} | {q['covered_fragments']}/{q['total_fragments']} | {q['supported_lines']}/{q['output_lines']} | {q['complete_summaries']}/{q['examples']} | {q['reference_nll_mean']:.6f} |")
    lines += ["", "The original FP16 implementation was previously checked against Hugging Face with FP16 numerical tolerances. Bitwise equality here is between that frozen FP16 implementation and its exact residual counterpart, not a claim of bitwise equality between different runtimes.","",
              ("The sealed ten-example final test was not run because the original model failed the fixed validation competence gate. Exact correction preserves its validation answers, including errors; it cannot increase source-model accuracy while also preserving the same function. The 95% application-accuracy target is unmeasured." if not c["test_original"] else "The ten-example final comparison is reported separately from validation. This synthetic assay does not establish application accuracy."),"",
              "## Complete-model speed and storage","", "Paired wall-time medians, five alternating samples after warmup, same physical model and GPU. Both execution alternatives use the same dense weights; this is a comparison of one model's two execution paths, not shared weights between independent replicas. The residual path also executes rank-215 factors, subtraction, addition and bitwise rounding repair.","",
              "| Workload | Original ms | Exact residual ms | Original / residual speed |","| --- | ---: | ---: | ---: |"]
    for r in c["paired_timings"]:
        lines.append(f"| Prefill {r['length']}, decode {r['decode_steps']} | {r['timings']['original']['wall_median_ms']:.2f} | {r['timings']['exact_residual']['wall_median_ms']:.2f} | {r['speedup']:.3f}× |")
    lines += ["", "| Representation | Parameter GB / model | Relative to original FP16 |","| --- | ---: | ---: |"]
    for name,size in storage.items():
        lines.append(f"| {name} | {size/1e9:.3f} | {size/storage['original_dense']:.3f}× |")
    lines += ["",f"Device memory: **{c['gpu_total_bytes']/1e9:.3f} GB**, with **{c['gpu_free_bytes_at_audit']/1e9:.3f} GB** free at the storage audit. The capacity table in the aggregate records exact 1/2/4-copy storage bounds. KV caches, activations and runtime allocations require additional space.","",
              "The stored matrix residual uses P = U·V in FP32, R = W − P, and exact repairs for any entries where FP16(P + R) differs from W. Decoding materializes the original dense matrix; doing so preserves its arithmetic but keeps dense inference cost. Directly adding factor and matrix-residual outputs is a separate floating-point reassociation and is not automatically bitwise lossless.","",
              f"Primary timings come from `{c['timing_artifact']}` (SHA-256 `{c['timing_artifact_sha256']}`). A preregistered timing amendment corrected a helper whose two-arm rotation/reversal accidentally repeated the same order. Original fixed-order results remain in the aggregate as diagnostics; the primary table uses explicitly alternating orders.","",
              "| Execution | Maximum measured allocator peak GB |","| --- | ---: |"]
    for name,r in c["memory"].items():
        peak=max(b["memory"]["packed"]["peak_allocated_bytes"] for b in r["benchmarks"])
        lines.append(f"| {name} | {peak/1e9:.3f} |")
    p=c["projection"]
    lines += ["", "## Independent-model packing diagnostic","",
              f"Two physical first-query projections, 128 unequal input tokens each. Exact residual correction versus original packed dense output was bitwise equal: **{p['bitwise_exact_control']}**. The direct factor-plus-stored-matrix-residual sum was bitwise equal: **{p['direct_residual_sum_bitwise']}**. Its measured relative error was {p['direct_residual_sum_parity']['relative_l2']:.6g}; it is not promoted as a lossless path.","",
              "| Projection execution | Wall median ms |","| --- | ---: |"]
    for name,r in p["timings"].items():
        lines.append(f"| {name} | {r['wall_median_ms']:.4f} |")
    lines += ["", "## Scope and prior work","",
              "The user explicitly redirected the parent lossy screen to exact correction. That screen was stopped and remains incomplete; its completed artifacts were preserved. Its original Q4 quality assay failed its competence gate (8/12 fragments), while original FP16 scored 12/12. A prospectively declared worked-example correction supplied the fresh fixtures used here. No claim is made that the original invalid assay became valid.","",
              "The preserved parent runtime did fit four independent copies of the deployed Q4_K_M artifact on this GPU. Q4 uses quantized weights and a different runtime; it is not a lossless representation of the original FP16 model. Its exact identity, GPU offload and private contexts are recorded in `20261010_campaign_v2/gguf_benchmark_m4/`.","",
              *["- "+s for s in c["limits"]],"",f"Residual construction, verification and checkpoint writing took {c['weight_audit_seconds']:.1f} seconds. Full residual shards and the decoded checkpoint remain under the campaign's `weights/` directory; every shard and producing source is hashed.","",
              "## Reproduction","","```bash",
              "data/experiments/moba_paper/environment/bin/python -m experiments.structure_net.campaign_llama_cotenancy \\",
              "  --lossless --control data/experiments/llama_tensorized_cotenancy/20261010_control_v1 \\",
              "  --output data/experiments/llama_tensorized_cotenancy/NEW_LOSSLESS_ROOT","```","",
              "Then run `campaign_llama_cotenancy --lossless-timing <new-root>/weights/decoded_model.pt --output <new-timing-root> --control data/experiments/llama_tensorized_cotenancy/20261010_control_v1` through the same Python module invocation. Finalize with `experiments.neural_architecture_lab.finalize_llama_lossless --campaign <new-root> --timing <new-timing-root>/timing/result.json --register`.","",
              "The launcher selects GPU 0 before CUDA initialization and runs one producer at a time. Fixtures, original checkpoint, fixed factors, protocol and dependency hashes are recorded in each producer. No model training was performed.",""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",type=Path,required=True)
    parser.add_argument("--timing",type=Path,required=True)
    parser.add_argument("--register",action="store_true")
    args=parser.parse_args()
    value=aggregate(args.campaign,args.timing)
    target=args.campaign/"campaign_results.json"
    if target.exists() and read(target) != value:
        raise ValueError("preserve different completed aggregate")
    write_json(target,value)
    text=report(value,target)
    if REPORT.exists() and REPORT.read_text()!=text:
        raise ValueError("preserve different report")
    REPORT.write_text(text)
    evidence=LosslessResidualEvidence(str(target),sha256(target),str(REPORT),sha256(REPORT),value["classification"],197,
                                      value["test_original"]["examples"] if value["test_original"] else 0,value["all_weights_bitwise"],value["all_outputs_bitwise"])
    if args.register:
        from structure_net.logging.standardized_logging import LoggingConfig,StandardizedLogger
        logger=StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",queue_dir="data/experiment_queue",sent_dir="data/experiment_sent",rejected_dir="data/experiment_rejected",enable_wandb=False,auto_upload=False,enable_chromadb=True,chromadb_path="data/chroma_db"))
        record=evidence.record()
        logger.log_hypothesis(record)
        exact=logger.hypotheses_collection.get(ids=[record["id"]],include=["metadatas"])
        if exact["ids"] != [record["id"]] or any(exact["metadatas"][0].get(k)!=record[k] for k in ("tested","confirmed","confirmation_status","description")):
            raise RuntimeError("exact registry readback failed")
        semantic=logger.hypotheses_collection.query(query_texts=["Llama lossless tensorization exact residual correction bitwise full model"],n_results=10)
        if record["id"] not in semantic["ids"][0]:
            raise RuntimeError("semantic registry readback failed")
        saved={"schema":"nal.llama-lossless-registry.v1","hypotheses":[record],"exact_readback_verified":True,"semantic_readback_verified":True}
        write_json(Path("data/hypothesis_registry")/(record["id"]+".json"),saved)
        write_json(args.campaign/"registry_readback.json",saved)
    print(json.dumps({"report":str(REPORT),"classification":value["classification"],"registered":args.register,"campaign_sha256":sha256(target)}))


if __name__ == "__main__":
    main()
