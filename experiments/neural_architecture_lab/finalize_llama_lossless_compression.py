"""Audit the full archive census, report, and conservatively register storage results."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import run_llama_lossless_compression as study
from neural_architecture_lab.llama_lossless_compression_hypothesis import LosslessCompressionEvidence, PartialCompressionEvidence

codec = study.codec
REPORT = Path("docs/08 - Analysis/2026-10-10_llama32-lossless-compression.md")


def partial(root):
    original = study.read(root / "original/result.json")
    failure = study.read(root / "failure.json")
    producer = study.read(root / "producer.json")
    expected = producer.pop("fingerprint")
    if study.fingerprint(producer) != expected or failure["error"] != "CPU contiguous tensors required":
        raise ValueError("unexpected partial campaign or producer")
    for path, digest in producer["sources"].items():
        copies = list((root / "producer_source").glob("*_" + Path(path).name))
        if len(copies) != 1 or codec.digest_file(copies[0]) != digest:
            raise ValueError("frozen partial producer changed")
    manifest = codec.read_manifest(root / "original")
    if codec.digest_file(root / "original/manifest.json") != original["manifest_sha256"] or manifest["payload_sha256"] != original["payload_sha256"]:
        raise ValueError("verified original archive changed")
    if len(manifest["tensors"]) != 254 or not original["bitwise_equal"]:
        raise ValueError("incomplete original checkpoint archive")
    target = root / "campaign_results.json"
    c = {"schema": study.SCHEMA, "hypothesis_id": study.HYPOTHESIS_ID,
         "status": "closed_partial_after_runtime_fault_and_scope_redirection",
         "classification": "original_archive_verified_factor_stage_failed",
         "producer_fingerprint": expected, "requested": 199, "completed": 1, "failed": 1, "not_run": 197,
         "original": original, "failure": failure, "full_campaign_completed": False,
         "factor_residual_total": None, "factor_residual_total_reason": "factor stage failed; residual archive stage not run",
         "inference_speed_tested": False, "two_for_one_supported": False,
         "two_compressed_originals_vs_one_raw": 2 * original["archive_bytes"] / original["raw_bytes"],
         "continuation": "llama32-3b-lossless-deployment-efficiency-v1",
         "implementation_correction": "After this run, tensor_bytes canonicalizes non-contiguous CPU tensor storage; original logical bits/shapes/dtypes are unchanged. The failed campaign was not rerun or relabeled complete."}
    if target.exists() and study.read(target) != c:
        raise ValueError("preserve existing partial aggregate")
    codec.write_json(target, c)
    text = f"""# Llama 3.2 3B: partial lossless archive result

Date: 2026-10-10. Hypothesis `{study.HYPOTHESIS_ID}`. Profile: **EXPLORATORY**.
Status: closed partial after a runtime fault and user redirection to deployment.
Classification: **`{c['classification']}`**. The planned full component census did not complete.

Aggregate `{target}`; SHA-256 `{codec.digest_file(target)}`.
[Original plan](../07%20-%20Status%20Reports/2026-10-10_llama32-lossless-compression-plan.md).

The complete original model compressed from **{original['raw_bytes']/1e9:.3f} GB** of tensors to **{original['archive_bytes']/1e9:.3f} GB**, a **{100*(1-original['archive_bytes']/original['raw_bytes']):.2f}%** storage reduction. All 197 matrices and 57 normalization vectors decoded bit for bit after persistence. The archive includes model metadata and can be decoded without the original checkpoint. The tied embedding/output matrix is stored once.

This does **not** meet the clarified two-for-one deployment goal. Two independent archives alone occupy **{2*original['archive_bytes']/1e9:.3f} GB**, **{c['two_compressed_originals_vs_one_raw']:.3f}×** one original model's raw parameter budget, before decoding and runtime memory. Dense inference still needs the original weights.

The selected original layout was adjacent-value XOR plus bit shuffle, then Zstandard level 3, in 1 MiB chunks. Encode/write/hash took **{original['encode_seconds']:.1f} seconds**; decode/read/hash/direct verification took **{original['decode_verify_seconds']:.1f} seconds**. These are descriptive CPU costs, not inference measurements.

| Original component | Raw GB | Encoded payload GB |
| --- | ---: | ---: |
"""
    for name, row in original["families"].items():
        text += f"| {name} | {row['raw_bytes']/1e9:.6f} | {row['payload_bytes']/1e9:.6f} |\n"
    text += f"""
The total includes manifest bytes; family rows show payload only. Experiment logs, source snapshots and filesystem block rounding are excluded.

Of 199 planned archives, **1 completed, 1 failed, and 197 were not run**. The factor archive stopped on a non-contiguous SVD core. Its partial payload and `failure.json` remain unchanged. A later implementation correction canonicalizes tensor storage before byte encoding; logical values, shapes and dtypes are preserved, while physical strides are not an archive contract. A focused regression test covers this case. No full factors-plus-residuals total is claimed.

The screen in `screen.json` remains exploratory evidence only. The original lowest three FP16 bit planes were nearly empty; factor bit planes were much more evenly occupied. Full-model factor/residual savings cannot be inferred from those slices. The original archive result is unaffected by the later factor-stage failure.

Input hashes were verified against the previous exact-residual study. Producing source and plan copies are retained under `producer_source/`. Finalization verified the original archive manifest/payload hashes and frozen source fingerprint. The archive preserves original accuracy; the earlier 10/12 validation result and unmeasured 95% application target remain unchanged.

The user clarified that success means resident GPU memory and execution speed. The next study is [the exact deployment experiment](../07%20-%20Status%20Reports/2026-10-10_llama32-lossless-deployment-plan.md), with no archival-only result counted as deployment success.

Decode the verified archive with:

```bash
env PYTHONPATH=.:src data/experiments/moba_paper/environment/bin/python \\
  -m experiments.structure_net.run_llama_lossless_compression \\
  --decode {root}/original --output /tmp/restored_llama.pt
```

Finalize this partial record with `experiments.neural_architecture_lab.finalize_llama_lossless_compression --campaign {root} --partial --register`. A new complete archival run would require a new output root.
"""
    if REPORT.exists() and REPORT.read_text() != text:
        raise ValueError("preserve existing report")
    REPORT.write_text(text)
    evidence = PartialCompressionEvidence(str(target), codec.digest_file(target), str(REPORT), codec.digest_file(REPORT),
                                         original["raw_bytes"], original["archive_bytes"])
    return c, evidence


def audit(root):
    c = study.read(root / "campaign_results.json")
    producer = study.read(root / "producer.json")
    saved_fingerprint = producer.pop("fingerprint")
    if study.fingerprint(producer) != saved_fingerprint or c["producer_fingerprint"] != saved_fingerprint:
        raise ValueError("producer fingerprint mismatch")
    if c["producer_sha256"] != codec.digest_file(root / "producer.json") or c["screen_sha256"] != codec.digest_file(root / "screen.json"):
        raise ValueError("campaign input hash mismatch")
    for i, (path, expected) in enumerate(producer["sources"].items()):
        # JSON key sorting changes order; match preserved copies by basename.
        matches = list((root / "producer_source").glob("*_" + Path(path).name))
        if len(matches) != 1 or codec.digest_file(matches[0]) != expected:
            raise ValueError("frozen producing source mismatch")
    if c["status"] != "completed" or c["completed"] != 199 or c["requested"] != 199 or c["failed"] or c["reused"] or producer["shakedown"]:
        raise ValueError("incomplete full census")
    inputs = {r["id"]: r for r in producer["inputs"]}
    for record in c["records"]:
        path = Path(record["directory"])
        if study.read(path / "result.json") != record or record["sha256"] != inputs[record["id"]]["sha256"]:
            raise ValueError("archive result or input identity mismatch")
        if codec.digest_file(path / "manifest.json") != record["manifest_sha256"]:
            raise ValueError("archive manifest mismatch")
        manifest = codec.read_manifest(path)
        if manifest["payload_sha256"] != record["payload_sha256"]:
            raise ValueError("archive payload identity mismatch")
        if sum((path / n).stat().st_size for n in ("manifest.json", "payload.bin")) != record["archive_bytes"]:
            raise ValueError("archive size mismatch")
        if sum(e["numel"] * e["width"] for e in manifest["tensors"]) != record["raw_bytes"] or len(manifest["tensors"]) != record["tensor_count"]:
            raise ValueError("tensor census mismatch")
    summary = study.summarize(c["records"])
    if any(c[key] != value for key, value in summary.items()):
        raise ValueError("aggregate gate or totals mismatch")
    return c


def report(c, root):
    dense = c["groups"]["original"]
    lines = ["# Llama 3.2 3B: lossless component compression", "",
        f"Date: 2026-10-10. Status: completed. Hypothesis `{study.HYPOTHESIS_ID}`. Profile: **EXPLORATORY**.",
        f"Classification: **`{c['classification']}`**; confirmation remains false for this one-checkpoint exploratory study.", "",
        f"Aggregate `{root / 'campaign_results.json'}`; SHA-256 `{codec.digest_file(root / 'campaign_results.json')}`.", "",
        "[Fixed exploratory plan](../07%20-%20Status%20Reports/2026-10-10_llama32-lossless-compression-plan.md).",
        "The complete archives preserve every tensor bit and typed checkpoint metadata. Decode restores the original dense tensors. No training or new accuracy evaluation was performed.", "",
        "| Component | Raw tensor GB | Archive GB | Savings | Selected layout |",
        "| --- | ---: | ---: | ---: | --- |"]
    for name, row in c["groups"].items():
        lines.append(f"| {name} | {row['raw_bytes']/1e9:.3f} | {row['archive_bytes']/1e9:.3f} | {100*(1-row['fraction_of_raw']):.2f}% | {c['selected_layouts'][name]} |")
    combined = c["compressed_factors_plus_residuals_bytes"]
    lines += ["", f"Compressed factors plus exact residuals total **{combined/1e9:.3f} GB**, or **{c['combined_vs_compressed_original']:.2f}×** the compressed original model. Factors alone remain a lossy model and are not an eligible lossless replacement.", "",
        "All numbers include actual payload and manifest bytes. Experiment result logs, producer-source copies and filesystem allocation rounding are excluded. Full-model figures come from all encoded chunks, not sample extrapolation.", "",
        "## Integrity and coverage", "",
        f"Requested/completed/failed/reused archives: **199 / {c['completed']} / {c['failed']} / {c['reused']}**. Total tensors: **{sum(g['tensor_count'] for g in c['groups'].values())}**. Every persisted chunk was reopened, decoded and compared directly with its source bytes. All passed: **{c['all_tensors_bitwise']}**.", "",
        "The census includes all 197 original matrices, 57 normalization vectors, both cores for every factorized matrix, all 197 FP32 residual matrices, and their exact repair indices/values. The embedding/output matrix is tied and stored once. Standalone decoding preserves integer dictionary keys, tensor shapes, dtypes and checkpoint configuration.", "",
        "Input checkpoint hashes and every residual shard hash were verified against the completed parent study before encoding. Producing source and the plan were content-hashed and copied into the artifact root. Finalization checks frozen sources, all manifests and payload hashes, the complete census, and recomputed size gates.", "",
        "## Component breakdown", "",
        "Payload only below; manifest overhead is included in the total table above.", "",
        "| Group / family | Raw GB | Encoded payload GB |", "| --- | ---: | ---: |"]
    families = {}
    for record in c["records"]:
        for name, row in record["families"].items():
            target = families.setdefault((record["group"], name), {"raw_bytes": 0, "payload_bytes": 0})
            for key, value in row.items():
                target[key] += value
    for (group, name), row in sorted(families.items()):
        lines.append(f"| {group} / {name} | {row['raw_bytes']/1e9:.6f} | {row['payload_bytes']/1e9:.6f} |")
    screen = study.read(root / "screen.json")
    lines += ["", "## Selection and cost", "",
        "Fixed first/middle/last samples from embedding and layers 0, 14, 27, plus all norms/repairs, screened four reversible layouts with Zstandard level 3. The smallest summed payload chose one layout per group. Raw fallback avoids payload expansion. This is exploratory selection on the same checkpoint, not a held-out model comparison.", "",
        "| Group | Raw-layout Zstd MB | Byte shuffle MB | Bit shuffle MB | XOR + bit shuffle MB |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for group, sizes in screen["totals"].items():
        lines.append("| " + group + " | " + " | ".join(f"{sizes[k]/1e6:.3f}" for k in codec.LAYOUTS) + " |")
    original_bits = screen["bit_occupancy"]["original/torch.float16"]
    factor_bits = screen["bit_occupancy"]["factors/torch.float16"]
    original_fractions = [100 * n / original_bits["count"] for n in original_bits["one_counts_little_endian"][:3]]
    factor_fractions = [100 * n / factor_bits["count"] for n in factor_bits["one_counts_little_endian"][:3]]
    lines += ["", "On the fixed screening slices, the lowest three original FP16 bit planes were set in "
        + ", ".join(f"{p:.3f}%" for p in original_fractions) + " of values. The corresponding factor bit planes were set in "
        + ", ".join(f"{p:.3f}%" for p in factor_fractions) + ". This diagnoses substantial redundancy in the original representation; all exceptional bits are still retained. These frequencies describe the screen, not an exhaustive bit census.", "",
        "| Group | Encode/write/hash seconds | Decode/read/hash/verify seconds |",
        "| --- | ---: | ---: |"]
    for group, row in c["groups"].items():
        lines.append(f"| {group} | {row['encode_seconds']:.2f} | {row['decode_verify_seconds']:.2f} |")
    lines += ["", "Single CPU process, one Torch thread, 1 MiB chunks. These are descriptive wall times including file I/O and integrity checks, not isolated throughput or inference benchmarks. CUDA was not used; this CPU-only cell follows the plan.", "",
        "## Boundaries", "", *["- " + value for value in c["limitations"]], "",
        f"After decoding, the original model still occupies {dense['raw_bytes']/1e9:.3f} GB in weight tensors. No lower-rank exact neural network or faster dense operator was established. The earlier 10/12 validation result and unmeasured 95% application target are unchanged.", "",
        "## Reproduction", "", "```bash",
        "env PYTHONPATH=.:src data/experiments/moba_paper/environment/bin/python \\",
        "  -m experiments.structure_net.run_llama_lossless_compression \\",
        "  --output data/experiments/llama_lossless_compression/NEW_ROOT", "",
        "env PYTHONPATH=.:src data/experiments/moba_paper/environment/bin/python \\",
        "  -m experiments.structure_net.run_llama_lossless_compression \\",
        f"  --decode {root}/original --output /tmp/restored_llama.pt", "```", "",
        "The decode command uses only the archive. Use `--shakedown` with a new output root for the lifecycle pilot; finalize with `experiments.neural_architecture_lab.finalize_llama_lossless_compression --campaign <root> --register`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--partial", action="store_true", help="finalize the preserved original-only runtime-fault result")
    args = parser.parse_args()
    if args.partial:
        c, evidence = partial(args.campaign)
    else:
        c = audit(args.campaign)
        text = report(c, args.campaign)
        if REPORT.exists() and REPORT.read_text() != text:
            raise ValueError("preserve existing report")
        REPORT.write_text(text)
        evidence = LosslessCompressionEvidence(str(args.campaign / "campaign_results.json"),
            codec.digest_file(args.campaign / "campaign_results.json"), str(REPORT), codec.digest_file(REPORT),
            c["classification"], c["completed"], c["all_tensors_bitwise"], c["groups"]["original"]["raw_bytes"],
            c["groups"]["original"]["archive_bytes"], c["compressed_factors_plus_residuals_bytes"])
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
        codec.write_json(Path("data/hypothesis_registry") / (record["id"] + ".json"), saved)
        codec.write_json(args.campaign / "registry_readback.json", saved)
    print(json.dumps({"classification": c["classification"], "report": str(REPORT), "registered": args.register}))


if __name__ == "__main__":
    main()
