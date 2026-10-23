"""Bind the repository comparison to pinned sources and existing outputs.

This offline audit does not train or generate. It replays descriptive ROUGE
over the same 32 already generated public XSum examples.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path("data/experiments/ncm_paper_regime")
AUDIT = ROOT / "20261023_repository_training_audit_v2"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-reference-repository-comparison.md")


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = AUDIT / "comparison.json"
    if output.exists():
        raise ValueError("completed comparison is append-only")
    manifest = read(AUDIT / "manifest.json")
    for request in manifest["requests"]:
        if sha256(Path(request["path"])) != request["sha256"]:
            raise ValueError("upstream metadata differs from snapshot")
    source_records = read(AUDIT / "source_manifest.json") + read(AUDIT / "supplement_source_manifest.json")
    for record in source_records:
        if "error" in record:
            raise ValueError("missing source download")
        # PR 7 was preserved separately from the fork's unchanged main branch.
        directory = (AUDIT / "pr7_source" if record["commit"] == "d7377f42dfb5661a22d4b98a4c50f5420965111c"
                     else AUDIT / "sources" / record["repo"].replace("/", "--"))
        if sha256(directory / record["path"]) != record["sha256"]:
            raise ValueError("downloaded source differs from pinned hash")
    ncm = manifest["repos"]["HazyResearch/ncm"]
    if ncm["truncated"] or len(ncm["files"]) != 39 or any("train" in p.lower() for p in ncm["files"]):
        raise ValueError("review NCM training-availability conclusion")

    original = ROOT / "20261022_released_xsum_control_v2"
    outputs = read(original / "outputs.json")
    control = read(original / "result.json")
    if (not control["author_local_exact"] or control["cases"] != 32 or len(outputs) != 32
            or [r["index"] for r in outputs] != list(range(32))
            or not all(r["text"] == r["author_text"] and r["author_text"].strip() for r in outputs)):
        raise ValueError("original 32-case inference control no longer agrees")
    sys.path.insert(0, str((ROOT / "released_benchmark_dependencies_v1").resolve()))
    from rouge_score.rouge_scorer import RougeScorer
    scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = [scorer.score(r["reference"], r["author_text"]) for r in outputs]
    rouge = {k: sum(row[k].fmeasure for row in rows) * 100 / len(rows)
             for k in ("rouge1", "rouge2", "rougeL")}
    saved = read(AUDIT / "released32_descriptive_rouge.json")
    if saved["source_sha256"] != sha256(original / "outputs.json") or any(
            abs(rouge[k] - saved["rouge"][k]) > 1e-10 for k in rouge):
        raise ValueError("descriptive ROUGE replay differs")
    converged = ROOT / "20261023_converged_native_results_v1/campaign_results.json"
    quality = read(converged)
    if quality["grading"]["secondary"]["passed"] != 41 or quality["original_nine_pass_score"]["passed"] != 0:
        raise ValueError("review measured quality conclusions")
    local = [Path(__file__), REPORT, Path("experiments/structure_net/ncm_native_concepts.py"),
             Path("experiments/structure_net/ncm_native_query_concepts.py"),
             Path("experiments/structure_net/ncm_native_word_neighborhood.py"),
             Path("experiments/structure_net/train_ncm_native_query_neighborhood.py"),
             Path("experiments/structure_net/ncm_paired_word_control.py"),
             Path("experiments/structure_net/ncm_isolated_quality_train.py"),
             Path("docs/07 - Status Reports/2026-10-23_ncm-quality-teacher-plan.md")]
    inputs = [original / "outputs.json", original / "result.json", converged]
    inputs += [p for p in AUDIT.rglob("*") if p.is_file()]
    result = {
        "kind": "pinned_source_comparison_with_existing_output_score_replay",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "native_hybrid_materially_differs_from_released_NCM",
        "upstream_commits": {k: v["sha"] for k, v in manifest["repos"].items()},
        "downloaded_source_hashes_verified": len(source_records),
        "released_XSum_control": {"cases": 32, "author_local_exact": True, "rouge": rouge,
                                  "full_benchmark_replicated": False, "application_accuracy": None},
        "native_hybrid": {"nine_pass_passed": 0, "converged_passed": 41, "development_cases": 64},
        "NCM_training_entry_point_present_in_pinned_release": False,
        "causal_training_references": ["hao-ai-lab/Consistency_LLM", "hao-ai-lab/JacobiForcing"],
        "local_sources": {str(p): sha256(p) for p in local},
        "input_hashes": {str(p): sha256(p) for p in dict.fromkeys(inputs)},
        "new_training": False, "new_generation": False, "new_judgments": False,
        "NCM_training_replicated": False, "speed_tested": False, "goal_95_established": False,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"classification": result["classification"], "rouge32": rouge,
                      "verified_upstream_sources": len(source_records)}, allow_nan=False))


if __name__ == "__main__":
    main()
