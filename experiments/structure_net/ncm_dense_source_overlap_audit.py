"""Append an independent source-contamination audit to dense development results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_dense_semantic_development as study
from experiments.structure_net import ncm_source_split_integrity as integrity
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records

ROOT = study.ROOT
HYPOTHESIS = "ncm-dense-semantic-source-overlap-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-dense-source-overlap-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-dense-semantic-source-overlap.md")
DEST = ROOT / "20261023_dense_semantic_source_overlap_v1"
WORKFLOW = ROOT / "20261023_dense_semantic_source_overlap_workflow_v1"
CASES = ROOT / "20261023_reviewed_draft_data_v1/cases.json"
CASES_SHA = "ac3c56f7d03cf9160de9e719ef042b44a0bf12dececd778949b46e7288aac983"
MODULE = "experiments.structure_net.ncm_dense_source_overlap_audit"


def inputs():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("source overlap audit is explicitly CPU only")
    study.training.clean.dependencies()
    q = study.reference.q
    path = study.RESULTS / "campaign_results.json"
    campaign = q.read(path)
    receipt = q.read(study.RESULTS / (study.HYPOTHESIS + "_registry_readback.json"))
    terminal = q.read(study.WORKFLOW / "result.json")
    if (receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != file_hash(path)
            or receipt["exact"]["campaign_sha256"] != file_hash(path)
            or terminal["status"] != "dense_semantic_development_completed"
            or campaign["classification"] != "dense_semantic_development_quality_failed"
            or campaign["selected_checkpoint_sha256"] != study.CHECKPOINT
            or len(terminal["stages"]) != 7 or any(r["exit_code"] for r in terminal["stages"])):
        raise ValueError("original fixed semantic evaluation must finish unchanged and be registered")
    if file_hash(CASES) != CASES_SHA or file_hash(study.reference.SOURCES / "cases.json") != study.SOURCE_SHA:
        raise ValueError("unchanged complete TRAIN and development sources required")
    rows = q.read(CASES)["train"]
    development = q.read(study.reference.SOURCES / "cases.json")
    if len(rows) != 690 or [len(development[k]) for k in study.reference.COHORTS] != [64, 64]:
        raise ValueError("all690 TRAIN and both full64-case development cohorts required")
    outputs = q.read(study.OUTPUTS / "outputs.json")
    scores = {s: q.read(study.GRADE / f"{s}_secondary_case_scores.json") for s in development}
    paths = [CASES, study.reference.SOURCES / "cases.json", study.OUTPUTS / "outputs.json", path,
        study.RESULTS / (study.HYPOTHESIS + "_registry_readback.json"), study.WORKFLOW / "result.json"]
    paths += [study.GRADE / f"{s}_secondary_case_scores.json" for s in development]
    for p in [study.OUTPUTS / "outputs.json", *[study.GRADE / f"{s}_secondary_case_scores.json" for s in development]]:
        if campaign["artifacts"].get(str(p)) != file_hash(p):
            raise ValueError("registered raw prediction/judgment artifact differs")
    return rows, development, outputs, scores, paths


def analyze(training, development, outputs, scores):
    all_dev = [r for values in development.values() for r in values]
    if any(text_hash(r["source"]) != r["source_sha256"] for r in [*training, *all_dev]):
        raise ValueError("actual source bytes must match recorded identity")
    by_id = {r["id"]: r for r in outputs}
    if [r["id"] for r in all_dev] != [r["id"] for r in outputs]:
        raise ValueError("original complete prediction order must remain unchanged")
    for row in all_dev:
        if any(by_id[row["id"]][k] != row[k] for k in (*study.reference.IDENTITY, "source")):
            raise ValueError("original prediction/source identity differs")
    conflicts = integrity.cross_overlaps(training, all_dev)
    overlapping = {r["heldout_id"] for r in conflicts}
    targets = {}
    for row in training:
        targets.setdefault(row["target_text"], []).append(row["id"])
    summary = {}
    for split, values in development.items():
        cases = scores[split]
        if [r["id"] for r in cases] != [r["id"] for r in values]:
            raise ValueError("all original ordered case judgments required")
        if any(r["text"] != by_id[r["id"]]["text"] for r in cases):
            raise ValueError("original judged prediction differs")
        passed = [r["id"] for r in cases if r["passed"] is True]
        exact = {r["id"]: targets[by_id[r["id"]]["text"]] for r in values if by_id[r["id"]]["text"] in targets}
        summary[split] = {"cases": len(values), "valid": sum(r["valid"] for r in cases),
            "raw_semantic_passes": len(passed), "cases_with_source_overlap": sum(r["id"] in overlapping for r in values),
            "passing_cases_with_source_overlap": [k for k in passed if k in overlapping],
            "exact_TRAIN_target_outputs": exact, "passing_exact_TRAIN_target_outputs": [k for k in passed if k in exact]}
    within = []
    signatures = [integrity.signature(r) for r in all_dev]
    for i, row in enumerate(all_dev):
        for j in range(i):
            value = integrity.overlap(all_dev[j], row, signatures[j], signatures[i])
            if value is not None:
                within.append({"first_id": all_dev[j]["id"], "second_id": row["id"], **value})
    return {"TRAIN_development_conflicts": conflicts, "within_development_conflicts": within, "cohorts": summary}


def audit():
    rows, development, outputs, scores, paths = inputs()
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh append-only source-overlap evidence and amendment required")
    files = [Path(__file__), Path(integrity.__file__), PLAN]
    p = study.reference.data.freeze(DEST, "CPU_source_split_overlap_audit_not_model_quality", [*paths, *files],
        {"hypothesis_id": HYPOTHESIS, "TRAIN_cases": 690, "development_cases": 128,
         "normalization": "NFKC_casefold_unicode_words", "near_overlap": "at_least32_shared_word5grams_and80percent_shorter_containment",
         "source_family_rules": "observed_sharegpt_hex_answer_variants;ultrachat_numeric_ids_preserved",
         "new_judge_calls": False, "case_exclusions": False, "model_training": False,
         "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (DEST / "producer_source" / f"source_overlap_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = study.reference.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(DEST / "producer.json", p)
    details = analyze(rows, development, outputs, scores)
    write_json(DEST / "overlap_details.json", details)
    artifacts = {str(path): file_hash(path) for path in [*paths, DEST / "overlap_details.json", DEST / "producer.json", *files]}
    classification = "development_split_contaminated" if details["TRAIN_development_conflicts"] else "no_overlap_detected_under_fixed_checks"
    result = {"hypothesis_id": HYPOTHESIS, "classification": classification,
        "TRAIN_development_conflict_pairs": len(details["TRAIN_development_conflicts"]),
        "within_development_conflict_pairs": len(details["within_development_conflicts"]),
        "cohorts": details["cohorts"], "original_campaign_classification_unchanged": "dense_semantic_development_quality_failed",
        "clean_generalization_accuracy_established": False, "original_case_scores_changed": False,
        "model_training": False, "new_judge_calls": False, "final_test_opened": False, "speed_tested": False,
        "artifacts": artifacts, "producer_fingerprint": p["fingerprint"]}
    study.reference.data.verify(p)
    path = DEST / "campaign_results.json"
    write_json(path, result)
    table = "\n".join(f"| {s} | {v['raw_semantic_passes']}/{v['cases']} | {v['cases_with_source_overlap']}/{v['cases']} | {len(v['passing_exact_TRAIN_target_outputs'])} |"
        for s, v in result["cohorts"].items())
    REPORT.write_text(f"""# Source overlap invalidates the dense student generalization estimate

Date: 2026-10-23. Data audit `{HYPOTHESIS}`. Classification **{classification}**.
Campaign `{path}`; SHA256 `{file_hash(path)}`.

| Original development cohort | Raw semantic passes | Cases with TRAIN source overlap | Passing outputs identical to TRAIN targets |
| --- | --- | --- | --- |
{table}

The original128-case study remains a recorded quality failure with all original
cases, predictions and first judgments preserved. Its metadata check established
distinct IDs, session IDs, group IDs and exact source hashes. It did not detect
related session-name variants, normalized duplicate passages or substantial text
containment. The recorded source-disjoint flag is therefore insufficient to
establish clean generalization. No cases are removed and no revised accuracy
is manufactured by changing the denominator.

The passing long-source prediction exactly reproduces a32-case TRAIN target.
Its development source `sharegpt_uD7CBng_2` differs from the training passage
`sharegpt_uD7CBng_4` by two characters; the shorter source is contained in the
longer. This is evidence of memorization on an overlapping passage.

This independent audit compares all690 approved TRAIN sources with all128
development sources. It joins observed sharegpt, hexadecimal and answer session
variants while preserving distinct ultrachat numeric IDs. It also tests shared
public-turn hashes, NFKC/case/word-normalized text equality and at least32 shared
word5grams covering80% of the shorter passage. The same checks audit pairs
inside development. These conservative checks do not prove all semantic
duplicates absent, but they catch the observed failure and must precede new
heldout selection. Conflict pairs and all raw cohort counts are retained.

Future student quality needs fresh development sources excluded by complete
source families, existing group IDs, public-turn identity and text overlap,
before outputs or judgments are produced. All690 approved TRAIN labels may
remain training data; they must not be called independent evaluation examples.
Both sealed final source files remain unopened and unhashed by this audit.
Their eligibility also needs metadata-based split checks before final use.
The frozen-body95% complete-summary goal remains unmet; no speed gain follows.
""")
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}), flush=True)


def register():
    study.training.clean.dependencies()
    register_records(DEST / "campaign_results.json", REPORT, DEST, {HYPOTHESIS: {
        "name": "Dense student development source contamination",
        "description": "Independent family, public-turn and normalized text overlap audit; original predictions, judgments and denominators preserved.",
        "classification": study.reference.q.read(DEST / "campaign_results.json")["classification"]}})


def workflow():
    inputs()
    if any(p.exists() for p in (WORKFLOW, DEST, REPORT)):
        raise ValueError("fresh CPU source-audit workflow required")
    WORKFLOW.mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": []}
    try:
        for stage in ("audit", "register"):
            child = subprocess.Popen([sys.executable, "-u", "-m", MODULE, stage], env=os.environ)
            code = child.wait()
            receipt = {"status": "observed_worker_terminal", "stage": stage, "exit_code": code,
                "child_pid": child.pid, "observer_pid": os.getpid(), "observer": "subprocess.wait"}
            journal["stages"].append(receipt)
            write_json(WORKFLOW / (stage + "_receipt.json"), receipt)
            if DEST.exists():
                write_json(DEST / ("worker_terminal_receipt.json" if stage == "audit" else "registry_terminal_receipt.json"), receipt)
            if code:
                raise RuntimeError("source-overlap stage failed: " + stage)
        journal["status"] = "source_overlap_audit_completed"
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("audit", "register", "workflow"))
    args = parser.parse_args()
    {"audit": audit, "register": register, "workflow": workflow}[args.stage]()
