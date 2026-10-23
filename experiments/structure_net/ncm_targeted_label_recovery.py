"""One reviewed correction of rejected TRAIN drafts; no student evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.structure_net import ncm_additional_whole_labels as labels
from experiments.structure_net import ncm_expanded_word_data as replay_data
from experiments.neural_architecture_lab.ncm_result_registry import register_records

q = labels.prior.q
base = labels.prior.data
ROOT = base.ROOT
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-targeted-label-recovery-plan.md")
HYPOTHESIS = "ncm-targeted-training-label-revisions-v1"
PREP = ROOT / "20261023_targeted_label_sources_v1"
WORKFLOW = ROOT / "20261023_targeted_label_workflow_v1"
ARCHIVE = ROOT / "20261023_targeted_label_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-targeted-training-label-revisions.md")
BASELINE = ROOT / "20261023_reviewed_draft_data_v1"
AVAILABILITY = ROOT / "20261023_judge_gateway_completion_probe_v2/result.json"
ABORT = ROOT / "20261023_additional_revision_transport_abort_v1"
_original_root = labels.root
_original_producing = labels.producing
_configured = False


def root(pilot, stage):
    if stage in ("draft", "critique"):
        return _original_root(False, stage)
    return ROOT / ("20261023_targeted_label_"+("pilot_" if pilot else "main_")+stage+"_v1")


def select_targets(cases, failures):
    by_id = {r["id"]: r for r in cases}
    if len(by_id) != len(cases) or len({r["label"]["id"] for r in failures}) != len(failures):
        raise ValueError("unique original cases and rejected labels required")
    selected = []
    for row in failures:
        label, score = row["label"], row["score"]
        case = by_id.get(label["id"])
        if (case is None or score["id"] != label["id"] or score["text"] != label["text"] or score["valid"] is not True
                or type(score["passed"]) is not bool or type(row["format_capacity_fit"]) is not bool
                or (score["passed"] is True and row["format_capacity_fit"] is True)
                or any(case[k] != label[k] for k in ("id", "source_id", "group_id", "source_sha256", "source", "spans"))):
            raise ValueError("only unchanged, validly reviewed rejected TRAIN drafts qualify")
        selected.append(case)
    return selected


def pilot_cases(cases):
    return [next(r for r in cases if r["bin"] == b) for b in range(4)]


def pilot_gate(result):
    return (result["endpoint"]["cases"] == 4 and result["endpoint"]["valid"] == 4
        and result["endpoint"]["passed"] == 4 and result["accepted_labels"] == 4 and result["pilot_gate"] is True)


def limited_pool(*args, **kwargs):
    if args:
        args = (2,)+args[1:]
        kwargs.pop("max_workers", None)
    else:
        kwargs["max_workers"] = 2
    return ThreadPoolExecutor(*args, **kwargs)


def producing(destination, stage, inputs, config):
    files = [Path(__file__), PLAN]
    extra = [BASELINE / n for n in ("producer.json", "result.json", "accepted_labels.json", "teacher_failures.json", "judgment_cache.json")]
    extra += [AVAILABILITY, ABORT / "decision_before_signal.json", ABORT / "observation.json"]
    extra += [labels.PREP / n for n in ("train_sources.json", "result.json")]
    extra += [_original_root(False, "draft") / "labels.json",
        _original_root(False, "critique") / "secondary_critique/case_scores.json"]
    if destination not in (PREP, WORKFLOW):
        extra += [PREP / n for n in ("producer.json", "result.json", "sources.json", "public_proof.json", "worker_terminal_receipt.json")]
    p = _original_producing(destination, stage, list(inputs)+files+extra, {**config,
        "targeted_source_population": 74, "original_source_population": 512,
        "unchanged_admitted_drafts": 438, "new_transport_attempt_limit": 3,
        "old_interrupted_response_outcomes_unknown": True, "student_task_accuracy": None})
    p["hypothesis_id"] = HYPOTHESIS
    p["config"].update(hypothesis_id=HYPOTHESIS, workers=2)
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = base.file_hash(path)
        (destination / "producer_source" / f"targeted_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    base.write_json(destination / "producer.json", p)
    return p


def prepare():
    cases = labels.setup(False)
    base.terminal(BASELINE)
    result = q.read(BASELINE / "result.json")
    if result["cases"] != 690 or result["additional_rejected"] != 74:
        raise ValueError("exact existing reviewed-draft training population required")
    for name, expected in result["files"].items():
        if base.file_hash(BASELINE / name) != expected:
            raise ValueError("current training data changed")
    if not q.read(AVAILABILITY)["both_tiny_completions_succeeded"]:
        raise ValueError("new documented judge availability required")
    failures = q.read(BASELINE / "teacher_failures.json")
    selected = select_targets(cases, failures)
    if len(selected) != 74 or [sum(r["bin"] == b for r in selected) for b in range(4)] != [16, 31, 7, 20]:
        raise ValueError("fixed rejected population or band counts changed")
    admitted = {r["id"] for r in q.read(BASELINE / "accepted_labels.json")}
    if (len(admitted) != 438 or admitted & {r["id"] for r in selected}
            or any(r["pool_split"] != "train" for r in selected)):
        raise ValueError("accepted labels and evaluation sources cannot enter targeted revision")
    original = {r["id"]: r for r in q.read(_original_root(False, "draft") / "labels.json")}
    scores = {r["id"]: r for r in q.read(_original_root(False, "critique") / "secondary_critique/case_scores.json")}
    if any(r["label"] != original[r["label"]["id"]] or r["score"] != scores[r["label"]["id"]]
            or r["format_capacity_fit"] != labels.label_fits(r["label"]) for r in failures):
        raise ValueError("original rejection decisions changed")
    proof = labels.wide.verify_public(selected)
    p = producing(PREP, "select_original_rejected_TRAIN_labels", [], {"cases": 74, "pilot_cases": 4})
    base.write_json(PREP / "sources.json", selected)
    base.write_json(PREP / "public_proof.json", proof)
    base.verify(p)
    base.write_json(PREP / "result.json", {"status": "targeted_TRAIN_sources_prepared", "cases": 74,
        "sources_sha256": base.file_hash(PREP / "sources.json"), "pilot_ids": [r["id"] for r in pilot_cases(selected)],
        "student_task_accuracy": None, "final_test_opened": False, "producer_fingerprint": p["fingerprint"]})


def setup(pilot):
    labels.setup(False)  # Original public-source and 128-token qualification checks.
    base.terminal(PREP)
    if base.file_hash(PREP / "sources.json") != q.read(PREP / "result.json")["sources_sha256"]:
        raise ValueError("fixed74source selection changed")
    cases = q.read(PREP / "sources.json")
    if not pilot:
        base.terminal(root(True, "audit"))
        if not pilot_gate(q.read(root(True, "audit") / "result.json")):
            raise ValueError("strict4/4targeted pilot gate failed")
    return pilot_cases(cases) if pilot else cases


def configure():
    global _configured
    if _configured:
        return
    labels.configure()
    labels.wide.root, labels.wide.setup, labels.wide.producing = root, setup, producing
    labels.wide.ThreadPoolExecutor = limited_pool
    labels.wide.first.ThreadPoolExecutor = limited_pool
    _configured = True


def exhausted(destination):
    groups = {}
    for path in destination.glob("**/attempts/*.json"):
        raw = q.read(path)
        key = (str(path.parent), path.stem.rsplit("_", 1)[0])
        groups.setdefault(key, []).append((path, raw))
    return [str(path) for group in groups.values() if len(group) == 3
        and all(raw.get("valid") is False for _, raw in group) for path, _ in group]


def finalize():
    cases = setup(False)
    import tiktoken
    from transformers import AutoTokenizer
    storage = tiktoken.get_encoding("cl100k_base")
    native = AutoTokenizer.from_pretrained(labels.wide.MODEL_PATH, local_files_only=True)
    roots, outputs, decisions = [], {}, {}
    for pilot in (True, False):
        selected = pilot_cases(cases) if pilot else cases
        revision, audit = root(pilot, "revision"), root(pilot, "audit")
        for path in (revision, audit): base.terminal(path); roots.append(path)
        rows = replay_data.replay.replay_generation(revision, selected)
        for row in rows:
            ids = native.encode(row["text"], add_special_tokens=False)
            if (row["stored_tokens"] != len(storage.encode(row["text"], disallowed_special=()))
                    or row["native_tokens_including_eos"] != len(ids)+1
                    or row["native_roundtrip_exact"] != (native.decode(ids, skip_special_tokens=False) == row["text"])
                    or row["native_special_tokens_in_text"] != any(v in native.all_special_ids for v in ids)
                    or any(c["source_quotes"] != [row["source"].split("\n")[int(k[1:])] for k in c["evidence_ids"]] for c in row["claims"])):
                raise ValueError("raw target capacity or source-evidence metadata changed")
        scores = replay_data.replay.replay_grade(audit, rows, "audit")
        expected = [r for r, s in zip(rows, scores) if s["passed"] is True and labels.label_fits(r)]
        if q.read(audit / "accepted_labels.json") != expected:
            raise ValueError("accepted revision export differs from first judgments")
        outputs[pilot], decisions[pilot] = rows, scores
    retained = {r["id"]: r for r in outputs[True]}
    if any(r != retained[r["id"]] for r in outputs[False] if r["id"] in retained):
        raise ValueError("pilot revisions were regenerated")
    endpoint = q.read(root(False, "audit") / "result.json")
    if endpoint["endpoint"]["cases"] != 74 or endpoint["endpoint"]["valid"] != 74:
        raise ValueError("complete74case valid first judgments required")
    cache = q.read(root(False, "audit") / "judgment_cache.json")
    previous = q.read(BASELINE / "judgment_cache.json")
    if len(previous) != 1444 or any(cache.get(k) != v for k, v in previous.items()):
        raise ValueError("all1444earlier judgments including negatives must remain exact")
    inputs = [p for directory in roots for p in directory.rglob("*")
        if p.is_file() and "producer_source" not in p.parts]
    inputs += [WORKFLOW / "producer.json"]
    p = producing(ARCHIVE, "replay_targeted_TRAIN_label_revision_outcome", inputs, {"cases": 74})
    aggregate = {"schema": "ncm.targeted-training-label-revisions.v1", "hypothesis_id": HYPOTHESIS,
        "classification": "targeted_training_labels_revised_student_quality_unmeasured",
        "cases": 74, "accepted_revisions": endpoint["accepted_labels"], "accepted_by_bin": endpoint["accepted_by_bin"],
        "endpoint": endpoint["endpoint"], "baseline438_unchanged": True, "prior1444_judgments_exact": True,
        "all_first_candidates_and_judgments_replayed": True, "student_task_accuracy": None,
        "NCM_success_established": False, "final_test_opened": False, "speed_tested": False,
        "confirmed": False, "artifacts": {str(path): base.file_hash(path) for path in inputs},
        "producer_fingerprint": p["fingerprint"]}
    base.write_json(ARCHIVE / "campaign_results.json", aggregate)
    REPORT.write_text(f'''# Revisions of rejected NCM training drafts

Date: 2026-10-23. Classification `{aggregate['classification']}`.
Aggregate `{ARCHIVE / 'campaign_results.json'}`, SHA256 `{base.file_hash(ARCHIVE / 'campaign_results.json')}`.

All 74 previously rejected TRAIN drafts received one first-observed revision
and a first semantic judgment; {endpoint['accepted_labels']} revisions meet the
unchanged semantic and capacity criteria. Accepted counts by length band are
{endpoint['accepted_by_bin']}. The four strict pilot results were retained.
Every first candidate, failure, and all 1,444 preceding judgments remain exact.
The earlier interrupted all-512 revision study remains failed.

The 438 admitted original drafts and running 690-example word model were not
changed. These are prospective TRAIN labels for a possible later iteration,
not student accuracy. No sealed source was opened and no speed was measured.
''')
    base.verify(p)
    base.write_json(ARCHIVE / "result.json", {"status": "finalized_training_labels_only", "task_accuracy": None})


def register():
    aggregate = q.read(ARCHIVE / "campaign_results.json")
    register_records(ARCHIVE / "campaign_results.json", REPORT, ARCHIVE, {HYPOTHESIS: {
        "name": "One revision of rejected NCM TRAIN labels", "description": "Fixed74rejectedTRAINdrafts, oneSonnetrevisionandTerrareview, strict4/4pilot, unchanged128tokencriteria, fullrawreplayandall1444priorjudgmentspreserved. Futuretraininglabels only, not studentaccuracy.",
        "confirmation_status": aggregate["classification"], "task_accuracy_measured": False,
        "NCM_success_established": False, "student_training_admitted": False}})


def workflow():
    if WORKFLOW.exists() or any(path.exists() for path in [PREP, ARCHIVE, *[root(p, s) for p in (True, False) for s in ("revision", "audit")]]):
        raise ValueError("fresh targeted study roots required")
    p = producing(WORKFLOW, "supervise_targeted_label_revisions_without_GPU", [], {"cases": 74, "pilot_cases": 4})
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "task_accuracy": None}
    base.write_json(WORKFLOW / "progress.json", journal)
    schedule = [("prepare", None, PREP), ("revision", True, root(True, "revision")), ("audit", True, root(True, "audit")),
        ("revision", False, root(False, "revision")), ("audit", False, root(False, "audit")),
        ("finalize", None, ARCHIVE), ("register", None, ARCHIVE)]
    try:
        for stage, pilot, destination in schedule:
            if destination.exists() != (stage == "register"): raise ValueError("existing stage cannot be repeated")
            base.verify(p)
            args = [sys.executable, "-u", "-m", "experiments.structure_net.ncm_targeted_label_recovery", stage]
            if pilot is False: args.append("--main")
            child = subprocess.Popen(args)
            journal.update(active_stage=("pilot_" if pilot else "main_" if pilot is False else "")+stage, child_pid=child.pid)
            base.write_json(WORKFLOW / "progress.json", journal)
            stop_evidence = []
            while child.poll() is None:
                if stage in ("revision", "audit"):
                    stop_evidence = exhausted(destination)
                    if stop_evidence:
                        base.write_json(destination / "admission_impossible_before_stop.json", {"reason": "a required request exhausted all3attempts without a valid response", "attempts": {path: base.file_hash(path) for path in stop_evidence}})
                        child.terminate(); break
                time.sleep(2)
            code = child.wait()
            receipt = {"status": "observed_worker_terminal", "observer": "subprocess.wait", "observer_pid": os.getpid(), "child_pid": child.pid, "exit_code": code, "command": args, "stage": journal["active_stage"]}
            if destination.exists(): base.write_json(destination / ("registry_terminal_receipt.json" if stage == "register" else "worker_terminal_receipt.json"), receipt)
            journal["stages"].append(receipt); base.write_json(WORKFLOW / "progress.json", journal)
            if code != 0: raise RuntimeError("targeted label stage failed: "+journal["active_stage"])
            base.terminal(destination)
            if stage in ("revision", "audit") and q.read(destination / "result.json")["status"] not in ("generated", "completed"):
                raise ValueError("all required generation/judgment schemas must be valid")
            if stage == "audit" and pilot is True and not pilot_gate(q.read(destination / "result.json")):
                raise ValueError("strict4/4pilot failed; main not admitted")
        base.verify(p)
        journal.update(status="completed_targeted_TRAIN_label_revisions", active_stage=None, final_test_opened=False)
        base.write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        base.write_json(WORKFLOW / "failure.json", journal)
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["prepare", "revision", "audit", "finalize", "register", "workflow"])
    parser.add_argument("--main", action="store_true")
    args = parser.parse_args(); configure()
    if args.stage == "prepare": prepare()
    elif args.stage == "revision": labels.wide.generate(not args.main, True)
    elif args.stage == "audit": labels.wide.grade(not args.main, True)
    elif args.stage == "finalize": finalize()
    elif args.stage == "register": register()
    else: workflow()


if __name__ == "__main__":
    from experiments.structure_net import ncm_targeted_label_recovery as canonical
    canonical.main()
