"""Record exact valid-label recovery without rescuing failed teacher quality."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net.ncm_expanded_label_recovery import PARENT, RECOVERY
from experiments.neural_architecture_lab.ncm_result_registry import register_records

AUDIT = ROOT / "20261022_expanded_recovery_audit_v1"
PILOT = ROOT / "20261022_expanded_recovery_audit_pilot_v1"
DEST = ROOT / "20261023_expanded_label_quality_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-expanded-label-quality.md")


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh completed result required")
    parent, recovery, audit = [q.read(root / "result.json") for root in (PARENT, RECOVERY, AUDIT)]
    if (parent["status"] != "failed_data_gate" or recovery["status"] != "available_labels_recovered_pending_quality_audit"
            or audit["cases"] != 128 or audit["status"] != "completed" or audit["valid"] != 128):
        raise ValueError("complete preserved parent, recovery and128 valid judgments required")
    terminal = q.read(AUDIT / "worker_terminal_receipt.json")
    if terminal["status"] != "observed_worker_terminal" or terminal["exit_code"] != 0:
        raise ValueError("authoritative audit termination required")
    for root in (PARENT, RECOVERY, AUDIT, PILOT):
        verify(q.read(root / "producer.json"))
    if file_hash(RECOVERY / "train.json") != recovery["train_sha256"] or file_hash(PARENT / "train.json") != recovery["train_sha256"]:
        raise ValueError("recovered labels differ from their failed parent")
    ordered = sorted(q.read(RECOVERY / "train.json"), key=lambda r: text_hash("expanded-audit-v1:"+r["id"]))
    scores, packets = q.read(AUDIT / "case_scores.json"), q.read(AUDIT / "packets.json")
    if [r["id"] for r in scores] != [r["id"] for r in ordered[:128]]:
        raise ValueError("audit selection/order changed")
    sys.path.append(str(Path("data/experiments/ncm_memory/dependencies").resolve()))
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    passed, flags = 0, Counter()
    for label, score, packet in zip(ordered, scores, packets):
        if label["id"] != packet["case_id"] or label["target"] != score["text"]:
            raise ValueError("audit source/label pairing changed")
        attempts = [q.read(path) for path in score["attempts"]]
        if not attempts[-1]["valid"] or any(a["valid"] for a in attempts[:-1]):
            raise ValueError("first valid semantic outcome was not preserved")
        parsed = q.parse(attempts[-1]["response"]["choices"][0]["message"]["content"], packet)
        if parsed != score["evaluations"] or not score["valid"]:
            raise ValueError("raw judgment replay differs")
        n = len(enc.encode(label["target"], disallowed_special=()))
        value = q.passes(parsed[0], label["target"], n)
        if value != score["passed"] or n != score["tokens"]:
            raise ValueError("whole-summary pass calculation differs")
        passed += value
        flags.update(k for k in ("faithful", "salient", "readable") if not parsed[0][k])
    if passed != audit["passed"]:
        raise ValueError("audit numerator changed")
    gate = passed >= 122
    excluded = Counter(r["reason"] for r in q.read(PARENT / "excluded.json"))
    paths = [Path(__file__), Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in (PARENT, RECOVERY, AUDIT, PILOT):
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    DEST.mkdir()
    aggregate = {"schema": "nal.ncm-expanded-label-quality.v1", "classification": "label_quality_gate_passed" if gate else "label_quality_gate_failed",
        "parent": parent, "recovery": recovery, "audit": audit, "failed_flags": dict(flags),
        "exclusion_reasons": dict(excluded), "first_valid_replay_verified": True,
        "original_parent_reclassified": False, "student_training_admitted": gate,
        "student_accuracy": None, "student_trained": False, "speed_tested": False,
        "final_test_opened": False, "confirmed": False, "underpowered": True,
        "artifacts": {str(p): file_hash(p) for p in paths}}
    campaign = DEST / "campaign_results.json"
    write_json(campaign, aggregate)
    REPORT.write_text(f'''# Broader teacher labels fail the independent quality gate

Date: 2026-10-23. `ncm-expanded-valid-label-recovery-v1` is complete, classified
**{aggregate['classification']}**. Recovery preserves7,923 existing valid labels
but does not qualify them for a larger student-training run. This is teacher
label quality, not student accuracy. The NCM attachment retains its earlier
0/64 development result under Terra; no new model score or speed is established.

The fixed independent sample passes **{passed}/128={100*passed/128:.4f}%** against
the unchanged complete-summary faithful/salient/readable rubric and128-token
cap. The predeclared point gate is122/128. Wilson95% interval is
{100*audit['wilson_95'][0]:.4f}–{100*audit['wilson_95'][1]:.4f}%. All128 judgments
are valid; none needed a retry. First-valid raw responses, parser replay, case
order, source/label identity, tokens and every pass decision are verified.
Failure flags: `{dict(flags)}`. The15 failed labels are readable and salient
but introduce unsupported assertions or remove material restrictions. Examples
include dropping the qualifying-purchases restriction on reward points and
identifying a speaker as an AI when the supplied passage does not establish it.
These failures are not exact-reference disagreements or decoder corruption.

The original8,192-source teacher campaign finished2,048 batches with2,045 valid,
three invalid after the existing attempt cap and36 retried batches. It keeps
its **failed_data_gate** classification. Of269 excluded labels,257 exceed the
prototype's48-GPT2-token canvas without truncation and12 have no valid response.
No missing or long label was quietly replaced. The separate recovery replayed
all raw requests/responses, preserved each first valid answer, reconstructed
accepted/excluded files exactly and copied the7,923 labels byte-for-byte. No
additional teacher request or semantic quality filtering occurred.

Audit cases are the first128 in the prospective hash order, one source/session/
group per example. A separate four-case systems pilot returned four valid
judgments and three semantic passes; four valid responses alone satisfied its
declared lifecycle requirement. The audit uses the qualified Terra model
`codex_sdk/gpt-5.6-terra` with four workers after the teacher process terminated.
The128 sample is a development data-quality gate, not independently collected
human gold, a confidence guarantee, a student result or a final task test.

Because the gate fails, do not feed these labels to a scaled student run under
this protocol. A new declared data-quality intervention must qualify its final
labels before training. Source-copying feasibility may use weak summaries only
as diagnostic semantic queries: judge its actual resulting summaries directly,
and do not treat the references as gold or the selectors as trained NCM success.
The failed codec and failed free-text generator are separate defects; imperfect
labels do not by themselves explain or excuse the student's0/64 result.

Aggregate: `{campaign}`; SHA256 `{file_hash(campaign)}`. Parent raw responses,
failed batches, exclusions, exact recovered file, all first-valid judgments,
source/protocol seals and the authoritative worker-terminal receipt remain
append-only. Registry confirmed=false, student_accuracy_measured=false. The
final task test remains sealed and original-body weights stay unchanged.
''')
    print(json.dumps({"status": aggregate["classification"], "passed": passed, "cases": 128, "report": str(REPORT)}), flush=True)


def register():
    register_records(DEST / "campaign_results.json", REPORT, DEST, {
        "ncm-expanded-valid-label-recovery-v1": {"name": "Recovered broader NCM teacher labels and independent quality audit",
            "description": "7923labels recovered exactly from the preserved failed8192source teacher campaign, with no new teacher calls. Independent qualified Terra audit113/128=88.28125percent, allvalid,0retries, misses122/128gate. All15failfaithfulness. Broader student training not admitted; not student accuracy.",
            "confirmation_status": "label_quality_gate_failed", "task_accuracy_measured": False,
            "student_accuracy_measured": False, "student_training_admitted": False},
    })
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
