"""Archive the complete wider training-label population, including failures."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_wide_source_labels as labels
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261023_wide_vetted_labels_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-wide-vetted-training-labels.md")


def replay_generation(root, expected_cases):
    rows = labels.q.read(root / "labels.json")
    if [r["id"] for r in rows] != [c["id"] for c in expected_cases]:
        raise ValueError("fixed training source order changed")
    by_id = {r["id"]: r for r in rows}
    for batch in labels.q.read(root / "batches.json"):
        parsed = []
        for path in batch["attempts"]:
            raw = labels.q.read(path); request = raw["request"]
            if (raw["request_sha256"] != labels.q.digest(request)
                    or request["model"] != labels.q.JUDGES["primary"]
                    or request["messages"][0]["content"] != labels.SYSTEM
                    or request["temperature"] != 0 or request["max_tokens"] != 2048):
                raise ValueError("teacher request identity changed")
            passages = json.loads(request["messages"][1]["content"])["passages"]
            cases = [by_id[p["id"]] for p in passages]
            for passage, case in zip(passages, cases):
                if passage["source_lines"] != [{"line_id": f"L{i}", "text": line} for i, line in enumerate(case["source"].split("\n"))]:
                    raise ValueError("teacher source lines changed")
            value = None
            try:
                if raw["http_status"] == 200 and raw["response"].get("model") == labels.q.JUDGES["primary"]:
                    value = labels.teacher.parse(raw["response"]["choices"][0]["message"]["content"], cases)
            except (ValueError, KeyError, TypeError, IndexError):
                pass
            if bool(raw["valid"]) != (value is not None):
                raise ValueError("teacher schema replay differs")
            parsed.append(value)
        if not 1 <= len(parsed) <= 3 or any(v is not None for v in parsed[:-1]) or parsed[-1] != batch["values"]:
            raise ValueError("first schema-valid teacher candidate replaced")
        for identifier, value in batch["values"].items():
            row = by_id[identifier]
            claims = [{k: c[k] for k in ("claim", "evidence_ids")} for c in row["claims"]]
            if value != {"id": identifier, "summary": row["text"], "claims": claims}:
                raise ValueError("saved teacher label differs from raw response")
    return rows


def replay_grade(root, rows, stage):
    cache = labels.q.read(root / "judgment_cache.json")
    for key, entry in cache.items():
        labels.first.audit_row(entry["packet"], entry["role"], entry["row"])
        if key != labels.first.packet_key(entry["packet"], entry["role"]):
            raise ValueError("first judgment identity differs")
    endpoint = labels.q.read(root / "result.json")["endpoint"]
    scores = labels.q.read(root / ("secondary_" + stage) / "case_scores.json")
    packets = labels.q.read(root / ("secondary_" + stage) / "packets.json")
    counts = {"valid": 0, "passed": 0}
    if len(scores) != len(rows) or len(packets) != len(rows):
        raise ValueError("complete teacher denominator required")
    for row, score, packet in zip(rows, scores, packets):
        expected = labels.q.packet(row["id"], row["source"], [{"candidate_id": "summary", "text": row["text"]}])
        key = labels.first.packet_key(expected, "secondary"); entry = cache[key]
        valid = entry["row"]["status"] == "valid"
        passed = labels.q.passes(entry["row"]["evaluations"][0], row["text"], row["stored_tokens"]) if valid else None
        if (packet != expected or score["id"] != row["id"] or score["text"] != row["text"]
                or score["packet_key"] != key or score["first_origin"] != entry["origin"]
                or score["tokens"] != row["stored_tokens"] or score["valid"] != valid or score["passed"] != passed):
            raise ValueError("whole-label first decision replay differs")
        counts["valid"] += valid; counts["passed"] += passed is True
    if any(counts[k] != endpoint[k] for k in counts):
        raise ValueError("teacher endpoint differs")
    return scores


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh finalization required")
    cases = labels.setup(False)  # Only112train sources; no diagnostic/sealed text.
    import tiktoken
    from transformers import AutoTokenizer
    encoding = tiktoken.get_encoding("cl100k_base")
    tokenizer = AutoTokenizer.from_pretrained(labels.MODEL_PATH, local_files_only=True)
    roots, outputs, results = [], {}, {}
    for pilot in (True, False):
        selected = [next(c for c in cases if c["bin"] == i) for i in range(4)] if pilot else cases
        for stage in ("draft", "critique", "revision", "audit"):
            root = labels.root(pilot, stage); labels.terminal(root); roots.append(root)
            results[("pilot_" if pilot else "main_") + stage] = labels.q.read(root / "result.json")
            if stage in ("draft", "revision"):
                rows = replay_generation(root, selected)
                for row in rows:
                    native = tokenizer.encode(row["text"], add_special_tokens=False)
                    if (row["stored_tokens"] != len(encoding.encode(row["text"], disallowed_special=()))
                            or row["native_tokens_including_eos"] != len(native) + 1
                            or row["native_roundtrip_exact"] != (tokenizer.decode(native, skip_special_tokens=False) == row["text"])
                            or row["native_special_tokens_in_text"] != any(v in tokenizer.all_special_ids for v in native)
                            or any(c["source_quotes"] != [row["source"].split("\n")[int(k[1:])] for k in c["evidence_ids"]] for c in row["claims"])):
                        raise ValueError("capacity/evidence metadata differs")
                outputs[pilot, stage] = rows
                if not pilot:
                    prior = {r["id"]: r for r in outputs[True, stage]}
                    if any(r != prior[r["id"]] for r in rows if r["id"] in prior):
                        raise ValueError("pilot training text regenerated")
            else:
                rows = outputs[pilot, "draft" if stage == "critique" else "revision"]
                scores = replay_grade(root, rows, stage)
                if stage == "audit":
                    accepted = [r for r, score in zip(rows, scores) if score["passed"] is True and labels.label_fits(r)]
                    if labels.q.read(root / "accepted_labels.json") != accepted:
                        raise ValueError("accepted-label export changed")
    audit = results["main_audit"]
    accepted = labels.q.read(labels.root(False, "audit") / "accepted_labels.json")
    if len(accepted) != audit["accepted_labels"] or audit["endpoint"]["cases"] != 112:
        raise ValueError("training-label denominator/count differs")
    paths = [Path(__file__), Path(labels.__file__), labels.PLAN, Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in roots:
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    artifacts = {str(p): file_hash(p) for p in dict.fromkeys(paths)}
    DEST.mkdir()
    aggregate = {"schema": "nal.ncm-wide-vetted-labels.v1", "classification": "training_labels_vetted_student_quality_unmeasured",
        "stages": results, "population": 112, "accepted_labels": len(accepted),
        "accepted_by_bin": audit["accepted_by_bin"], "all_first_decisions_replayed": True,
        "teacher_population95_claim": False, "student_training_admitted": False,
        "task_accuracy_measured": False, "NCM_success_established": False, "speed_tested": False,
        "final_test_opened": False, "confirmed": False, "artifacts": artifacts}
    campaign = DEST / "campaign_results.json"; write_json(campaign, aggregate)
    REPORT.write_text(f'''# Wider training labels vetted; student quality remains unmeasured

Date: 2026-10-23. Classification `training_labels_vetted_student_quality_unmeasured`.
Aggregate `{campaign}`, SHA256 `{file_hash(campaign)}`.

The fixed112public training sources span512–2048cl100k tokens, with28sources
per band. Every source received one Sonnet draft, one Terra critique, one
Sonnet revision and one Terra final audit. Four pilot texts and their first
judgments were retained exactly; there were no semantic rerolls or selection
between draft and revision. The4/4final pilot gate passed before main production.

First drafts passed93/112whole-label audits. Final revisions passed109/112
({109/112:.2%}), all112judgments valid. The final population retains all three
failures; one final candidate also fails the format/capacity checks. The109
accepted training labels fit both128stored cl100k tokens and128native tokens
including EOS, have exact native round trips, nonempty evidence-linked claims
and no embedded special tokens. Accepted counts by increasing source-length
band are27,28,26,28. The final Wilson95%interval is92.42–99.08%; no teacher
population95%claim is established by a97.32%point estimate.

Raw first schema-valid teacher candidates and first-valid judgments were
replayed offline against saved labels and scores. Token lengths, native
round trips, evidence quotes and exact pilot reuse were checked. Maximum
three attempts allowed transport/schema recovery only. All eight workers
reached observed terminal success. Generation and judging stayed sequential.

These are accepted teacher training labels, **not student accuracy**. Terra
both critiques and audits; this is not independent human gold. Diagnostic16,
new sealed64and original sealed256sources were excluded. No student training
is admitted by this result, and no model, speed or compression claim follows.
The separate2048-update native NCM study scored0/64development summaries.
''')
    print(json.dumps({"status": "finalized", "accepted_labels": len(accepted), "population": 112}), flush=True)


def register():
    register_records(DEST / "campaign_results.json", REPORT, DEST, {"ncm-wide-vetted-labels-v1": {
        "name": "Public longer-fragment teacher labels vetted individually",
        "description": "One fixed Sonnetdraft Terracritique Sonnetrevision Terrafinalaudit on112disjoint public training sources512-2048tokens. Final109/112whole-labelpasses,112valid,109capacity/evidenceaccepted,bin27/28/26/28. Traininglabels only; no studentaccuracy or trainingadmission.",
        "confirmation_status": "training_labels_vetted_student_quality_unmeasured",
        "task_accuracy_measured": False, "student_training_admitted": False, "NCM_success_established": False}})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
