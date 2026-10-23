"""Candidate-level recovery; preserve every original NCM verdict unchanged."""
import argparse
import json
from pathlib import Path
import shutil

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_recipe_teacher import verify_public
from experiments.structure_net.ncm_recipe_test import TEST, ARMS, register as register_test

ROOT = TEST.parent / "20261022_test_recovery_v1"
NOTE = Path("docs/07 - Status Reports/2026-10-22_ncm-recipe-candidate-recovery.md")
HYPOTHESIS = "ncm-recipe-update-test-candidate-recovery-v1"


def first_valid_candidate(packet, candidate_id, attempts):
    one = {**packet, "candidates": [c for c in packet["candidates"] if c["candidate_id"] == candidate_id]}
    for path in attempts:
        raw = q.read(path)
        if raw.get("http_status") != 200 or raw.get("response", {}).get("model") != q.JUDGES["primary"]:
            continue
        content = raw["response"]["choices"][0]["message"]["content"].strip()
        try:
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            obj = json.loads(content)
            evaluation = next(g for g in obj["evaluations"] if g["candidate_id"] == candidate_id)
            parsed = q.parse(json.dumps({"evaluations": [evaluation]}), one)[0]
            return parsed, str(path)
        except (ValueError, KeyError, StopIteration, TypeError, IndexError):
            continue
    return None, None


def recover(args):
    import tiktoken
    if ROOT.exists() or args.report.exists():
        raise ValueError("fresh recovery root/report required")
    old = q.read(TEST / "campaign_results.json")
    for path, digest in old["artifacts"].items():
        if file_hash(path) != digest:
            raise ValueError("original test evidence changed")
    verify_public(q.read(TEST / "selection.json")["test"])
    ROOT.mkdir()
    (ROOT / "attempts").mkdir()
    (ROOT / "producer_source").mkdir()
    source = [Path(__file__), NOTE, Path("experiments/structure_net/ncm_memory_accuracy.py"),
              Path("experiments/structure_net/ncm_memory_accuracy_v2.py"),
              Path("experiments/structure_net/ncm_recipe_test.py"),
              Path("experiments/structure_net/ncm_recipe_teacher.py")]
    for i, p in enumerate(source):
        shutil.copyfile(p, ROOT / "producer_source" / f"{i:03d}_{p.name}")
    producer = {"schema": "nal.ncm-candidate-recovery.v1", "hypothesis_id": HYPOTHESIS,
        "source_hashes": {str(p): file_hash(p) for p in source},
        "original_campaign_sha256": file_hash(TEST / "campaign_results.json"),
        "judge": q.JUDGES["primary"], "rubric": q.RUBRIC, "new_ncm_requests_allowed": 0,
        "rule": "first existing valid candidate; only an invalid parent may get a new candidate-only packet"}
    producer["fingerprint"] = q.digest(producer)
    write_json(ROOT / "producer.json", producer)
    packets = {p["case_id"]: p for p in q.read(TEST / "accuracy/packets.json")}
    candidates = q.read(TEST / "accuracy/candidate_map.json")
    enc = tiktoken.get_encoding("cl100k_base")
    recovered = []
    for row in q.read(TEST / "accuracy/judgments.json"):
        packet = packets[row["case_id"]]
        for candidate in packet["candidates"]:
            cid = candidate["candidate_id"]
            identity = candidates[cid]
            grade, origin = first_valid_candidate(packet, cid, row["attempts"])
            repair = None
            if grade is None and identity["arm"] == "parent":
                single = {**packet, "case_id": packet["case_id"]+"-parent-only-repair", "candidates": [candidate]}
                write_json(ROOT / "parent_repair_packet.json", single)
                repair = q.request(single, "primary", ROOT)
                if repair["status"] == "valid":
                    grade, origin = first_valid_candidate(single, cid, repair["attempts"])
            recovered.append({**identity, "valid": grade is not None, "grade": grade,
                "origin_attempt": origin, "new_parent_repair": repair,
                "passed": q.passes(grade, candidate["text"], len(enc.encode(candidate["text"], disallowed_special=()))) if grade else None})
    summary = {}
    for arm in ARMS:
        rs = [r for r in recovered if r["arm"] == arm]
        valid = [r for r in rs if r["valid"]]
        k = sum(r["passed"] for r in valid)
        summary[arm] = {"requested": 64, "valid": len(valid), "passes": k,
            "task_accuracy": k/64 if len(valid) == 64 else None,
            "wilson_95": q.wilson(k, len(valid)), "missing": 64-len(valid)}
    for p, digest in producer["source_hashes"].items():
        if file_hash(p) != digest:
            raise ValueError("recovery source changed")
    write_json(ROOT / "case_scores.json", recovered)
    result = {"schema": "nal.ncm-candidate-recovery-campaign.v1", "hypothesis_id": HYPOTHESIS,
        "status": "completed", "classification": "candidate_recovery_updated_below_95" if summary["updated"]["valid"] == 64 and summary["updated"]["passes"] < 61 else "candidate_recovery_unqualified",
        "models": summary, "original_packet_comparison_remains_invalid": True,
        "validation_granularity_changed_posthoc": True, "new_ncm_judgments": 0,
        "speed_tested": False, "human_rated": False, "underpowered": True,
        "artifacts": {str(p): file_hash(p) for p in ROOT.rglob("*.json") if "producer_source" not in p.parts}}
    result["artifacts"][str(TEST / "campaign_results.json")] = file_hash(TEST / "campaign_results.json")
    write_json(ROOT / "campaign_results.json", result)
    table = "\n".join(f"| {arm} | {v['passes']}/{v['valid']} | {v['missing']} |" for arm,v in summary.items())
    args.report.write_text(f'''# NCM recipe update: recovered candidate-level test judgments

**Updated NCM: {summary['updated']['passes']}/{summary['updated']['valid']} passing complete summaries.**
The 95% point target is not met. Date: 2026-10-22. EXPLORATORY, one seed.
ID: `{HYPOTHESIS}`. Classification: `{result['classification']}`.
Aggregate: `{ROOT / 'campaign_results.json'}`.
SHA256: `{file_hash(ROOT / 'campaign_results.json')}`.
[Recovery protocol](../07%20-%20Status%20Reports/2026-10-22_ncm-recipe-candidate-recovery.md).

| Model | Passing / valid candidate judgments | Missing |
| --- | ---: | ---: |
{table}

The original [whole-packet comparison](2026-10-22_ncm-recipe-update-test.md)
remains invalid: one parent evaluation contradicted its own claim labels in all
three replies, although both NCM failures in that packet were valid. This new
diagnostic parses each candidate separately with the unchanged parser, original
source lines and candidate IDs/text. It uses the first valid evaluation in
chronological order, never the most favorable. No NCM output, judgment or
Boolean decision was regenerated or changed. Only the parent candidate with no
valid existing decision was eligible for a separate schema-repair packet.
Every raw reply and lineage is retained. Validation granularity changed after
outcomes were known; this cannot rescue the original invalid comparison or
the failed training-reference gate.

The 64 source-disjoint test cases are the previously inspected original cohort,
disjoint from the 32 cleaned training pairs. Judge:
`{q.JUDGES['primary']}`, unchanged source-grounded faithfulness, salience,
readability, nonempty and <=128-token rubric. Public LongMemEval file SHA and
source spans were verified before repair. No private passages or original
private targets were sent. Judge calibration previously accepted24/24 good and
rejected24/24 bad agent-authored controls; this is not human-gold accuracy.
Wilson intervals in the aggregate exclude judge error and model-seed variation.

The [training update](2026-10-22_ncm-recipe-update.md) implements real nested
decoder cutoffs, same-level conditioning corruption, equal level loss and paired
codec hashes with a frozen parent body and frozen encoder/codebooks. It fitted
the small training set, but source-disjoint summary competence is not established.
Neither model has a useful test pass rate. No speedup or full application-range
claim follows from this experiment. Original XSum sanity control: ROUGE-L27.89,
codec reconstruction99.82% on128 cases; it is not a full paper replication.

The training comparison remains `recipe_update_training_positive_control_failed`:
updated29/32, legacy30/32, teacher references30/32 under a declared31/32 gate.
The separately recorded fresh-body inference check matched the cached output
on one training example; it proves integration, not generalization.

Reproduce candidate recovery from the saved raw replies with the moba Python,
memory dependencies and tiktoken cache:

```text
python -m experiments.structure_net.recover_ncm_recipe_test recover --report NEW_REPORT
python -m experiments.structure_net.recover_ncm_recipe_test register --report NEW_REPORT
```

Artifact roots are pinned in source; reproduction needs a new output root.
''')
    print(json.dumps({"classification": result["classification"], "models": summary}), flush=True)


def register(args):
    # Reuse the exact JSON/semantic read-back implementation with explicit new
    # identity and root; the original module file and evidence remain unchanged.
    import experiments.structure_net.ncm_recipe_test as original
    original.TEST, original.HYPOTHESIS = ROOT, HYPOTHESIS
    original.register(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["recover", "register"])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args)
