"""Prospective secondary country-identity endpoint after development wording failures."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .jspace_attention_study import DEFAULT_ROOT, verify, checked
from .jspace_attention_tasks import ANSWERS
from .jspace_causal_components import cpu_only, paired_test
from .jspace_relation_study import ROOT, file_hash, write_json, digest
from .jspace_relation_tasks import COUNTRIES
from .jspace_tasks import normalize_answer, wilson_interval


PROTOCOL = ROOT/"docs/07 - Status Reports/jspace/2026-11-13_jspace-attention-identity-amendment.md"
IDENTITIES = {normalize_answer(word):country for country in COUNTRIES
              for word in [country]+[answers[country] for answers in ANSWERS.values()]}


def freeze(root):
    cpu_only()
    plan = verify(root)
    checked(root, "discovery")
    if (root/"evaluation").exists():
        raise ValueError("Secondary endpoint must be declared before evaluation")
    value = {"scope": "secondary_attribute_agnostic_country_identity_not_exact_task_success",
             "created_at": datetime.now(timezone.utc).isoformat(), "scientific_fingerprint": plan["scientific_fingerprint"],
             "identities": IDENTITIES, "producer_sha256": {str(Path(__file__).resolve()):file_hash(__file__), str(PROTOCOL):file_hash(PROTOCOL)},
             "development_sha256": file_hash(root/"discovery/records.json"), "evaluation_started": False}
    value["fingerprint"] = digest(value)
    write_json(root/"identity_audit_plan.json", value)
    return value


def analyze(root):
    cpu_only()
    verify(root)
    checked(root, "evaluation")
    plan = json.loads((root/"identity_audit_plan.json").read_text())
    if (plan["identities"] != IDENTITIES or plan["fingerprint"] != digest({k:v for k,v in plan.items() if k != "fingerprint"})
            or any(file_hash(p) != h for p,h in plan["producer_sha256"].items())):
        raise ValueError("Secondary endpoint changed")
    records = json.loads((root/"evaluation/records.json").read_text())
    tasks = json.loads((root/"tasks.json").read_text())["pairs"]["evaluation"]
    selected = json.loads((root/"selection.json").read_text())["method"]
    identity = lambda row:IDENTITIES.get(normalize_answer(row["answer"] or ""))
    source = [identity(r["source"]) == p["source"]["intermediate"] for r,p in zip(records,tasks)]
    donor = [identity(r["donor"]) == p["donor"]["intermediate"] for r,p in zip(records,tasks)]
    eligible = [s and d for s,d in zip(source,donor)]
    arms, paired = {}, {}
    for arm in records[0]["edits"]:
        successes = [identity(r["edits"][arm]) == p["source"]["swap_to"] for r,p in zip(records,tasks)]
        paired[arm] = [{"case_id":r["source"]["case_id"], "target_correct":ok} for r,ok,e in zip(records,successes,eligible) if e]
        arms[arm] = {"eligible": wilson_interval(sum(s and e for s,e in zip(successes,eligible)), sum(eligible)),
                     "all_worlds": wilson_interval(sum(successes), len(successes)),
                     "by_attribute": {a:{"total":sum(e and p["source"]["attribute"] == a for e,p in zip(eligible,tasks)),
                         "target_correct":sum(ok and e and p["source"]["attribute"] == a for ok,e,p in zip(successes,eligible,tasks))} for a in ANSWERS}}
    value = {"scope": plan["scope"], "confirmed": False, "status": "completed", "plan_sha256": file_hash(root/"identity_audit_plan.json"),
             "evaluation_manifest_sha256": file_hash(root/"evaluation_manifest.json"), "source": wilson_interval(sum(source),len(source)),
             "donor": wilson_interval(sum(donor),len(donor)), "eligible": sum(eligible), "arms": arms,
             "J_minus_random": paired_test(paired[selected],paired["random_"+selected]) if any(eligible) else None,
             "interpretation": "Collapses capitals, demonyms and country names to country identity; cannot replace the frozen exact task score or establish attribute execution."}
    write_json(root/"identity_audit.json", value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "analyze"))
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](DEFAULT_ROOT), allow_nan=False))
