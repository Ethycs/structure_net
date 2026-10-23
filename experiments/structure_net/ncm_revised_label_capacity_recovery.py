"""Continue unchanged first drafts through one capacity-aware revision."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_revised_label_pipeline as old
from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-revised-label-capacity-amendment.md")
BASE_ROOT, BASE_PRODUCING, BASE_REQUEST = old.root, old.producing, old.request


def root(pilot, stage):
    if stage == "draft" or (pilot and stage == "critique"):
        return BASE_ROOT(pilot, stage)
    return Path(str(BASE_ROOT(pilot, stage)).removesuffix("_v1")+"_v2")


def capacity_critique(draft, critique):
    return {**critique, "draft_capacity": {
        "stored_tokens": draft["stored_tokens"], "native_tokens_including_eos": draft["native_tokens_including_eos"],
        "stored_ceiling": 128, "native_ceiling_including_eos": 128,
        "fits_both": draft["stored_tokens"] <= 128 and draft["native_tokens_including_eos"] <= 128}}


def request(index, cases, destination, drafts, critiques):
    if drafts is None: raise ValueError("first-draft regeneration forbidden")
    supplemented = {key: capacity_critique(value, critiques[key]) for key, value in drafts.items()}
    return BASE_REQUEST(index, cases, destination, drafts, supplemented)


def producing(destination, stage, inputs, config):
    extra = [Path(__file__), PLAN]
    for role in ("primary", "secondary"):
        qual = ROOT/("20261023_budget128_judge_"+role+"_v1")
        extra += [qual/"producer.json", qual/"result.json"]
    p = BASE_PRODUCING(destination, stage, inputs+extra, {**config, "capacity_metadata_for_every_revision": True})
    for path in (Path(__file__), PLAN):
        p["source_hashes"][str(path)] = file_hash(path)
        (destination/"producer_source"/("capacity_"+path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k:v for k,v in p.items() if k != "fingerprint"})
    write_json(destination/"producer.json", p)
    return p


def grade(pilot, audit):
    cases = old.setup(pilot)
    source = root(pilot, "revision" if audit else "draft")
    old.require_terminal(source)
    labels = q.read(source/"labels.json")
    if (len(labels) != len(cases) or [r["id"] for r in labels] != [r["id"] for r in cases]
            or not all(r["teacher_valid"] for r in labels)):
        raise ValueError("all fixed first schema-valid candidates required")
    if q.read(source/"result.json")["status"] not in ("generated", "failed_format_or_capacity"):
        raise ValueError("unrecognized source stage")
    stage = "audit" if audit else "critique"
    destination = root(pilot, stage)
    inputs = [source/n for n in ("labels.json", "producer.json", "result.json")]
    cache = {}
    if audit:
        # v1 final judgments win too when the new revision text is identical.
        previous = [root(pilot, "critique"), BASE_ROOT(True, "audit")]
        for prior in previous:
            old.require_terminal(prior)
            inputs.append(prior/"judgment_cache.json")
            for key, entry in q.read(prior/"judgment_cache.json").items():
                old.first.remember(cache, key, entry)
    p = producing(destination, "capacity_recovery_"+stage, inputs, {"pilot":pilot, "audit":audit})
    write_json(destination/"judgment_cache.json", cache)
    old.first.DEST = destination
    old.first.freeze = producing
    endpoint = old.first.grade_condition(labels, stage, "secondary", cache, inputs)
    capacity = all(r["text"].strip() and r["claims"] and r["stored_tokens"] <= 128
                   and r["native_tokens_including_eos"] <= 128 for r in labels)
    verify(p)
    result = {"status":endpoint["status"], "endpoint":endpoint, "pilot":pilot,
        "all_final_capacity_passed":capacity if audit else None,
        "label_quality_gate": (capacity and endpoint["valid"] == len(cases)
            and endpoint["passed"] >= (10 if pilot else 122)) if audit else None,
        "student_training_admitted":False, "NCM_success_established":False,
        "producer_fingerprint":p["fingerprint"]}
    write_json(destination/"result.json",result)
    print(json.dumps(result),flush=True)


def main(stage, pilot):
    old.root, old.producing, old.request = root, producing, request
    old.SYSTEM += ("\nFor this version prefer40-60words. Each draft critique includes exact "
        "stored and native token counts with ceilings. If either exceeds its ceiling, "
        "shorten the revision while preserving the main point and all material qualifications. "
        "Review every draft once, including those already fitting both ceilings.")
    if stage == "revision": old.generate(pilot, True)
    else: grade(pilot, stage == "audit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage",choices=["critique","revision","audit"])
    parser.add_argument("--main",action="store_true")
    args = parser.parse_args()
    if args.stage == "critique" and not args.main: raise ValueError("reuse the original pilot critique")
    main(args.stage, not args.main)
