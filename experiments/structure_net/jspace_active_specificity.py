"""Prospective stricter interpretation, layered over immutable base results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .jspace_active_study import DEFAULT_ROOT, verify, cpu_only
from .jspace_relation_study import ROOT, digest, file_hash, write_json


PROTOCOL = ROOT/"docs/07 - Status Reports/jspace/2026-11-13_jspace-active-cav-heads-amendment-01.md"
TEST = ROOT/"tests/structure_net/test_jspace_active_specificity.py"


def refine(result):
    classification = result["classification"]
    components = result["components"]
    if "24" not in components:
        return {"classification": classification, "answer_only_dense_control_passed": None,
                "reason": "causal endpoint not reached"}
    success = components["24"]["answer_donor"]["dense"]["target_success"]
    passed = success >= .8
    if not passed and classification != "country_CAV_causality_not_established":
        classification = "CAV_transfer_supported_country_specificity_unresolved_dense_answer_control"
    return {"classification": classification, "answer_only_dense_control_passed": passed,
            "answer_only_dense_control_success": success,
            "reason": "country specificity requires a manipulable answer-only control at the same site"}


def freeze(root):
    cpu_only()
    base = verify(root)
    if (root/"evaluation").exists() or (root/"analysis.json").exists():
        raise ValueError("Amendment must precede evaluation capture")
    paths = [Path(__file__), PROTOCOL, TEST]
    value = {"base_study_id": base["study_id"], "amendment": 1, "base_scientific_fingerprint": base["scientific_fingerprint"],
             "frozen_at": datetime.now(timezone.utc).isoformat(), "evaluation_started": False,
             "minimum_answer_only_dense_success": .8,
             "source_sha256": {str(p): file_hash(p) for p in paths},
             "scope": "stricter interpretation only; no capture, fitting, selection or head-quality changes"}
    value["scientific_fingerprint"] = digest(value)
    for path in paths:
        target = root/"amendment_01_source"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(path, target)
    write_json(root/"amendment_01_plan.json", value)
    return value


def apply(root):
    cpu_only()
    base = verify(root)
    plan = json.loads((root/"amendment_01_plan.json").read_text())
    if plan["base_scientific_fingerprint"] != base["scientific_fingerprint"] or digest({k: v for k, v in plan.items() if k != "scientific_fingerprint"}) != plan["scientific_fingerprint"]:
        raise ValueError("Amendment fingerprint mismatch")
    for path, expected in plan["source_sha256"].items():
        if file_hash(path) != expected or file_hash(root/"amendment_01_source"/Path(path).relative_to(ROOT)) != expected:
            raise ValueError("Amendment source changed")
    result = json.loads((root/"result.json").read_text())
    aggregate = json.loads((root/"campaign_results.json").read_text())
    if aggregate["result_sha256"] != file_hash(root/"result.json") or result["status"] != "completed":
        raise ValueError("Base study is incomplete or changed")
    interpretation = refine(result)
    amended = {**result, "base_classification": result["classification"],
               "classification": interpretation["classification"], "specificity_control": interpretation,
               "base_scientific_fingerprint": base["scientific_fingerprint"],
               "scientific_fingerprint": digest({"base": base["scientific_fingerprint"], "amendment": plan["scientific_fingerprint"]}),
               "base_result_sha256": file_hash(root/"result.json"),
               "amendment_plan_sha256": file_hash(root/"amendment_01_plan.json"), "prospective_amendment": 1}
    write_json(root/"amended_result.json", amended)
    write_json(root/"campaign_results_amended.json", {**amended, "result_path": "amended_result.json",
                                                     "result_sha256": file_hash(root/"amended_result.json")})
    return {"classification": amended["classification"], "base_classification": result["classification"],
            "specificity_control": interpretation, "scientific_fingerprint": amended["scientific_fingerprint"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "apply"))
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](args.output), allow_nan=False))
