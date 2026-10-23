"""Compare frozen native baseline outputs under the qualified application budget."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_budget128_profile as profile
from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net import ncm_source_readout_grade as first
from experiments.structure_net.ncm_native_label_quality import dependencies
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, PREP, freeze, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-budget128-fixed-baseline-plan.md")
PILOT = ROOT / "20261023_budget128_baselines_pilot_v1"
GRADING = ROOT / "20261023_budget128_baselines_grading_v1"
BASELINES = {
    "native_3b": ROOT / "20261023_native_parent_control_v1",
    "native_1b": ROOT / "20261023_intact_decoder_outputs_v1",
}
OLD_PASSES = {"native_3b": 49, "native_1b": 50}


def same_sources(rows, development):
    keys = ("id", "source_id", "group_id", "source_sha256", "source")
    if len(rows) != 64 or len(development) != 64:
        raise ValueError("all64 original development cases required")
    if any(tuple(a[k] for k in keys) != tuple(b[k] for k in keys) for a, b in zip(rows, development)):
        raise ValueError("fixed source identities/order differ")
    if any(r.get("NCM_model") is not False or r.get("source_truncated") is not False for r in rows):
        raise ValueError("unchanged complete-source native baseline required")


def quality_gate(endpoint):
    return endpoint["cases"] == 64 and endpoint["valid"] == 64 and endpoint["passed"] >= 61


def producing(root, stage, inputs, config):
    sources = [Path(__file__), PLAN, Path(profile.__file__), profile.PLAN,
               Path(first.__file__), first.AMENDMENT]
    p = freeze(root, stage, inputs + sources, {**config, "rubric": profile.RUBRIC,
        "profile": "budget128-v1", "NCM_model": False, "new_model_generation": False,
        "original_old_profile_passes": OLD_PASSES, "workers": 4, "max_attempts": 3,
        "no_semantic_retries": True, "final_test_opened": False, "speed_tested": False})
    for path in sources:
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("baseline128_" + path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def inputs_and_cases():
    dependencies()
    profile.install()
    inputs = [PREP / "development_sources.json"]
    for role in ("secondary", "primary"):
        root = ROOT / ("20261023_budget128_judge_" + role + "_v1")
        verify(q.read(root / "producer.json"))
        if (q.read(root / "result.json")["status"] != "passed"
                or q.read(root / "worker_terminal_receipt.json")["exit_code"] != 0):
            raise ValueError("both qualified128-token judge workers must be terminal")
        inputs += [root / n for n in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    development = q.read(PREP / "development_sources.json")
    verify_public(development)
    cases = {}
    for name, root in BASELINES.items():
        verify(q.read(root / "producer.json"))
        result = q.read(root / "result.json")
        if (result["cases"] != 64 or result["status"] != "generated_pending_task_grading"
                or result["output_sha256"] != file_hash(root / "outputs.json")):
            raise ValueError("complete unchanged frozen baseline outputs required")
        rows = q.read(root / "outputs.json")
        same_sources(rows, development)
        cases[name] = rows
        inputs += [root / n for n in ("producer.json", "result.json", "outputs.json")]
    return inputs, cases


def grade(pilot):
    inputs, cases = inputs_and_cases()
    cache = {}
    if not pilot:
        if (q.read(PILOT / "worker_terminal_receipt.json")["exit_code"] != 0
                or q.read(PILOT / "result.json")["status"] != "passed"):
            raise ValueError("terminal four-output judge lifecycle pilot required")
        verify(q.read(PILOT / "producer.json"))
        inputs += [PILOT / n for n in ("producer.json", "result.json", "judgment_cache.json", "worker_terminal_receipt.json")]
        for key, entry in q.read(PILOT / "judgment_cache.json").items():
            first.audit_row(entry["packet"], entry["role"], entry["row"])
            if key != first.packet_key(entry["packet"], entry["role"]):
                raise ValueError("pilot first-judgment identity changed")
            first.remember(cache, key, entry)
    root = PILOT if pilot else GRADING
    p = producing(root, "systems_judge_lifecycle_only" if pilot else "fixed_baseline_budget128_re_evaluation",
                  inputs, {"pilot": pilot, "cases_per_baseline": 2 if pilot else 64, "joint_minimum": 61})
    write_json(root / "judgment_cache.json", cache)
    first.DEST, first.freeze = root, producing
    results = {}
    for name, rows in cases.items():
        secondary = first.grade_condition(rows[:2] if pilot else rows, name, "secondary", cache, inputs)
        primary = None
        if not pilot and quality_gate(secondary):
            primary = first.grade_condition(rows, name, "primary", cache, inputs)
        results[name] = {"secondary": secondary, "primary": primary,
                         "both_judges_quality_gate": primary is not None and quality_gate(primary)}
        write_json(root / "progress.json", results)
    valid = all(r["secondary"]["valid"] == (2 if pilot else 64) for r in results.values())
    verify(p)
    result = {"status": ("passed" if valid else "failed_lifecycle") if pilot else "completed",
        "systems_only": pilot, "results": results, "old_profile_passes": OLD_PASSES,
        "NCM_model": False, "NCM_success_established": False, "new_model_generation": False,
        "profile": "budget128-v1", "student_quality_measured": False, "speed_tested": False,
        "final_test_opened": False, "producer_fingerprint": p["fingerprint"]}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "grade"])
    grade(parser.parse_args().command == "pilot")
