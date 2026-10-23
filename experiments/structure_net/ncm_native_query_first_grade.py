"""Grade fixed NCM outputs while preserving all exact prior baseline judgments."""
import json
from pathlib import Path

from experiments.structure_net import train_ncm_native_query_neighborhood as run
from experiments.structure_net import ncm_budget128_profile as profile
from experiments.structure_net import ncm_source_readout_grade as first
from experiments.structure_net.ncm_native_label_quality import dependencies
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, PREP, verify

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-neighborhood-first-judgment-amendment.md")
GRADE = ROOT / "20261023_native_query_neighborhood_grade_v2"
BASE = ROOT / "20261023_budget128_baselines_grading_v1"
CAMPAIGN = ROOT / "20261023_budget128_baselines_results_v1/campaign_results.json"
REGISTRY = Path("data/hypothesis_registry/ncm-budget128-fixed-baselines-v1.json")


def load_prior_cache():
    dependencies(); profile.install()
    registered = run.q.read(REGISTRY)
    if (registered["status"] != "registered_and_read_back"
            or file_hash(CAMPAIGN) != registered["record"]["campaign_sha256"]):
        raise ValueError("registered baseline aggregate identity changed")
    for path, expected in run.q.read(CAMPAIGN)["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("prior baseline evidence changed: " + path)
    cache = {}
    for key, entry in run.q.read(BASE / "judgment_cache.json").items():
        if key != first.packet_key(entry["packet"], entry["role"]):
            raise ValueError("prior packet/profile/role/model identity differs")
        first.audit_row(entry["packet"], entry["role"], entry["row"])
        first.remember(cache, key, entry)
    return cache


def producing(root, stage, inputs, config):
    extra = [Path(__file__), PLAN, BASE / "judgment_cache.json", CAMPAIGN, REGISTRY]
    p = run.producing(root, stage, inputs + extra, {**config,
        "all_prior_exact_baseline_judgments_preserved": True, "negative_judgments_preserved": True})
    for path in (Path(__file__), PLAN):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("student_first_" + path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = run.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def main():
    cache = load_prior_cache()
    prior_count = len(cache)
    if run.q.read(run.OUTPUTS / "worker_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("generation worker must be observed terminal")
    verify(run.q.read(run.OUTPUTS / "producer.json"))
    generated = run.q.read(run.OUTPUTS / "result.json")
    if (generated["status"] != "generated_pending_task_grade" or generated["cases"] != 64
            or file_hash(run.OUTPUTS / "outputs.json") != generated["output_sha256"]):
        raise ValueError("complete unchanged fixed outputs required")
    cases = run.q.read(run.OUTPUTS / "outputs.json")
    development = run.q.read(PREP / "development_sources.json")
    identity = ("id", "source_id", "group_id", "source_sha256", "source")
    if (len(cases) != 64 or len(development) != 64
            or any(tuple(a[k] for k in identity) != tuple(b[k] for k in identity) for a, b in zip(cases, development))
            or any(r["source_truncated"] or r["AR_fallback"] or r["fixed_forward_passes"] != 9 for r in cases)):
        raise ValueError("original64source order and fixed nine-pass endpoint required")
    inputs = [run.OUTPUTS / n for n in ("outputs.json", "producer.json", "result.json", "worker_terminal_receipt.json")]
    for role in ("secondary", "primary"):
        root = ROOT / ("20261023_budget128_judge_" + role + "_v1")
        verify(run.q.read(root / "producer.json"))
        if (run.q.read(root / "result.json")["status"] != "passed"
                or run.q.read(root / "worker_terminal_receipt.json")["exit_code"] != 0):
            raise ValueError("both qualified128-token judges required")
        inputs += [root / n for n in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    p = producing(GRADE, "source_disjoint_NCM_summary_first_judgments", inputs,
        {"cases": 64, "secondary_then_primary_joint_minimum": 61, "rubric": profile.RUBRIC,
         "prior_baseline_cache_entries": prior_count, "original_v1_grading_deferred": True})
    write_json(GRADE / "judgment_cache.json", cache)
    first.DEST, first.freeze = GRADE, producing
    secondary = first.grade_condition(cases, "summary", "secondary", cache, inputs)
    primary = None
    if secondary["valid"] == 64 and secondary["passed"] >= 61:
        primary = first.grade_condition(cases, "summary", "primary", cache, inputs)
    passed = primary is not None and primary["valid"] == 64 and primary["passed"] >= 61
    verify(p)
    result = {"status": "completed", "secondary": secondary, "primary": primary,
        "classification": "short_development_gate_passed_final_required" if passed else "NCM_summary_development_failed",
        "development_gate": passed, "prior_baseline_cache_entries": prior_count,
        "all_prior_exact_baseline_judgments_preserved": True, "original_v1_grading_deferred": True,
        "NCM_model": True, "NCM_success_established": False, "final_test_opened": False,
        "speed_tested": False, "producer_fingerprint": p["fingerprint"]}
    write_json(GRADE / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
