"""Guard against attaching fresh accuracy to different timed tensors."""
from copy import deepcopy

import pytest

from experiments.neural_architecture_lab.finalize_jspace_compact_performance import local_timings, native_timings, summarize


def evidence(recover=False):
    plan = {"fingerprint": "f" * 64, "recover": recover, "source_arm": "r1536_h64_gb",
            "candidate": "combined", "performance_root": "/performance"}
    records = {arm: [{"identity": str(i), "answer": 0, "prediction": int(i >= 400)} for i in range(512)]
               for arm in ("native", "control", "fast")}
    accuracy = {"count": 512, "correct": 400, "overall": 400 / 512}
    result = {"status": "completed", "confirmed": False, "fingerprint": plan["fingerprint"],
              "fresh_evaluation_count": 512, "recover": recover, "source_arm": plan["source_arm"],
              "candidate": plan["candidate"], "results": {},
              "timing_provenance": {"root": "/fresh" if recover else "/performance",
                  "stage": "runtime" if recover else "full_r1536_h64_gb_combined", "tensor_identity_verified": not recover}}
    deployment, reload = {"checkpoints": {}}, {}
    for arm in records:
        result["results"][arm] = {"accuracy": deepcopy(accuracy), "tensor_bytes": 6000,
            "parameter_count": 2500, "trained_in_this_study": recover and arm != "native"}
        if arm != "native":
            deployment["checkpoints"][arm] = {"tensor_bytes": 6000, "parameter_count": 2500,
                "trained": recover, "timing_tensors_match_parent_candidate": None if recover else True}
            reload[arm] = {"records_exact": True, "accuracy": deepcopy(accuracy)}
    runtime = {arm: {"timings": {"hf": {"tokens_per_second": rate / 2}, "cuda_graph": {"tokens_per_second": rate}},
                     "isolated_block_prefill": [{"seconds_per_call": .1 if arm in ("fast", "candidate") else .2}]}
               for arm, rate in (("native", 42.), ("native_repeat", 41.8),
                                 ("control" if recover else "reference", 41.),
                                 ("fast" if recover else "candidate", 40.), ("reference_repeat", 41.1))}
    bench = {"pairs": [{"identity": "dev0", "mode": mode, "summary": {
        execution: {"reference": {"wall_seconds": .4 if execution == "eager" else .2},
                    "candidate": {"wall_seconds": .3 if execution == "eager" else .1}}
        for execution in ("eager", "graph")}} for mode in ("prefill", "decode")]}
    return plan, result, deployment, reload, records, runtime, bench


def test_local_win_is_separate_from_full_model_regression_and_never_confirmation():
    value = summarize(*evidence())
    assert value["classification"] == "local_speedup_with_measured_quality"
    assert value["confirmed"] is False
    assert value["local_screen_timings"]["prefill/graph"]["speed_ratio_to_compact"] == 2.
    assert value["local_screen_timings"]["prefill/eager"]["speed_ratio_to_compact"] == pytest.approx(4 / 3)
    assert value["whole_model_ratios"]["cuda_graph"]["to_native"] < 1.
    assert "not generation quality" in value["scope"]


@pytest.mark.parametrize("mutation", ["confirmed", "reload", "identity", "export_identity", "accuracy", "training", "timing_stage", "pair_order"])
def test_invalid_evidence_is_not_indexed(mutation):
    args = evidence()
    plan, result, deployment, reload, records, runtime, bench = args
    if mutation == "confirmed":
        result["confirmed"] = True
    elif mutation == "reload":
        reload["fast"]["records_exact"] = False
    elif mutation == "identity":
        result["timing_provenance"]["tensor_identity_verified"] = False
    elif mutation == "export_identity":
        deployment["checkpoints"]["fast"]["timing_tensors_match_parent_candidate"] = False
    elif mutation == "accuracy":
        result["results"]["fast"]["accuracy"]["overall"] = .99
    elif mutation == "training":
        result["results"]["fast"]["trained_in_this_study"] = True
    elif mutation == "timing_stage":
        result["timing_provenance"]["stage"] = "different_candidate"
    else:
        records["fast"][0], records["fast"][1] = records["fast"][1], records["fast"][0]
    with pytest.raises(ValueError):
        summarize(*args)


def test_recovered_export_cannot_inherit_screening_microbenchmarks():
    args = evidence(recover=True)
    args[-1]["pairs"][0]["summary"]["graph"]["candidate"]["wall_seconds"] = .00001
    value = summarize(*args)
    assert value["local_screen_applies_to_exported_tensors"] is False
    assert value["recovered_export_prefill_seconds"]["fast"] == .1
    args[1]["timing_provenance"]["stage"] = "full_r1536_h64_gb_combined"
    with pytest.raises(ValueError, match="timing stage"):
        summarize(*args)


def test_duplicate_prompts_cannot_artificially_reweight_local_speed():
    bench = evidence()[-1]
    bench["pairs"].append(deepcopy(bench["pairs"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        local_timings(bench)


def native_evidence():
    plan = {"parent_root": "/parent", "fast_options": {"geometry_mode": "bf16"}, "source_arm": "r1536_h64_gb",
            "candidate": "combined", "recover": False}
    native_plan = {"parent_root": "/parent", "candidate_options": plan["fast_options"],
                   "case_identities": [str(i) for i in range(8)], "config": {"paired_repeats": 1}}
    evidence = {"candidate": "r1536_h64_gb/combined", "reference": "original native block35", "quality_selection": False,
                "final_norm_in_timing": False, "pairs": [], "summary": {}}
    for mode in ("prefill", "decode"):
        evidence["summary"][mode] = {metric: {"speedup": 2.} for metric in ("wall_seconds", "cuda_event_seconds")}
        for identity in native_plan["case_identities"]:
            timings = [{"repeat": 0, "order": order, "variant": variant,
                        "wall_seconds": .2 if variant == "reference" else .1,
                        "cuda_event_seconds": .2 if variant == "reference" else .1}
                       for order in (["reference", "candidate"], ["candidate", "reference"]) for variant in order]
            evidence["pairs"].append({"identity": identity, "mode": mode, "tokens": 100, "timings": timings,
                "native_graph": {"graph_error": {"equal": True}}, "fast_graph": {"graph_error": {"equal": True}},
                "summary": {metric: {"native": .2, "fast": .1} for metric in ("wall_seconds", "cuda_event_seconds")}})
    return plan, native_plan, evidence


def test_native_speed_is_audited_from_balanced_observations_not_borrowed_from_compact_control():
    plan, native_plan, evidence = native_evidence()
    value = native_timings(plan, native_plan, evidence)
    assert value["measurements"]["prefill"]["wall_seconds"]["speed_ratio_to_native"] == 2.
    assert value["measurements"]["decode"]["prompts_faster_than_native"] == 8
    assert "no separately custom-fused native baseline" in value["scope"]
    plan["recover"] = True
    assert native_timings(plan, native_plan, evidence)["applies_to_exported_tensors"] is False


@pytest.mark.parametrize("mutation", ["parent", "summary", "order", "graph"])
def test_native_comparator_rejects_mismatched_or_unbalanced_evidence(mutation):
    plan, native_plan, evidence = native_evidence()
    if mutation == "parent":
        native_plan["parent_root"] = "/different_parent"
    elif mutation == "summary":
        evidence["summary"]["prefill"]["wall_seconds"]["speedup"] = 3.
    elif mutation == "order":
        evidence["pairs"][0]["timings"].pop()
    else:
        evidence["pairs"][0]["fast_graph"]["graph_error"]["equal"] = False
    with pytest.raises(ValueError):
        native_timings(plan, native_plan, evidence)
