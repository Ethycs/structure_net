"""Joint gates must not be rescued by a subset of successful measurements."""
from copy import deepcopy
import json

import pytest

from neural_architecture_lab.jspace_allblocks_hypothesis import (
    FAIL, PASS, STUDY_ID, THRESHOLDS, V2_THRESHOLDS, JSpaceAllBlocksRecord, required_stages,
)


def quality(deltas=(0, 0, 0, 0)):
    families = {}
    for name, delta in zip(("arc", "boolq", "hellaswag", "winogrande"), deltas):
        families[name] = dict(count=128, native_correct=100, compact_correct=100+delta,
            native_accuracy=100/128, compact_accuracy=(100+delta)/128, accuracy_drop=-delta/128,
            wins=max(delta, 0), losses=max(-delta, 0))
    overall = {k: sum(v[k] for v in families.values()) for k in ("count", "native_correct", "compact_correct", "wins", "losses")}
    overall.update(native_accuracy=overall["native_correct"]/512, compact_accuracy=overall["compact_correct"]/512,
                   accuracy_drop=(overall["native_correct"]-overall["compact_correct"])/512)
    return dict(overall=overall, by_family=families)


def evidence(*, speed=1.4, deltas=(0, 0, 0, 0)):
    config = dict(THRESHOLDS, layers=36, group_size=6, epochs=12, microbatch=4, accumulation=4,
        learning_rate=1.2e-4, temperature=2., kl_weight=.7, ce_weight=.3, gradient_clip=1., native_bytes=10000)
    plan = dict(schema="jspace.qwen-parallel.v1", study_id=STUDY_ID, systems_only=False, fingerprint="a"*64, config=config)
    q = quality(deltas)
    gates = dict(overall_quality=q["overall"]["accuracy_drop"] <= .01,
        every_family_quality=all(v["accuracy_drop"] <= 4/128 for v in q["by_family"].values()),
        deployed_size=True, whole_model_speed=speed >= 1.4, packed_reload_exact=True, graph_static_tokens_exact=True)
    passed = all(gates.values())
    result = dict(study_id=STUDY_ID, status="completed", confirmed=False, fingerprint=plan["fingerprint"], evaluation_count=512,
        quality=q, targets_met=passed, classification=PASS if passed else FAIL, packed_reload_exact=True,
        acceptance=dict(passed=passed, gates=gates, thresholds=THRESHOLDS, measured_size_reduction=.38,
            whole_model_speedup=speed, native_baseline_tokens_per_second=42.),
        results={arm: dict(tensor_bytes=6200 if arm == "compact" else 10000,
            runtime=dict(graph_static_exact=True, timings={"cuda_graph": {"tokens_per_second": rate}}))
            for arm, rate in (("native_before", 40.), ("compact", speed*42), ("native_after", 44.))})
    training = {}
    for group in range(6):
        history = [{"epoch": epoch, "development": {"ce": .7 if epoch == 4 else 1-epoch*.01, "count": 64}}
                   for epoch in range(13)]
        training[str(group)] = dict(status="completed", config=config, completed_epochs=12, updates=384,
            selection_metric="packed_development_ce", history=history, selected_epoch=4, selected=deepcopy(history[4]["development"]))
    return dict(plan=plan, result=result, training=training,
        manifests={name: dict(status="completed", fingerprint=plan["fingerprint"]) for name in required_stages()})


def test_all_targets_can_pass_without_confirming_or_counting_groups_as_replicates():
    record = JSpaceAllBlocksRecord.from_evidence(evidence())
    assert record.targets_met and record.classification == PASS and record.confirmed is False
    assert record.completed_groups == 6 and record.independent_confirmations == 0
    assert record.whole_model_speedup == 1.4 and len(record.development_ce_by_group["5"]) == 13
    assert json.loads(json.dumps(record.to_dict())) == record.to_dict()


def test_a_real_completed_negative_result_is_retained():
    record = JSpaceAllBlocksRecord.from_evidence(evidence(speed=1.399))
    assert record.classification == FAIL and not record.targets_met
    assert record.gates["deployed_size"] and not record.gates["whole_model_speed"]


def test_overall_quality_cannot_hide_one_failed_family():
    record = JSpaceAllBlocksRecord.from_evidence(evidence(deltas=(-5, 4, 0, 0)))
    assert record.gates["overall_quality"] and not record.gates["every_family_quality"]
    assert record.classification == FAIL


def amended_evidence(*, deltas=(-20, 0, 0, 0), speed=1.4):
    value = evidence(deltas=deltas, speed=speed)
    value["plan"]["schema"] = "jspace.qwen-parallel.v2"
    value["plan"]["config"].update(V2_THRESHOLDS, rank_gate="actual_sketch_svd_shape_fp32")
    result = value["result"]; acceptance = result["acceptance"]
    acceptance["thresholds"] = dict(V2_THRESHOLDS)
    acceptance["diagnostics"] = {"every_family_quality": acceptance["gates"].pop("every_family_quality"),
        "original_one_percentage_point_quality": acceptance["gates"]["overall_quality"]}
    acceptance["primary_quality_scope"] = "aggregate_accuracy"
    acceptance["gates"]["overall_quality"] = result["quality"]["overall"]["accuracy_drop"] <= .05
    result["targets_met"] = acceptance["passed"] = all(acceptance["gates"].values())
    result["classification"] = PASS if result["targets_met"] else FAIL
    return value


def test_v2_five_point_aggregate_budget_keeps_family_and_one_point_diagnostics():
    record = JSpaceAllBlocksRecord.from_evidence(amended_evidence())
    assert record.targets_met and record.confirmed is False
    assert record.compact_correct == 380 and record.native_correct == 400
    assert record.quality_thresholds["maximum_accuracy_drop"] == .05
    assert record.quality_diagnostics == {"every_family_quality": False, "original_one_percentage_point_quality": False}
    assert "every_family_quality" not in record.gates
    assert not JSpaceAllBlocksRecord.from_evidence(amended_evidence(deltas=(-26, 0, 0, 0))).targets_met
    assert not JSpaceAllBlocksRecord.from_evidence(amended_evidence(speed=1.39)).targets_met


@pytest.mark.parametrize("change", ["hidden_veto", "promoted_diagnostic", "wrong_scope", "wrong_rank"])
def test_v2_rejects_misreported_amendment(change):
    value = amended_evidence()
    if change == "hidden_veto":
        value["result"]["acceptance"]["gates"]["every_family_quality"] = False
    elif change == "promoted_diagnostic":
        value["result"]["acceptance"]["diagnostics"]["every_family_quality"] = True
    elif change == "wrong_scope":
        value["result"]["acceptance"]["primary_quality_scope"] = "per_family"
    else:
        value["plan"]["config"]["rank_gate"] = "different"
    with pytest.raises(ValueError):
        JSpaceAllBlocksRecord.from_evidence(value)


def test_v2_closeout_verifies_original_production_manifest_and_hardlink(tmp_path):
    import os
    from experiments.neural_architecture_lab.finalize_jspace_allblocks import reused_stage_provenance, digest, sha
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(); new.mkdir()
    old_plan = {"source_sha256": {}, "input_sha256": {}}
    old_plan["fingerprint"] = digest(old_plan)
    (old / "plan.json").write_text(json.dumps(old_plan))
    (old / "artifact").write_bytes(b"unchanged")
    os.link(old / "artifact", new / "artifact")
    stage = "groups/group_00/capture"
    path = old / stage / "manifest.json"; path.parent.mkdir(parents=True)
    original = {"status": "completed", "fingerprint": old_plan["fingerprint"], "artifacts": {"artifact": sha(old / "artifact")}}
    path.write_text(json.dumps(original))
    plan = {"schema": "jspace.qwen-parallel.v2", "reuse": {"root": str(old), "source_fingerprint": old_plan["fingerprint"],
        "manifest_sha256": {stage + "/manifest.json": sha(path)}}}
    wrapper = {"provenance_kind": "verified_immutable_reuse", "artifacts": original["artifacts"],
        "reused_from": {"root": str(old), "source_fingerprint": old_plan["fingerprint"], "manifest": stage + "/manifest.json", "manifest_sha256": sha(path)},
        "original_stage_metadata": {k: v for k, v in original.items() if k != "artifacts"}}
    assert reused_stage_provenance(new, plan, stage, wrapper)[str(path)] == sha(path)
    broken = deepcopy(wrapper); broken["original_stage_metadata"]["fingerprint"] = "new producer"
    with pytest.raises(ValueError, match="production metadata"):
        reused_stage_provenance(new, plan, stage, broken)
    (new / "artifact").unlink(); (new / "artifact").write_bytes(b"unchanged")
    with pytest.raises(ValueError, match="hardlink"):
        reused_stage_provenance(new, plan, stage, wrapper)


@pytest.mark.parametrize("mutation", ["confirmation", "fingerprint", "missing_stage", "group", "budget", "selection", "rate", "threshold", "promote"])
def test_invalid_or_partial_evidence_cannot_be_registered(mutation):
    value = evidence()
    if mutation == "confirmation": value["result"]["confirmed"] = True
    elif mutation == "fingerprint": value["manifests"]["evaluation"]["fingerprint"] = "other"
    elif mutation == "missing_stage": del value["manifests"]["groups/group_04/block_07/fit_result"]
    elif mutation == "group": del value["training"]["5"]
    elif mutation == "budget": value["training"]["3"]["updates"] = 383
    elif mutation == "selection": value["training"]["2"]["selected_epoch"] = 12
    elif mutation == "rate": value["result"]["acceptance"]["native_baseline_tokens_per_second"] = 40.
    elif mutation == "threshold": value["plan"]["config"]["minimum_speedup"] = 1.3
    else:
        value = evidence(speed=1.1); value["result"]["targets_met"] = True; value["result"]["classification"] = PASS
    with pytest.raises(ValueError):
        JSpaceAllBlocksRecord.from_evidence(value)


def test_redundancy_keeps_identically_numbered_layer_coordinates_separate():
    from experiments.neural_architecture_lab.finalize_jspace_allblocks import redundancy_summary
    relation = {"selected_anchor_width": 24, "metrics": {"candidates": {"24": {"interfaces": {
        "compressed_dev": {"redundancy": dict(task_uses=8, unique_elements=3, redundant_occurrences=5,
            shared_all_tasks=1, selected_not_eligible_all_tasks=2)}}}}}}
    result = redundancy_summary({"0": relation, "1": deepcopy(relation)})
    totals = result["totals"]["compressed_dev"]
    assert totals["unique_layer_qualified_elements"] == 6
    assert totals["task_uses"] == 16 and totals["redundant_occurrences"] == 10


def test_runtime_rate_must_match_all_wall_samples():
    from experiments.neural_architecture_lab.finalize_jspace_allblocks import verify_runtime
    prompts = [str(i) for i in range(8)]
    timings = {}
    for arm, repeats in (("cuda_graph", 3), ("eager_static", 1), ("hf_eager", 1)):
        samples = [dict(arm=arm, repeat=r, identity=i, tokens=32, seconds=2.) for r in range(repeats) for i in prompts]
        timings[arm] = dict(samples=samples, total_seconds=16*repeats, tokens_per_second=16.)
    outputs = {arm: [[1]*32 for _ in prompts] for arm in timings}
    runtime = dict(timings=timings, generated_tokens=outputs, graph_static_exact=True)
    verify_runtime(runtime, prompts)
    runtime["timings"]["cuda_graph"]["tokens_per_second"] = 160.
    with pytest.raises(ValueError, match="wall samples"):
        verify_runtime(runtime, prompts)


def test_parameter_count_includes_tied_readout_once_and_excludes_buffers(tmp_path):
    import torch
    from experiments.neural_architecture_lab.finalize_jspace_allblocks import packed_parameter_count
    model = tmp_path / "native"; model.mkdir()
    (model / "config.json").write_text(json.dumps(dict(tie_word_embeddings=True, vocab_size=11, hidden_size=4)))
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    stack = {"_extra_state": dict(packed=True, stripped=True, layer_indices=tuple(range(36))),
             "fixed_buffer": torch.ones(19)}
    names = ("qkv_weight", "qkv_bias", "attention_output", "gate_up_weight", "mlp_output", "polynomial_bias")
    for layer in range(36):
        for name in names: stack[f"blocks.{layer}.{name}"] = torch.ones(2, 3, dtype=torch.bfloat16)
    norm = {"weight": torch.ones(4, dtype=torch.bfloat16)}
    (tmp_path / "evaluation").mkdir()
    torch.save(dict(fingerprint="a", stack=stack, norm=norm), tmp_path / "evaluation/packed.pt")
    exported = sum(t.numel()*t.element_size() for d in (stack, norm) for t in d.values() if isinstance(t, torch.Tensor))
    result = packed_parameter_count(tmp_path, dict(fingerprint="a", model={"path": str(model)}),
        dict(results={"compact": {"tensor_bytes": exported+88+16}}))
    assert result["model_parameters"] == 36*6*6 + 44 + 4
    assert result["external_tied_readout_bytes"] == 88 and result["other_resident_buffer_bytes"] == 16
