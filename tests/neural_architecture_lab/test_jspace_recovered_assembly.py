from copy import deepcopy
import json

import pytest

from neural_architecture_lab.jspace_recovered_assembly_record import JSpaceRecoveredAssemblyRecord, ARMS, LAYERS, scores
from experiments.neural_architecture_lab import finalize_jspace_recovered_assembly as finalizer


def records(correct):
    families = ("arc", "boolq", "hellaswag", "winogrande")
    return [dict(identity=str(i), family=families[i//16], answer=0, prediction=int(i >= correct),
        logits=[1., 0.] if i < correct else [0., 1.]) for i in range(64)]


def evidence(hybrid=24, all_compact=27):
    plan = dict(schema="jspace.recovered-assembly.v1", fingerprint="producer", arms=deepcopy(LAYERS),
        primary_arm="hybrid", count=64, minimum_correct=47, maximum_accuracy_drop=.05,
        expected_native_correct=50, expected_accepted_two_correct=49)
    counts = dict(native=50, accepted_two=49, all_compact=all_compact, hybrid=hybrid)
    runs = {}
    for arm in ARMS:
        rows = records(counts[arm])
        blocks = [dict(layer=i, compact=i in LAYERS[arm], storage_bytes=60 if i in LAYERS[arm] else 100,
            logical_tensor_bytes=60 if i in LAYERS[arm] else 100) for i in range(36)]
        size = 400+sum(b["storage_bytes"] for b in blocks)
        runs[arm] = dict(status="completed", fingerprint="producer", arm=arm, confirmed=False, compact_layers=LAYERS[arm],
            records=rows, accuracy=scores(rows), resident_tensor_state=dict(dtype="float32", blocks=blocks,
                endpoints=dict(storage_bytes=400, logical_tensor_bytes=400), storage_bytes=size, logical_tensor_bytes=size),
            metrics=dict(layers=[dict(layer=i, parameter_bytes=40 if i in LAYERS[arm] else 80) for i in range(36)]),
            control=dict(passed=True, predictions_exact=True, logits_pass=True, logit_atol=5e-4, logit_rtol=1e-5, logit_max_abs_delta=0.))
    native = runs["native"]["records"]; comparisons = {}
    for arm in ARMS[1:]:
        score = scores(runs[arm]["records"], native)
        comparisons[arm] = dict(paired=dict(overall=dict(count=64, native_correct=50, compact_correct=counts[arm],
            accuracy_drop=(50-counts[arm])/64, wins=score["paired_to_native"]["gains"], losses=score["paired_to_native"]["losses"])),
            tensor_storage_bytes=runs[arm]["resident_tensor_state"]["storage_bytes"],
            tensor_storage_reduction=1-runs[arm]["resident_tensor_state"]["storage_bytes"]/4000)
    aggregate = dict(status="completed", fingerprint="producer", primary_arm="hybrid", minimum_correct=47,
        maximum_accuracy_drop=.05, confirmed=False, confirmation_used=False, speed_measured=False,
        original_native_bf16_storage_measured=False, original_native_bf16_tensor_bytes=None,
        native_tensor_storage_bytes=4000, accuracy={a:runs[a]["accuracy"] for a in ARMS}, comparisons=comparisons,
        primary_quality_pass=hybrid >= 47)
    return dict(status="completed", confirmed=False, plan=plan, aggregate=aggregate, arms=runs, endpoint_parameters=100,
        references={a:dict(records=deepcopy(runs[a]["records"])) for a in ("native", "accepted_two")})


@pytest.mark.parametrize("hybrid,passes", [(24, False), (46, False), (47, True)])
def test_primary_classification_comes_from_assembled_hybrid(hybrid, passes):
    record = JSpaceRecoveredAssemblyRecord.from_evidence(evidence(hybrid, all_compact=50))
    assert record.primary_quality_pass is passes
    assert record.classification == "hybrid_assembly_development_gate_"+("passed" if passes else "failed")
    assert record.primary_metric == hybrid/64 and record.arms["all_compact"]["correct"] == 50
    assert record.assembly_evaluated and not record.confirmed and not record.speed_measured
    assert record.storage["hybrid"]["matched_fp32_reduction"] == pytest.approx(.34)
    assert record.model_parameters["hybrid"] == 480


@pytest.mark.parametrize("problem", ["arm", "composition", "wrong_count", "identity", "control_logits",
    "wrong_gate", "storage", "parameters", "speed", "confirmation", "bf16", "paired"])
def test_false_or_unpaired_claims_are_rejected(problem):
    value = evidence()
    if problem == "arm": del value["arms"]["all_compact"]
    elif problem == "composition": value["arms"]["hybrid"]["compact_layers"] = list(range(36))
    elif problem == "wrong_count": value["aggregate"]["accuracy"]["hybrid"]["correct"] = 47
    elif problem == "identity": value["arms"]["hybrid"]["records"][0]["identity"] = "other"
    elif problem == "control_logits": value["arms"]["native"]["records"][0]["logits"] = [3., 0.]
    elif problem == "wrong_gate": value["aggregate"]["primary_quality_pass"] = True
    elif problem == "storage": value["arms"]["hybrid"]["resident_tensor_state"]["storage_bytes"] -= 1
    elif problem == "parameters": value["arms"]["hybrid"]["metrics"]["layers"].pop()
    elif problem == "speed": value["aggregate"]["speed_measured"] = True
    elif problem == "confirmation": value["aggregate"]["confirmed"] = True
    elif problem == "bf16": value["aggregate"]["original_native_bf16_storage_measured"] = True
    else: value["aggregate"]["comparisons"]["hybrid"]["paired"]["overall"]["losses"] = 0
    with pytest.raises(ValueError): JSpaceRecoveredAssemblyRecord.from_evidence(value)


@pytest.mark.parametrize("mutation", ["arm_receipt", "selected_state"])
def test_closeout_rechecks_authenticated_evidence_before_report_or_store(tmp_path, mutation):
    value = evidence(); value["record"] = JSpaceRecoveredAssemblyRecord.from_evidence(value).to_dict()
    receipt = tmp_path/"arm.json"; receipt.write_text('{"status":"completed"}')
    checkpoint = tmp_path/"selected.pt"; checkpoint.write_bytes(b"frozen-selected-state")
    value.update(artifact_sha256={str(receipt):finalizer.sha(receipt)}, analysis_source_sha256={},
        authenticated_large_artifacts={str(checkpoint):dict(sha256=finalizer.sha(checkpoint), identity=finalizer.identity(checkpoint))})
    (tmp_path/finalizer.ANALYSIS).write_text(json.dumps(value))
    finalizer.expected(tmp_path)
    if mutation == "arm_receipt": receipt.write_text('{"status":"failed"}')
    else: checkpoint.write_bytes(b"different-selected-state")
    with pytest.raises(ValueError): finalizer.expected(tmp_path)


def test_registry_preserves_failed_primary_and_does_not_substitute_storage_or_diagnostic(tmp_path):
    (tmp_path/finalizer.ANALYSIS).write_text("analysis")
    record = JSpaceRecoveredAssemblyRecord.from_evidence(evidence(24, all_compact=50))
    metadata = finalizer.metadata(tmp_path, record)
    assert metadata["primary_arm"] == "hybrid" and metadata["primary_metric"] == 24/64
    assert metadata["all_compact_correct"] == 50 and metadata["quality_pass"] is False
    assert metadata["storage_comparison"] == "matched_fp32_only"
    assert metadata["confirmed"] is False and metadata["speed_measured"] is False


def test_registration_requires_committed_report_before_any_database_write(tmp_path, monkeypatch):
    report = tmp_path/"report.md"; report.write_text("measured result")
    record = JSpaceRecoveredAssemblyRecord.from_evidence(evidence())
    monkeypatch.setattr(finalizer, "expected", lambda _:(evidence(), record))
    def uncommitted(_): raise ValueError("Commit exact report")
    monkeypatch.setattr(finalizer, "require_committed", uncommitted)
    monkeypatch.setattr(finalizer, "logger", lambda:pytest.fail("Database opened before commit gate"))
    with pytest.raises(ValueError, match="Commit"):
        finalizer.store(tmp_path, tmp_path/"registry.json", report)
