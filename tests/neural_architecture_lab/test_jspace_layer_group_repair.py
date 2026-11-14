"""Repair evidence must measure the trained model, not its search surrogate."""
from copy import deepcopy
import json
import math

import pytest

from neural_architecture_lab.jspace_layer_group_repair_record import (
    CONFIG, JSpaceLayerGroupRepairRecord, classify,
)


LAYERS = [2, 14, 30]
FAMILIES = ("arc", "boolq", "hellaswag", "winogrande")


def records(correct, epoch=0):
    result = []
    for index in range(64):
        choices = 4 if index // 16 in (0, 2) else 2
        prediction = int(index >= correct)
        logits = [0.] * choices
        logits[prediction] = 1. + .1 * epoch
        ce = math.log(sum(math.exp(x) for x in logits)) - logits[0]
        features = {str(layer): .6 - .05 * epoch + .01 * j + .0001 * index
                    for j, layer in enumerate(LAYERS)}
        result.append(dict(identity=str(index), family=FAMILIES[index // 16],
            answer=0, prediction=prediction, logits=logits, teacher_prediction=int(index >= 50),
            full_kl=.7 - .04 * epoch, choice_kl=.9 - .04 * epoch, choice_ce=ce,
            choice_margin=logits[0] - max(logits[1:]), valid_label_probability=.95,
            answer_full_vocab_ce=ce - math.log(.95), ce=ce - math.log(.95),
            feature_loss=sum(features.values()) / len(features), feature_loss_by_layer=features))
    return result


def metrics_for(rows):
    mean = lambda key: sum(row[key] for row in rows) / len(rows)
    metrics = {key: mean(key) for key in (
        "full_kl", "choice_kl", "choice_ce", "choice_margin", "valid_label_probability",
        "answer_full_vocab_ce", "feature_loss")}
    metrics.update(weighted_full_kl=.8 * metrics["full_kl"],
                   weighted_choice_kl=2. * metrics["choice_kl"],
                   weighted_choice_ce=.3 * metrics["choice_ce"],
                   weighted_feature_loss=metrics["feature_loss"])
    metrics["endpoint_loss"] = sum(metrics[key] for key in (
        "weighted_full_kl", "weighted_choice_kl", "weighted_choice_ce"))
    metrics["loss"] = metrics["endpoint_loss"] + metrics["weighted_feature_loss"]
    correct = sum(row["prediction"] == row["answer"] for row in rows)
    metrics.update(count=64, correct=correct, accuracy=correct / 64, teacher_accuracy=50 / 64,
        teacher_agreement=sum(r["prediction"] == r["teacher_prediction"] for r in rows) / 64,
        kl=metrics["full_kl"], ce=metrics["answer_full_vocab_ce"],
        selection="highest_dev_accuracy_then_lowest_conditional_choice_CE")
    return metrics


def evidence(counts=(27, 30, 33, 36, 38, 39, 40), *, feature_scale=1.):
    native = records(50)
    epochs, history = [], []
    for epoch, count in enumerate(counts):
        rows = records(count, epoch)
        for row in rows:
            row["feature_loss_by_layer"] = {
                key: value * feature_scale for key, value in row["feature_loss_by_layer"].items()}
            row["feature_loss"] *= feature_scale
        metrics = metrics_for(rows)
        history.append(dict(epoch=epoch, metrics=metrics,
                            checkpoint_manifest_sha256=f"checkpoint-{epoch}"))
        epochs.append(dict(epoch=epoch, metrics=metrics, records=rows))
    winner = max(history, key=lambda row: (row["metrics"]["correct"], -row["metrics"]["choice_ce"]))
    chosen = winner["epoch"]
    passed = counts[chosen] >= 47
    selection = dict(selected_layers=LAYERS.copy(), search_fingerprint="search")
    frozen = dict(sha256="fixed", tensor_count=350, bytes=10000)
    return dict(status="completed", confirmed=False,
        plan=dict(schema="jspace.layer-group-repair.v1", config=deepcopy(CONFIG),
            initial_correct=27, native_correct=50, minimum_correct=47, confirmed=False,
            fingerprint="campaign", assembly_fingerprint="assembly", tokens=20000,
            selected_layers=LAYERS.copy(), selection=deepcopy(selection),
            partition={"3060": list(range(24)), "2060_SUPER": list(range(24, 36))}),
        selection_proof=deepcopy(selection),
        training=dict(status="completed", confirmed=False, systems_only=False, updates=192,
            strict_reload_exact=True, frozen_state_unchanged=True, frozen_state=deepcopy(frozen),
            history=history, initial=history[0]["metrics"], selected=winner["metrics"],
            selected_epoch=chosen, selected_layers=LAYERS.copy(),
            selected_manifest_sha256=winner["checkpoint_manifest_sha256"],
            initializer_selected=chosen == 0, minimum_correct=47, native_correct=50, quality_pass=passed),
        verification=dict(status="completed", confirmed=False, fingerprint="campaign",
            selected_epoch=chosen, selected_layers=LAYERS.copy(),
            selected_manifest_sha256=winner["checkpoint_manifest_sha256"], passed=True,
            strict_state_load=True, predictions_exact=True, logits_pass=True, metrics_pass=True,
            frozen_state_unchanged=True, records=deepcopy(epochs[chosen]["records"]),
            metrics=deepcopy(winner["metrics"]), training_quality_pass=passed),
        pilot=dict(systems_only=True, updates=1, initial_control=dict(passed=True, correct=27),
            changed_selected_layers=LAYERS.copy(), frozen_state_unchanged=True,
            update=dict(gradient_parameter_count=21)),
        teacher=dict(status="completed", passed=True, dtype="float32", shape=[576, 2, 151936],
            slots=[1, 3], records=native, correct=50, count=64,
            control=dict(passed=True, predictions_exact=True, logits_pass=True)),
        capture=dict(status="completed", passed=True, fingerprint="campaign", dtype="float32",
            shape=[20000, 2048], selected_layers=LAYERS.copy(), native_control=dict(passed=True),
            parity_atol=1e-5, parity_rtol=2e-4, records=deepcopy(native),
            development_boundary_parity={str(i): dict(passed=True, max_abs_delta=0.) for i in LAYERS}),
        epochs=epochs, frozen_state=dict(passed=True, before=deepcopy(frozen), after=deepcopy(frozen)),
        delta_manifest=dict(schema="jspace.layer-group-repair.v1", selected_layers=LAYERS.copy(),
            parent_assembly_fingerprint="assembly", search_fingerprint="search",
            artifacts={f"block_{i:02d}.pt": f"sha-{i}" for i in LAYERS},
            metadata=dict(epoch=chosen, development=deepcopy(winner["metrics"]))),
        search_context=dict(records=records(50), selection=dict(layers=LAYERS.copy(), group_size=3,
            restored_correct=50, rescue_correct=23)),
        storage=dict(all_fp32=True, trainable_layers=LAYERS.copy(), parameter_tensor_count=21,
            native_norm_frozen=True, tied_embedding_readout_counted_once=True,
            model_parameters=3000, trainable_parameters=300))


def test_primary_is_repaired_model_accuracy_not_native_restoration_search_score():
    record = JSpaceLayerGroupRepairRecord.from_evidence(evidence())
    assert record.primary_metric == 40 / 64
    assert record.search_context["native_restored_score"]["correct"] == 50
    assert record.classification == "partial_recovery_below_gate" and not record.quality_pass
    assert record.selected_layers == LAYERS and record.updates == 192
    assert record.selected["paired_to_initial"]["gains"] == 13
    assert record.selected["paired_to_native"]["losses"] == 10
    assert not record.confirmed and not record.speed_measured and not record.packing_validated
    assert record.independent_confirmations == 0


@pytest.mark.parametrize("correct,expected", [(27, "no_recovery_over_warm_start"),
    (28, "partial_recovery_below_gate"), (46, "partial_recovery_below_gate"),
    (47, "development_gate_passed")])
def test_cumulative_gate_is_applied_once_to_actual_model(correct, expected):
    assert classify(correct) == expected
    value = evidence((27, 25, 24, 24, 25, 26, correct))
    record = JSpaceLayerGroupRepairRecord.from_evidence(value)
    assert record.classification == expected
    assert record.quality_pass == (correct >= 47) and not record.confirmed


def test_initializer_remains_eligible_and_zero_feature_loss_is_not_task_recovery():
    record = JSpaceLayerGroupRepairRecord.from_evidence(
        evidence((27, 26, 25, 24, 23, 22, 21), feature_scale=0.))
    assert record.selected_epoch == 0 and record.initializer_selected
    assert record.selected["metrics"]["feature_loss"] == 0
    assert record.primary_metric == 27 / 64 and record.classification == "no_recovery_over_warm_start"


def test_equal_accuracy_uses_conditional_ce_before_feature_fit():
    value = evidence((27, 48, 48, 48, 48, 48, 48))
    expected = min(value["epochs"][1:], key=lambda row: row["metrics"]["choice_ce"])
    record = JSpaceLayerGroupRepairRecord.from_evidence(value)
    assert record.selected_epoch == expected["epoch"]
    assert record.primary_metric == .75 and not record.confirmed


@pytest.mark.parametrize("problem", ["epochs", "updates", "missing_epoch", "selected_layer", "selection_proof",
    "teacher_precision", "teacher_decision", "capture_shape", "capture_parity", "gradient_count", "updated_layer",
    "fixed_state", "norm_delta", "delta_parent", "delta_epoch", "fresh_checkpoint", "fresh_fixed", "gate",
    "confirmed", "storage", "record_identity", "aggregate_accuracy", "choice_ce", "margin", "feature_layers",
    "feature_mean", "feature_aggregate", "objective", "endpoint_weight"])
def test_incomplete_or_inconsistent_scientific_evidence_is_rejected(problem):
    value = evidence()
    if problem == "epochs": value["plan"]["config"]["epochs"] = 3
    elif problem == "updates": value["training"]["updates"] = 191
    elif problem == "missing_epoch": value["epochs"].pop()
    elif problem == "selected_layer": value["training"]["selected_layers"] = [2, 14, 31]
    elif problem == "selection_proof": value["selection_proof"]["search_fingerprint"] = "other"
    elif problem == "teacher_precision": value["teacher"]["dtype"] = "bfloat16"
    elif problem == "teacher_decision": value["epochs"][1]["records"][0]["teacher_prediction"] = 1
    elif problem == "capture_shape": value["capture"]["shape"][0] -= 1
    elif problem == "capture_parity": value["capture"]["development_boundary_parity"]["14"]["passed"] = False
    elif problem == "gradient_count": value["pilot"]["update"]["gradient_parameter_count"] = 22
    elif problem == "updated_layer": value["pilot"]["changed_selected_layers"] = [2, 14]
    elif problem == "fixed_state": value["frozen_state"]["after"]["sha256"] = "changed"
    elif problem == "norm_delta": value["delta_manifest"]["artifacts"]["norm_gain.pt"] = "norm"
    elif problem == "delta_parent": value["delta_manifest"]["parent_assembly_fingerprint"] = "other"
    elif problem == "delta_epoch": value["delta_manifest"]["metadata"]["epoch"] = 0
    elif problem == "fresh_checkpoint": value["verification"]["selected_manifest_sha256"] = "other"
    elif problem == "fresh_fixed": value["verification"]["frozen_state_unchanged"] = False
    elif problem == "gate": value["training"]["quality_pass"] = True
    elif problem == "confirmed": value["confirmed"] = True
    elif problem == "storage": value["storage"]["native_norm_frozen"] = False
    elif problem == "record_identity": value["epochs"][1]["records"][0]["identity"] = "other"
    elif problem == "aggregate_accuracy": value["epochs"][1]["metrics"]["correct"] += 1
    elif problem == "choice_ce": value["epochs"][1]["records"][0]["choice_ce"] += .1
    elif problem == "margin": value["epochs"][1]["records"][0]["choice_margin"] += .1
    elif problem == "feature_layers": value["epochs"][1]["records"][0]["feature_loss_by_layer"]["3"] = .1
    elif problem == "feature_mean": value["epochs"][1]["records"][0]["feature_loss"] += .1
    elif problem == "feature_aggregate": value["epochs"][1]["metrics"]["feature_loss"] += .1
    elif problem == "objective": value["epochs"][1]["metrics"]["loss"] += .1
    elif problem == "endpoint_weight": value["epochs"][1]["metrics"]["weighted_choice_kl"] += .1
    with pytest.raises(ValueError):
        JSpaceLayerGroupRepairRecord.from_evidence(value)


@pytest.mark.parametrize("change", ["logits", "features"])
def test_fresh_records_must_match_selection_even_when_internal_metrics_are_consistent(change):
    value = evidence()
    rows = value["verification"]["records"]
    if change == "logits":
        # A common logit offset preserves predictions, CE and margins, but fails replay parity.
        for row in rows: row["logits"] = [x + .01 for x in row["logits"]]
    else:
        for row in rows:
            row["feature_loss_by_layer"]["14"] += .01
            row["feature_loss"] = sum(row["feature_loss_by_layer"].values()) / 3
        value["verification"]["metrics"] = metrics_for(rows)
    with pytest.raises(ValueError, match="Fresh decision/feature"):
        JSpaceLayerGroupRepairRecord.from_evidence(value)


def test_registry_metric_keeps_learned_accuracy_and_search_context_separate(tmp_path):
    from experiments.neural_architecture_lab import finalize_jspace_layer_group_repair as finalizer
    (tmp_path / finalizer.ANALYSIS).write_text("analysis")
    record = JSpaceLayerGroupRepairRecord.from_evidence(evidence())
    metadata = finalizer.metadata(tmp_path, record)
    assert metadata["primary_metric"] == 40 / 64 and metadata["correct"] == 40
    assert metadata["primary_metric_name"] == "selected_all_compact_group_repair_development_accuracy"
    assert metadata["native_restored_search_correct"] == 50
    assert metadata["trained_block_count"] == 3 and json.loads(metadata["trained_layers_json"]) == LAYERS
    assert metadata["native_norm_frozen"] is True and metadata["updates"] == 192
    assert metadata["confirmed"] is False and metadata["independent_confirmations"] == 0
    assert metadata["speed_measured"] is False and metadata["packing_validated"] is False


@pytest.mark.parametrize("which", ["receipt", "weights", "source", "archive", "typed_record"])
def test_readback_rejects_changed_analysis_dependencies(tmp_path, monkeypatch, which):
    from experiments.neural_architecture_lab import finalize_jspace_layer_group_repair as finalizer
    monkeypatch.setattr(finalizer, "ROOT", tmp_path)
    root = tmp_path / "campaign"
    root.mkdir()
    receipt = root / "verification.json"
    receipt.write_text("verified")
    weights = root / "block.pt"
    weights.write_bytes(b"selected weights")
    source = tmp_path / "producer.py"
    source.write_text("frozen source")
    archive = root / finalizer.ARCHIVE / "producer.py"
    archive.parent.mkdir()
    archive.write_bytes(source.read_bytes())
    value = evidence()
    value["record"] = JSpaceLayerGroupRepairRecord.from_evidence(value).to_dict()
    value.update(artifact_sha256={str(receipt): finalizer.sha(receipt)},
        analysis_source_sha256={str(source): finalizer.sha(source)},
        authenticated_large_artifacts={str(weights): dict(sha256=finalizer.sha(weights),
                                                          identity=finalizer.identity(weights))})
    (root / finalizer.ANALYSIS).write_text(json.dumps(value))
    _, record = finalizer.expected(root)
    assert record.selected["correct"] == 40
    if which == "receipt": receipt.write_text("failed")
    elif which == "weights": weights.write_bytes(b"changed weights")
    elif which == "source": source.write_text("changed source")
    elif which == "archive": archive.write_text("changed archived source")
    else:
        value["record"]["selected"]["correct"] = 50
        (root / finalizer.ANALYSIS).write_text(json.dumps(value))
    with pytest.raises(ValueError):
        finalizer.expected(root)
