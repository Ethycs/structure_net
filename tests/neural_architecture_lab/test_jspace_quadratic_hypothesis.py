"""Measured CPU prefill, task quality and complete training jointly gate results."""
from copy import deepcopy
import json

import pytest

from neural_architecture_lab.jspace_quadratic_hypothesis import (
    ARMS, FAIL, PASS, STUDY_ID, JSpaceQuadraticRecord, benchmark_measurements, paired_quality, required_stages,
)


def rows(deltas=(0, 0, 0, 0)):
    return [dict(identity=f"{family}-{i}", family=family, answer=0, prediction=0 if i < 100 + delta else 1)
        for family, delta in zip(("arc", "boolq", "hellaswag", "winogrande"), deltas) for i in range(128)]


def timing(speed=1.6):
    latencies = {}
    for arm in ("native", "compact"):
        latencies[arm] = {}
        for mode in ("eager", "compiled"):
            latencies[arm][mode] = {}
            for length in (128, 256):
                seconds = 2. if arm == "native" else 2. / speed
                latencies[arm][mode][str(length)] = dict(median_seconds=seconds,
                    samples=[dict(seconds=seconds, repeat=r, prompt=p) for r in range(5) for p in range(2)])
    return dict(threads=1, repeats=5, lengths=[128, 256], latencies=latencies,
        compilation=dict(backend="inductor", fullgraph=True, dynamic=False, eager_graph_break_fallback=False,
            options={"freezing": True}, fx_graphs={"native": [{"code": "native"}], "compact": [{"code": "compact"}]}),
        compiled_speedup={"128": speed, "256": speed}, resident_tensor_bytes={"native": 10000, "compact": 6000},
        parity=[dict(model=a, length=l, prompt=p, greedy_equal=True) for a in ("native", "compact") for l in (128, 256) for p in range(2)],
        all_compiled_greedy_equal=True)


def evidence(*, speed=1.6, deltas=(-12, -4, -4, -4)):
    config = dict(layers=36, rank=1024, head_dim=64, native_mlp_width=11008, epochs=12, temperature=2.,
        weight_decay=0., hyperdeterminant=False, dtype="float32", minimum_size_reduction=.38,
        minimum_speedup=1.4, maximum_accuracy_drop=.05, cpu_threads=1,
        microbatch=4, accumulation=4, learning_rate=1.2e-4, gradient_clip=1., seed=27861)
    plan = dict(schema="jspace.quadratic-recipe.v1", study_id=STUDY_ID, fingerprint="a"*64, config=config)
    records = dict(native=rows(), softmax=rows((-1, -1, -1, -1)), quadratic=rows(deltas))
    timings = {arm: timing(speed) for arm in ARMS}
    summaries = {}
    for arm in ARMS:
        quality = paired_quality(records["native"], records[arm])
        gates = dict(aggregate_accuracy=quality["overall"]["accuracy_drop"] <= .05,
                     tensor_reduction=True, cpu_prefill_speed=speed >= 1.4, compiled_greedy_parity=True)
        summaries[arm] = dict(quality=quality, gates=gates, targets_met=all(gates.values()), tensor_reduction=.4,
            tensor_bytes=timings[arm]["resident_tensor_bytes"], cpu_compiled_prefill_speedups=timings[arm]["compiled_speedup"])
    result = dict(schema=plan["schema"], study_id=STUDY_ID, fingerprint=plan["fingerprint"], status="completed", confirmed=False,
        targets_met=summaries["quadratic"]["targets_met"], arms=summaries,
        quadratic_vs_softmax=paired_quality(records["softmax"], records["quadratic"]))
    training = {}
    for arm in ARMS:
        for layer in range(36):
            history = [{"epoch": epoch, "development": {"loss": .7 if epoch == 3 else 1., "count": 64}} for epoch in range(13)]
            training[f"{arm}/{layer}"] = dict(status="completed", mode=arm, layer=layer, config=config,
                completed_epochs=12, updates=384, objective="normalized_feature + T^2 KL" if layer == 35 else "normalized_feature",
                final_layer=layer == 35, disk_reload_exact=True, history=history, selected_epoch=3,
                selected=deepcopy(history[3]["development"]), trainable_parameters={"parameter": {"gradient_updates": 384, "optimizer_updates": 384}})
    return dict(plan=plan, result=result, records=records, timing=timings, training=training,
        exports={arm: {"tensor_bytes": 6000, "model_parameters": 1000} for arm in ARMS},
        manifests={name: {"status": "completed", "fingerprint": plan["fingerprint"]} for name in required_stages()})


def test_complete_positive_measurement_remains_exploratory_and_aggregate_only():
    record = JSpaceQuadraticRecord.from_evidence(evidence())
    assert record.classification == PASS and record.targets_met
    assert record.native_correct == 400 and record.softmax_correct == 396 and record.quadratic_correct == 376
    assert record.confirmed is False and record.independent_confirmations == 0
    assert "every_family_quality" not in record.gates["quadratic"]
    assert "prefill" in record.scope and "512-case compiled-CPU accuracy" in record.scope
    assert json.loads(json.dumps(record.to_dict())) == record.to_dict()


@pytest.mark.parametrize("kwargs", [{"speed": 1.39}, {"deltas": (-26, 0, 0, 0)}])
def test_completed_negative_measurements_are_retained(kwargs):
    record = JSpaceQuadraticRecord.from_evidence(evidence(**kwargs))
    assert record.classification == FAIL and not record.targets_met


@pytest.mark.parametrize("fault", ["partial", "budget", "objective", "selection", "gradients", "confirmation", "promote", "fingerprint"])
def test_partial_or_misreported_training_cannot_be_registered(fault):
    value = evidence()
    if fault == "partial":
        del value["manifests"]["arms/quadratic/layer_02"]
    elif fault == "budget":
        value["training"]["quadratic/5"]["updates"] = 383
    elif fault == "objective":
        value["training"]["quadratic/35"]["objective"] = "cross_entropy"
    elif fault == "selection":
        value["training"]["quadratic/5"]["selected_epoch"] = 12
    elif fault == "gradients":
        value["training"]["quadratic/35"]["trainable_parameters"]["parameter"]["gradient_updates"] = 0
    elif fault == "confirmation":
        value["result"]["confirmed"] = True
    elif fault == "promote":
        value = evidence(speed=1.39); value["result"]["targets_met"] = True
    else:
        value["manifests"]["cpu_benchmark/quadratic"]["fingerprint"] = "different"
    with pytest.raises(ValueError):
        JSpaceQuadraticRecord.from_evidence(value)


@pytest.mark.parametrize("fault", ["median", "samples", "eager_ratio", "fallback", "parity", "native_size"])
def test_false_speedups_or_unmatched_controls_are_rejected(fault):
    value = evidence(); bench = value["timing"]["quadratic"]
    if fault == "median":
        bench["latencies"]["compact"]["compiled"]["128"]["median_seconds"] /= 2
    elif fault == "samples":
        bench["latencies"]["native"]["compiled"]["128"]["samples"][0]["repeat"] = 4
    elif fault == "eager_ratio":
        bench["compiled_speedup"]["128"] = 20.
    elif fault == "fallback":
        bench["compilation"]["eager_graph_break_fallback"] = True
    elif fault == "parity":
        bench["parity"][0]["greedy_equal"] = False
    else:
        bench["resident_tensor_bytes"]["native"] += 1
    with pytest.raises(ValueError):
        JSpaceQuadraticRecord.from_evidence(value)


def test_paired_architecture_control_requires_identical_examples_and_labels():
    native, candidate = rows(), rows()
    candidate[0]["identity"] = "different"
    with pytest.raises(ValueError, match="Unpaired"):
        paired_quality(native, candidate)


def test_registry_readback_checks_exact_metrics_and_nonconfirmation(tmp_path, monkeypatch):
    from experiments.neural_architecture_lab import finalize_jspace_quadratic as finalizer
    value = evidence()
    record = JSpaceQuadraticRecord.from_evidence(value)
    value.update(record=record.to_dict(), verification_sha256={}, analysis_source_sha256={})
    (tmp_path / "quadratic_closeout_analysis.json").write_text(json.dumps(value))
    experiment = finalizer.metadata(tmp_path, record)
    hypothesis = dict(tested=True, confirmed=False, confirmation_status=record.classification,
        evidence_count=1, direct_result_hashes_json=json.dumps(["stored-result"]))
    class Collection:
        def __init__(self, key, metadata):
            self.key, self.metadata = key, metadata
        def get(self, ids, include):
            assert ids == [self.key] and include == ["metadatas"]
            return {"ids": [self.key], "metadatas": [self.metadata]}
    class Logger:
        experiments_collection = Collection("stored-result", experiment)
        hypotheses_collection = Collection(STUDY_ID, hypothesis)
    monkeypatch.setattr(finalizer.common, "logger", lambda: Logger())
    result = finalizer.readback(tmp_path, tmp_path / "unused-registry.json", "stored-result")
    assert result["readback_verified"] and result["confirmed"] is False
    experiment["correct"] = 512
    with pytest.raises(ValueError, match="Experiment readback"):
        finalizer.readback(tmp_path, tmp_path / "unused-registry.json", "stored-result")
    experiment["correct"] = record.quadratic_correct
    hypothesis["confirmed"] = True
    with pytest.raises(ValueError, match="Hypothesis evidence"):
        finalizer.readback(tmp_path, tmp_path / "unused-registry.json", "stored-result")
