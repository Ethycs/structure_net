from copy import deepcopy
import json

import pytest

from neural_architecture_lab.jspace_parallel_single_layer_record import (
    JSpaceParallelSingleLayerRecord, CLASSIFICATION, CONFIG)
from experiments.neural_architecture_lab import finalize_jspace_parallel_single_layer as finalizer


def records(correct):
    families = ("arc","boolq","hellaswag","winogrande")
    return [dict(identity=str(i),family=families[i//16],answer=0,prediction=int(i>=correct),
                 logits=[1.,0.] if i<correct else [0.,1.]) for i in range(64)]


def inventory():
    return dict(packed_block=dict(tensor_bytes=120,parameter_count=20,fixed_geometry_and_buffer_bytes=40),
        selected_unpacked=dict(tensor_bytes=128),native_matched_fp32_bytes=10000,native_source_tensor_bytes=5000,
        native_block_fp32_bytes=200,native_block_source_bytes=100,native_parameters=2500,
        selected_whole_parameters=2470,native_headers={},selected_checkpoint_file_bytes=160)


def evidence():
    trials = []
    features = dict(relative_mse=.2,mse=.01,first_relative_mse=.1,content_relative_mse=.2,
        answer_relative_mse=.3,irreducible_outside_relative_mse=.1,retained_relative_mse=.1)
    for layer in range(2,36):
        selected = {12:44,15:46}.get(layer,48)
        initial = 49 if layer in (20,25) else selected-1
        history = []
        for epoch in range(13):
            metrics = dict(**features,correct=initial if epoch==0 else selected,count=64)
            metrics["relative_mse"] = .3-epoch*.01
            history.append(dict(epoch=epoch,metrics=metrics,checkpoint_sha256="checkpoint"))
        winner = max(history,key=lambda r:(r["metrics"]["correct"],-r["metrics"]["relative_mse"]))
        selected = winner["metrics"]["correct"];epoch = winner["epoch"]
        norm = dict(train_case_count=512,median_teacher_update_energy=.2,floor_fraction=1e-4,floor=.00002)
        plan = dict(layer=layer,trained_layers=[layer],native_prefix=list(range(layer)),native_suffix=list(range(layer+1,36)),
            config=CONFIG.copy(),rank=1024,head_dim=128,intermediate_size=11008,confirmation_used=False,fingerprint=str(layer))
        train = dict(status="completed",systems_only=False,strict_reload_exact=True,updates=384,trained_layer=layer,
            history=history,selected_epoch=epoch,selected=winner["metrics"],selected_checkpoint_sha256="checkpoint",
            initializer_selected=epoch==0,native_correct=50,minimum_correct=47,quality_pass=selected>=47,normalization=norm)
        fresh = dict(status="completed",passed=True,strict_state_load=True,fingerprint=str(layer),campaign_fingerprint=str(layer),
            checkpoint_sha256="checkpoint",selected_epoch=epoch,confirmed=False,training_quality_pass=selected>=47,
            comparison=dict(passed=True,feature_pass=True,logits_pass=True,predictions_exact=True,logit_max_abs_delta=0.),records=records(selected))
        pilot = dict(systems_only=True,updates=1,native_control=dict(passed=True,predictions_exact=True,logits_pass=True,correct=50,count=64))
        trials.append(dict(plan=plan,training=train,verification=fresh,pilot=pilot,initial_records=records(initial),selected_records=records(selected)))
    return dict(status="completed",confirmed=False,coordinator_result=dict(status="completed",confirmed=False,assembly_evaluated=False,
        verified_layers=list(range(2,36)),failed_layers=[],unattempted_layers=[],isolated_quality_misses=[12,15]),
        native_records=records(50),trials=trials,storage=finalizer.summarize_storage([(i,inventory()) for i in range(2,36)]))


def test_34_independent_results_are_counts_not_model_accuracy():
    record = JSpaceParallelSingleLayerRecord.from_evidence(evidence())
    assert record.classification == CLASSIFICATION and record.primary_metric == 32
    assert record.summary["primary_metric_unit"] == "count" and record.summary["unique_development_cases"] == 64
    assert record.summary["optimizer_updates"] == 13056 and record.summary["initializer_selected_layers"] == [20,25]
    assert record.summary["isolated_quality_misses"] == [12,15]
    assert not record.confirmed and not record.assembly_evaluated
    assert "accuracy" not in record.summary
    assert record.layers[10]["selected"]["correct"] == 44


@pytest.mark.parametrize("problem",["missing_trial","updates","selection","checkpoint","failed_verification","quality_miss_hidden",
    "changed_family","native_prefix","confirmation","assembly","normalization","pilot","storage"])
def test_incomplete_or_inflated_evidence_rejected(problem):
    value = evidence();trial = value["trials"][0]
    if problem == "missing_trial":value["trials"].pop()
    elif problem == "updates":trial["training"]["updates"] = 383
    elif problem == "selection":trial["training"]["selected_epoch"] = 4
    elif problem == "checkpoint":trial["verification"]["checkpoint_sha256"] = "wrong"
    elif problem == "failed_verification":trial["verification"]["passed"] = False
    elif problem == "quality_miss_hidden":value["coordinator_result"]["isolated_quality_misses"] = []
    elif problem == "changed_family":trial["selected_records"][0]["family"] = "boolq"
    elif problem == "native_prefix":trial["plan"]["native_prefix"] = []
    elif problem == "confirmation":value["confirmed"] = True
    elif problem == "assembly":value["coordinator_result"]["assembly_evaluated"] = True
    elif problem == "normalization":trial["training"]["normalization"]["floor"] *= 10
    elif problem == "pilot":trial["pilot"]["native_control"]["logits_pass"] = False
    elif problem == "storage":value["storage"]["all36_matched_fp32_projection"]["bytes"] += 1
    with pytest.raises(ValueError):JSpaceParallelSingleLayerRecord.from_evidence(value)


def test_identical_tensor_shapes_and_precision_comparisons_are_explicit():
    rows = [(i,inventory()) for i in range(2,36)]
    result = finalizer.summarize_storage(rows)
    assert result["all36_matched_fp32_projection"]["bytes"] == 7120
    assert result["all36_source_bf16_plus_fp32_compact_projection"]["bytes"] == 5720
    assert result["all36_matched_fp32_projection"]["reduction_fraction"] > 0
    assert result["all36_source_bf16_plus_fp32_compact_projection"]["reduction_fraction"] < 0
    assert result["independent_packed_state_bank_bytes"] == 4080
    broken = deepcopy(rows);broken[1][1]["packed_block"]["tensor_bytes"] += 4
    with pytest.raises(ValueError,match="shapes"):finalizer.summarize_storage(broken)


@pytest.mark.parametrize("mutation",["small_receipt","checkpoint"])
def test_later_stages_reject_changed_authenticated_bytes(tmp_path,mutation):
    value = evidence();value["record"] = JSpaceParallelSingleLayerRecord.from_evidence(value).to_dict()
    receipt = tmp_path/"verification.json";receipt.write_text('{"passed":true}')
    checkpoint = tmp_path/"selected.pt";checkpoint.write_bytes(b"authenticated-state")
    value.update(artifact_sha256={str(receipt):finalizer.sha(receipt)},analysis_source_sha256={},
        authenticated_large_artifacts={str(checkpoint):dict(sha256=finalizer.sha(checkpoint),identity=finalizer.identity(checkpoint))})
    (tmp_path/finalizer.ANALYSIS).write_text(json.dumps(value))
    finalizer.expected(tmp_path)
    if mutation == "small_receipt":receipt.write_text('{"passed":false}')
    else:checkpoint.write_bytes(b"changed-state")
    with pytest.raises(ValueError):finalizer.expected(tmp_path)


def test_registry_metadata_does_not_expose_isolated_pass_fraction_as_accuracy(tmp_path):
    (tmp_path/finalizer.ANALYSIS).write_text("analysis")
    value = finalizer.metadata(tmp_path,JSpaceParallelSingleLayerRecord.from_evidence(evidence()))
    assert value["primary_metric"] == 32 and value["primary_metric_unit"] == "count"
    assert "accuracy" not in value and value["assembly_evaluated"] is False
