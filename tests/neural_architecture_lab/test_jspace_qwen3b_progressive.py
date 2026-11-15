from copy import deepcopy
import json
from pathlib import Path

import pytest

from neural_architecture_lab.jspace_qwen3b_progressive_record import (
    JSpaceQwen3BProgressiveRecord as Record, CONFIG, STUDY_ID, quality_gates,
)
from experiments.neural_architecture_lab import finalize_jspace_qwen3b_progressive as final


def rows(correct):
    value=[];families=("arc","boolq","hellaswag","winogrande")
    for i in range(64):
        family=families[i//16];choices=2 if family in ("boolq","winogrande") else 4
        answer=i%choices;prediction=answer if i<correct else (answer+1)%choices
        logits=[-1.]*choices;logits[prediction]=2.
        value.append(dict(identity=f"{family}/{i%16}",family=family,answer=answer,prediction=prediction,logits=logits))
    return value


def metrics(correct,ce=1.):
    return dict(correct=correct,count=64,accuracy=correct/64,kl=.1,ce=ce,loss=.28+.3*ce)


def storage(cut):
    packed=(36-cut)*100+50;prefix=cut*200;total=packed+prefix+1000+20
    return dict(cut=cut,compressed_layers=36-cut,packed_suffix_and_norm_bytes=packed,
        frozen_native_prefix_bytes=prefix,tied_embedding_readout_bytes=1000,rotary_bytes=20,
        full_model_tensor_bytes=total,native_full_tensor_bytes=10000,
        full_model_reduction_fraction=1-total/10000,model_parameters=3000)


def stage(cut=28,selected=51,parent=51):
    layers=list(range(cut,36));reports=[];history=[]
    for epoch in range(13):
        correct=48 if epoch==0 and selected>=48 else selected
        ce=.5 if epoch==3 else 1.+epoch/100
        m=metrics(correct,ce)
        reports.append(dict(metrics=m,records=rows(correct)))
        history.append(dict(epoch=epoch,metrics=m,checkpoint_sha256=f"checkpoint{epoch}"))
    chosen=history[3]
    gates=quality_gates(selected,48,parent)
    quality=dict(passed=gates["quality_pass"],minimum_correct=gates["minimum_correct"],
        native=dict(correct=48,minimum_correct=gates["native_minimum_correct"],passed=gates["native_pass"]),
        parent=dict(correct=parent,minimum_correct=gates["parent_minimum_correct"],passed=gates["parent_pass"]))
    def control(correct):
        return dict(passed=True,predictions_exact=True,logits_pass=True,max_abs_logit_delta=0.,
                    correct=correct,count=64,accuracy=correct/64,atol=.03125,rtol=1/128)
    controls={k:control(parent) for k in ("parent","seam")}
    controls["native"]=control(48)
    fixed={"geometry":"fixed","E":"frozen"}
    raw=dict(capture=dict(status="completed",controls=controls,
        teacher_cache_diagnostic=dict(predictions_exact=True,correct=parent),teacher_cache_logit_gate_applied=False),
        fit=dict(status="completed"),pilot=dict(status="completed",passed=True,systems_only=True,discarded=True,
            layer_indices=layers,fp32_master_changed_by_block=[True]*len(layers),norm_fp32_master_changed=True,
            fixed_state_unchanged=True,checkpoint_strict_load=True,inheritance=dict(passed=True),fixed_state=fixed,
            update=dict(gradients_present_by_block=[True]*len(layers)),initial_records=reports[0]["records"]),
        train=dict(status="completed",systems_only=False,confirmed=False,config=deepcopy(CONFIG),completed_epochs=12,
            updates=384,cut=cut,layer_indices=layers,suffix_blocks=len(layers),strict_reload_exact=True,
            fixed_state_unchanged=True,fixed_state=fixed,history=history,selected_epoch=3,selected=chosen["metrics"],
            selected_checkpoint_sha256=chosen["checkpoint_sha256"],initial=history[0]["metrics"],initializer_selected=False,
            quality_pass=gates["quality_pass"],quality=dict(quality,count=64,maximum_drop=.05)),
        verify=dict(status="completed",passed=True,confirmed=False,fixed_state_unchanged=True,
            selected_checkpoint_sha256=chosen["checkpoint_sha256"],metrics=chosen["metrics"],
            records=reports[3]["records"],quality=quality,storage=storage(cut)))
    return dict(status="completed",plan=dict(cut=cut,parent_cut=cut+1,parent_correct=parent,native_correct=48,layers=layers,
        calibration_seed=20861+100000*(29-cut)),raw_stages=raw,epochs=reports,
        update_audit=dict(count=384,all_blocks_and_norm=True),selected_checkpoint=f"cut{cut}.pt",training_seconds=20.)


def evidence(stages=None,outcome=None):
    return dict(confirmed=False,plan=dict(confirmed=False,schema="jspace.qwen3b-progressive.v1",study_id=STUDY_ID,
        config=deepcopy(CONFIG),seed_cut=29,target_cut=0,seed_correct=51,native_correct=48,seed_checkpoint="seed.pt"),
        native_records=rows(48),seed_records=rows(51),seed_storage=storage(29),stages=stages or [],outcome=outcome)


def test_initial_seed_and_partial_progress_are_not_full_compression():
    r=Record.from_evidence(evidence())
    assert r.primary_metric==7 and r.accepted_correct==51 and not r.terminal
    r=Record.from_evidence(evidence([stage()]))
    assert r.primary_metric==8 and r.accepted_cut==28 and r.accepted_correct==51
    assert r.classification=="progressive_in_progress" and not r.confirmed and not r.speed_measured
    assert r.stages[0]["control_rows_independently_recomputed"] is False


def test_frozen_producer_actual_control_and_two_quality_schemas_are_supported():
    from experiments.structure_net import jspace_qwen3b_progressive as producer
    value=evidence([stage()]);s=value["stages"][0]
    cases=[dict(r,choices=list(range(len(r["logits"])))) for r in rows(51)]
    controls=s["raw_stages"]["capture"]["controls"]
    for name,n in (("parent",51),("seam",51),("native",48)):
        controls[name]=producer.old.compare_rows(rows(n),rows(n),cases,expected_correct=n)
        assert "records" not in controls[name]
    s["raw_stages"]["train"]["quality"]=producer.training.gates(51,native_correct=48,parent_correct=51,count=64)
    s["raw_stages"]["verify"]["quality"]=producer.old.gates(51,parent=51)
    assert s["raw_stages"]["train"]["quality"]!=s["raw_stages"]["verify"]["quality"]
    assert Record.from_evidence(value).accepted_compressed_layers==8


def test_complete_all36_requires_every_contiguous_passing_verified_stage():
    value=evidence([stage(cut) for cut in range(28,-1,-1)],
        dict(status="all_blocks_recovered",last_accepted_cut=0,compressed_blocks=36,confirmed=False))
    r=Record.from_evidence(value)
    assert r.primary_metric==36 and r.terminal and r.classification=="all36_recovered_on_reused_development"
    assert r.accepted_storage["frozen_native_prefix_bytes"]==0 and not r.confirmed
    value["stages"].pop(1)
    with pytest.raises(ValueError,match="contiguous"):Record.from_evidence(value)


def test_quality_failure_is_completed_and_keeps_last_accepted_parent():
    value=evidence([stage(selected=47)],dict(status="quality_stop",last_accepted_cut=29,
        compressed_blocks=7,failed_cut=28,confirmed=False))
    r=Record.from_evidence(value)
    assert r.classification=="progressive_quality_stop" and r.terminal and r.primary_metric==7
    assert r.stages[0]["selected"]["correct"]==47 and not r.stages[0]["quality"]["quality_pass"]
    value["stages"].append(stage(27,parent=47))
    with pytest.raises(ValueError,match="advance"):Record.from_evidence(value)


@pytest.mark.parametrize("kind",["resource_failure","execution_failure","in_progress"])
def test_resource_execution_and_running_states_do_not_become_quality_failures(kind):
    pending=dict(status=kind,plan=dict(cut=27,parent_cut=28,parent_correct=51),phase="capture",error="example")
    r=Record.from_evidence(evidence([stage(),pending]))
    assert r.primary_metric==8 and r.accepted_correct==51 and len(r.stages)==1
    assert r.terminal is (kind!="in_progress") and "quality" not in r.classification


@pytest.mark.parametrize("mutation,match",[
    (lambda v:v["stages"][0]["raw_stages"]["train"].update(selected_epoch=0),"selection"),
    (lambda v:v["stages"][0]["raw_stages"]["train"].update(updates=383),"12epoch"),
    (lambda v:v["stages"][0]["raw_stages"]["pilot"].update(norm_fp32_master_changed=False),"pilot"),
    (lambda v:v["stages"][0]["raw_stages"]["verify"].update(passed=False),"Fresh"),
    (lambda v:v["stages"][0]["raw_stages"]["train"].update(quality_pass=False),"budgets"),
    (lambda v:v["stages"][0]["plan"].update(calibration_seed=20861),"seed"),
    (lambda v:v.update(confirmed=True),"Declared"),
    (lambda v:v["stages"][0]["raw_stages"]["verify"]["storage"].update(full_model_tensor_bytes=1),"arithmetic"),
])
def test_rejects_selection_budget_precision_and_state_tampering(mutation,match):
    value=evidence([stage()]);mutation(value)
    with pytest.raises(ValueError,match=match):Record.from_evidence(value)


def test_argument_cannot_relabel_partial_prefix_as_all36():
    value=evidence([stage()],dict(status="all_blocks_recovered",last_accepted_cut=28,compressed_blocks=8,confirmed=False))
    with pytest.raises(ValueError,match="Only verified passing cut0"):Record.from_evidence(value)


def test_actual_record_logits_and_labels_determine_quality():
    value=evidence([stage()]);row=value["stages"][0]["epochs"][3]["records"][0]
    row["prediction"]=(row["answer"]+1)%len(row["logits"])
    with pytest.raises(ValueError,match="argmax"):Record.from_evidence(value)


def test_optional_raw_control_rows_enable_recomputation_and_cannot_be_forged():
    value=evidence([stage()]);controls=value["stages"][0]["raw_stages"]["capture"]["controls"]
    for name,n in (("parent",51),("seam",51),("native",48)):controls[name]["records"]=rows(n)
    assert Record.from_evidence(value).stages[0]["control_rows_independently_recomputed"] is True
    controls["seam"]["records"]=rows(50)
    with pytest.raises(ValueError,match="Saved control rows"):Record.from_evidence(value)


def test_packed_targets_are_distinct_from_unchanged_differentiable_calibration_cache():
    value=evidence([stage()]);capture=value["stages"][0]["raw_stages"]["capture"]
    capture["teacher_cache_diagnostic"].update(passed=True,logits_pass=True)
    capture["teacher_cache_logit_gate_applied"]=True
    capture["differentiable_teacher_cache_diagnostic"]=dict(count=64,correct=50,max_abs_logit_delta=.25,predictions_exact=False)
    capture["packed_target_receipt"]=dict(status="completed",shape=[576,4,151936],microbatch=4,training_count=512,
        development_count=64,fixed_state_unchanged=True,calibration_jacobians_changed=False,
        path="/separate/packed_teacher_logits.npy",sha256="a"*64)
    result=Record.from_evidence(value).stages[0]
    assert result["recovery_target_execution"]=="packed_inference" and result["differentiable_teacher_correct"]==50
    assert result["parent_correct"]==51 and result["selected"]["correct"]==51
    capture["packed_target_receipt"]["calibration_jacobians_changed"]=True
    with pytest.raises(ValueError,match="without changing VJP"):Record.from_evidence(value)


def test_snapshot_cannot_enter_registry(monkeypatch,tmp_path):
    value=evidence([stage()]);r=Record.from_evidence(value)
    monkeypatch.setattr(final,"expected",lambda root:(dict(value,snapshot=True),r))
    with pytest.raises(ValueError,match="cannot be registered"):
        final.store(tmp_path,tmp_path/"registry.json",tmp_path/"report.md")


def test_expected_checks_analysis_and_archived_source_lineage(monkeypatch,tmp_path):
    source=tmp_path/"source.py";source.write_text("source")
    artifact=tmp_path/"result.json";artifact.write_text("result")
    archive=tmp_path/"archive";archive.mkdir();(archive/"source.py").write_text("source")
    value=evidence();r=Record.from_evidence(value)
    value.update(record=r.to_dict(),artifact_sha256={str(artifact):final.sha(artifact)},authenticated_large_artifacts={},
        analysis_source_sha256={str(source):final.sha(source)},analysis_archive="archive",snapshot=True)
    (tmp_path/final.ANALYSIS).write_text(json.dumps(value));monkeypatch.setattr(final,"ROOT",tmp_path)
    assert final.expected(tmp_path)[1].primary_metric==7
    (archive/"source.py").write_text("tampered")
    with pytest.raises(ValueError,match="archived"):final.expected(tmp_path)


def test_report_and_metadata_keep_accuracy_distinct_from_compressed_count(monkeypatch,tmp_path):
    value=evidence([stage()],dict(status="requested_prefix_completed",last_accepted_cut=28,compressed_blocks=8,confirmed=False))
    r=Record.from_evidence(value);monkeypatch.setattr(final,"expected",lambda *args:(value,r))
    (tmp_path/final.ANALYSIS).write_text("analysis")
    result=final.report(tmp_path,tmp_path/"report.md")
    text=Path(result["report"]).read_text();meta=final.metadata(tmp_path,r)
    assert "8/36" in text and "51/64" in text and "No untouched confirmation" in text
    assert meta["primary_metric"]==8 and meta["primary_metric_unit"]=="layers" and meta["accuracy"]==51/64
    assert meta["confirmed"] is False and meta["full36_recovered"] is False
