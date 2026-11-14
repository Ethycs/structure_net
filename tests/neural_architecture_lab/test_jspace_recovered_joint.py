from copy import deepcopy
import json

import pytest

from neural_architecture_lab.jspace_recovered_joint_record import JSpaceRecoveredJointRecord,CONFIG,classify
from experiments.neural_architecture_lab import finalize_jspace_recovered_joint as finalizer


def records(correct):
    families = ("arc","boolq","hellaswag","winogrande")
    return [dict(identity=str(i),family=families[i//16],answer=0,prediction=int(i>=correct),
        logits=[1.,0.] if i<correct else [0.,1.],teacher_prediction=int(i>=50)) for i in range(64)]


def evidence(counts=(27,31,36,40)):
    native = records(50);epochs=[];history=[]
    for epoch,correct in enumerate(counts):
        rows = records(correct);kl=.7-epoch*.1;ce=2.-epoch*.2
        metrics = dict(correct=correct,count=64,accuracy=correct/64,full_kl=kl,choice_kl=kl*2,choice_ce=ce,
            weighted_full_kl=.8*kl,weighted_choice_kl=2*kl*2,weighted_choice_ce=.3*ce,
            loss=.8*kl+4*kl+.3*ce,teacher_accuracy=50/64,
            teacher_agreement=sum(a["prediction"] == b["prediction"] for a,b in zip(rows,native))/64,
            choice_margin=-1+epoch*.1,valid_label_probability=.95,answer_full_vocab_ce=ce+.1)
        history.append(dict(epoch=epoch,metrics=metrics,checkpoint_manifest_sha256="checkpoint"+str(epoch)))
        epochs.append(dict(epoch=epoch,metrics=metrics,records=rows))
    winner=max(history,key=lambda row:(row["metrics"]["correct"],-row["metrics"]["choice_ce"]))
    chosen=winner["epoch"];passed=counts[chosen]>=47
    return dict(status="completed",confirmed=False,
        plan=dict(config=deepcopy(CONFIG),initial_correct=27,native_correct=50,minimum_correct=47,confirmed=False,
            fingerprint="campaign",partition={"3060":list(range(24)),"2060_SUPER":list(range(24,36))}),
        training=dict(status="completed",confirmed=False,systems_only=False,updates=96,strict_reload_exact=True,
            history=history,initial=history[0]["metrics"],selected=winner["metrics"],selected_epoch=chosen,
            selected_manifest_sha256=winner["checkpoint_manifest_sha256"],initializer_selected=chosen==0,
            minimum_correct=47,native_correct=50,quality_pass=passed),
        verification=dict(status="completed",confirmed=False,fingerprint="campaign",selected_epoch=chosen,
            selected_manifest_sha256=winner["checkpoint_manifest_sha256"],passed=True,strict_state_load=True,
            predictions_exact=True,logits_pass=True,metrics_pass=True,fixed_state_unchanged=True,
            records=epochs[chosen]["records"],training_quality_pass=passed),
        pilot=dict(systems_only=True,updates=1,initial_control=dict(passed=True,correct=27),fixed_state_unchanged=True,
            parameter_changes=dict(changed_blocks=list(range(36)),unchanged_blocks=[])),
        teacher=dict(status="completed",passed=True,dtype="float32",shape=[576,2,151936],slots=[1,3],
            records=native,correct=50,count=64,control=dict(passed=True,predictions_exact=True,logits_pass=True)),
        epochs=epochs,storage=dict(all_selected_fp32=True,selected_blocks=36,tied_embedding_readout_counted_once=True,
            model_parameters=100,unpacked_persistent_tensor_bytes=500))


@pytest.mark.parametrize("correct,expected",[(0,"no_recovery_over_warm_start"),(27,"no_recovery_over_warm_start"),
    (28,"partial_recovery_below_gate"),(46,"partial_recovery_below_gate"),(47,"development_gate_passed"),(64,"development_gate_passed")])
def test_prospective_classification_boundaries(correct,expected):
    assert classify(correct)==expected


def test_actual_full_model_accuracy_not_fraction_of_layer_passes():
    record=JSpaceRecoveredJointRecord.from_evidence(evidence())
    assert record.primary_metric==40/64 and record.classification=="partial_recovery_below_gate"
    assert record.initial["correct"]==27 and record.native["correct"]==50 and record.updates==96
    assert record.selected["paired_to_initial"]["gains"]==13
    assert record.selected["paired_to_native"]["losses"]==10
    assert not record.quality_pass and not record.confirmed and not record.speed_measured and not record.packing_validated
    assert [r["epoch"] for r in record.epochs]==[0,1,2,3]


def test_initializer_can_win_without_being_called_recovery():
    record=JSpaceRecoveredJointRecord.from_evidence(evidence((27,26,25,26)))
    assert record.selected_epoch==0 and record.initializer_selected
    assert record.primary_metric==27/64 and record.classification=="no_recovery_over_warm_start"


def test_development_pass_never_implies_confirmation():
    record=JSpaceRecoveredJointRecord.from_evidence(evidence((27,39,45,48)))
    assert record.quality_pass and record.classification=="development_gate_passed"
    assert record.primary_metric==.75 and not record.confirmed and record.independent_confirmations==0


@pytest.mark.parametrize("problem",["schedule","updates","epoch","selection","checkpoint","fresh","fixed","pilot","teacher_precision",
    "teacher_decisions","labels","weights","gate","confirmed","storage"])
def test_incomplete_or_changed_evidence_rejected(problem):
    value=evidence()
    if problem=="schedule":value["plan"]["config"]["epochs"]=4
    elif problem=="updates":value["training"]["updates"]=95
    elif problem=="epoch":value["epochs"].pop()
    elif problem=="selection":value["training"]["selected_epoch"]=1
    elif problem=="checkpoint":value["verification"]["selected_manifest_sha256"]="wrong"
    elif problem=="fresh":value["verification"]["passed"]=False
    elif problem=="fixed":value["verification"]["fixed_state_unchanged"]=False
    elif problem=="pilot":value["pilot"]["parameter_changes"]["changed_blocks"].pop()
    elif problem=="teacher_precision":value["teacher"]["dtype"]="bfloat16"
    elif problem=="teacher_decisions":value["epochs"][1]["records"][0]["teacher_prediction"]=1
    elif problem=="labels":value["epochs"][1]["records"][0]["identity"]="other"
    elif problem=="weights":value["epochs"][1]["metrics"]["weighted_choice_kl"]+=1
    elif problem=="gate":value["training"]["quality_pass"]=True
    elif problem=="confirmed":value["confirmed"]=True
    elif problem=="storage":value["storage"]["all_selected_fp32"]=False
    with pytest.raises(ValueError):JSpaceRecoveredJointRecord.from_evidence(value)


def test_registry_uses_accuracy_fraction_and_exact_scope(tmp_path):
    (tmp_path/finalizer.ANALYSIS).write_text("analysis")
    metadata=finalizer.metadata(tmp_path,JSpaceRecoveredJointRecord.from_evidence(evidence()))
    assert metadata["primary_metric"]==40/64 and metadata["correct"]==40 and metadata["count"]==64
    assert metadata["primary_metric_name"]=="selected_full_model_development_accuracy"
    assert metadata["trained_blocks"]==36 and metadata["confirmed"] is False


@pytest.mark.parametrize("which",["receipt","weights"])
def test_authentication_rejects_postanalysis_changes(tmp_path,which):
    value=evidence();value["record"]=JSpaceRecoveredJointRecord.from_evidence(value).to_dict()
    receipt=tmp_path/"verification.json";receipt.write_text("verified")
    weights=tmp_path/"block.pt";weights.write_bytes(b"selected")
    value.update(artifact_sha256={str(receipt):finalizer.sha(receipt)},analysis_source_sha256={},
        authenticated_large_artifacts={str(weights):dict(sha256=finalizer.sha(weights),identity=finalizer.identity(weights))})
    (tmp_path/finalizer.ANALYSIS).write_text(json.dumps(value))
    finalizer.expected(tmp_path)
    if which=="receipt":receipt.write_text("failed")
    else:weights.write_bytes(b"changed")
    with pytest.raises(ValueError):finalizer.expected(tmp_path)
