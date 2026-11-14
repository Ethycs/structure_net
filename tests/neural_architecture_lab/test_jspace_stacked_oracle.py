from copy import deepcopy
import json

import pytest

from neural_architecture_lab.jspace_stacked_oracle_record import JSpaceStackedOracleRecord,RECURRENCE
from experiments.neural_architecture_lab import finalize_jspace_stacked_oracle as finalizer


def rows(correct):
    families=("arc","boolq","hellaswag","winogrande")
    return [dict(identity=str(i),family=families[i//16],answer=0,prediction=int(i>=correct),
        logits=[1.,0.] if i<correct else [0.,1.]) for i in range(64)]


def evidence():
    native=rows(50)
    return dict(status="completed",confirmed=False,
        plan=dict(layers=36,shape=[100,2048],basis_paths=[f"basis{i}" for i in range(36)],recurrence=RECURRENCE,
            fingerprint="oracle",native_correct=50,confirmed=False,
            cases=[dict(identity=r["identity"],family=r["family"],answer=r["answer"],split="development") for r in native]),
        evaluation=dict(status="completed",fingerprint="oracle",confirmed=False,layers=36,shape=[100,2048],
            quality_reference_is_diagnostic_only=True,native=dict(records=native,correct=50,count=64),
            oracle=dict(records=deepcopy(native),correct=50,count=64),
            paired=dict(overall=dict(agreement=1.,wins=0,losses=0)),
            native_control=dict(passed=True,predictions_exact=True,logits_pass=True,logit_max_abs_delta=0.),
            trajectory=[dict(boundary=i,valid_token_relative_rms=i*.006,answer_relative_rms=i*.004) for i in range(37)]),
        native_reference=dict(records=deepcopy(native)),separate_student=dict(arm="all_compact",records=rows(27)))


def test_primary_metric_is_teacher_guided_agreement_not_learned_accuracy():
    r=JSpaceStackedOracleRecord.from_evidence(evidence())
    assert r.primary_metric==1. and r.teacher_guided["correct"]==50 and r.native["correct"]==50
    assert r.teacher_guided["accuracy"]==50/64 and r.separate_student["correct"]==27
    assert r.classification=="teacher_guided_native_decisions_preserved"
    assert not r.confirmed and not r.learned_model and not r.deployable and not r.task_accuracy_upper_bound
    assert r.final_relative_rms["valid_token_relative_rms"]>0


def test_equal_task_accuracy_does_not_imply_native_decision_agreement():
    value=evidence();oracle=value["evaluation"]["oracle"]["records"]
    for index in (0,50):
        oracle[index]["prediction"]=1-oracle[index]["prediction"]
        oracle[index]["logits"]=list(reversed(oracle[index]["logits"]))
    value["evaluation"]["paired"]["overall"]=dict(agreement=62/64,wins=1,losses=1)
    r=JSpaceStackedOracleRecord.from_evidence(value)
    assert r.teacher_guided["correct"]==50 and r.primary_metric==62/64
    assert r.classification=="teacher_guided_decision_mismatch"


@pytest.mark.parametrize("problem",["basis","recurrence","trajectory","nan","native_control","identity","confirmation","diagnostic_gate","summary","pair"])
def test_changed_or_inflated_evidence_rejected(problem):
    value=evidence()
    if problem=="basis":value["plan"]["basis_paths"].pop()
    elif problem=="recurrence":value["plan"]["recurrence"]="native_update_only"
    elif problem=="trajectory":value["evaluation"]["trajectory"].pop()
    elif problem=="nan":value["evaluation"]["trajectory"][-1]["answer_relative_rms"]=float("nan")
    elif problem=="native_control":value["evaluation"]["native_control"]["passed"]=False
    elif problem=="identity":value["evaluation"]["oracle"]["records"][0]["identity"]="different"
    elif problem=="confirmation":value["confirmed"]=True
    elif problem=="diagnostic_gate":value["evaluation"]["quality_reference_is_diagnostic_only"]=False
    elif problem=="summary":value["evaluation"]["oracle"]["correct"]=64
    elif problem=="pair":value["evaluation"]["paired"]["overall"]["agreement"]=.99
    with pytest.raises(ValueError):JSpaceStackedOracleRecord.from_evidence(value)


def test_metadata_avoids_learned_model_accuracy_claim(tmp_path):
    (tmp_path/finalizer.ANALYSIS).write_text("analysis")
    m=finalizer.metadata(tmp_path,JSpaceStackedOracleRecord.from_evidence(evidence()))
    assert m["primary_metric_name"]=="teacher_guided_native_decision_agreement"
    assert m["primary_metric_unit"]=="fraction" and m["primary_metric"]==1.
    assert m["teacher_guided_correct"]==50 and "accuracy" not in m
    assert m["learned_model"] is False and m["deployable"] is False


def test_previously_authenticated_bank_identity_change_stops_closeout(tmp_path):
    value=evidence();value["record"]=JSpaceStackedOracleRecord.from_evidence(value).to_dict()
    bank=tmp_path/"boundary.npy";bank.write_bytes(b"original-bank")
    value.update(artifact_sha256={},analysis_source_sha256={},
        prior_authenticated_inputs={str(bank):dict(sha256=finalizer.sha(bank),identity=finalizer.identity(bank))})
    (tmp_path/finalizer.ANALYSIS).write_text(json.dumps(value))
    finalizer.expected(tmp_path);bank.write_bytes(b"changed-bank")
    with pytest.raises(ValueError,match="input changed"):finalizer.expected(tmp_path)
