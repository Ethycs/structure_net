from copy import deepcopy

import pytest

from neural_architecture_lab.jspace_residual_diagnostics_record import (
    ARMS,CLASSIFICATION,STUDY_ID,VARIANTS,JSpaceResidualDiagnosticsRecord)


def fixture():
    sources,summary = {},{}
    native=[]
    families=("arc","boolq","hellaswag","winogrande")
    for i in range(64):
        prediction=0 if i<50 else 1
        logits=[0.,0.,0.,0.] if i//16 in (0,2) else [0.,0.]
        logits[prediction]=2.
        native.append(dict(identity=str(i),family=families[i//16],answer=0,prediction=prediction,logits=logits))
    for index,name in enumerate(ARMS):
        rows=deepcopy(native)
        for i in range(dict(native=0,full_control=0,half_head=5,half_rank=23,mean_rank=20)[name]):
            rows[i]["prediction"]=1;rows[i]["logits"]=[0.,2.]+rows[i]["logits"][2:]
        count=sum(r["prediction"]==r["answer"] for r in rows)
        changed=sum(a["prediction"]!=b["prediction"] for a,b in zip(native,rows))
        fp=str(index)*64
        sources[name]=dict(status="completed",records=rows,count=64,correct=count,accuracy=count/64,fingerprint=fp)
        if name!="native":sources[name]["variant"]=VARIANTS[name]
        summary[name]=dict(correct=count,count=64,wins_against_native=0,losses_against_native=50-count,
            changed_predictions=changed,max_abs_label_logit_delta=2. if changed else 0.,fingerprint=fp,by_family=None)
    aggregate=dict(status="completed",confirmed=False,study_id="JSPACE-QWEN-ACCURACY-RECOVERY-01",
                   classification=CLASSIFICATION,arms=summary)
    return aggregate,sources


def test_record_is_distinct_paired_development_diagnostic_without_confirmation():
    aggregate,sources=fixture()
    record=JSpaceResidualDiagnosticsRecord.from_evidence(aggregate,sources,"a"*64)
    assert record.study_id==STUDY_ID and record.parent_study_id!=record.study_id
    assert not record.confirmed and record.independent_confirmations==0
    assert record.primary_metric==27/64 and len(record.arms)==5
    assert record.arms[3].losses==23 and record.arms[3].gains==0
    assert "No held-out" in record.scope


@pytest.mark.parametrize("error",["pairing","label","nan","prediction","count","paired_count","confirmation","missing"])
def test_invalid_or_inflated_evidence_is_rejected(error):
    aggregate,sources=fixture()
    row=sources["half_rank"]["records"][0]
    if error=="pairing":row["identity"]="other"
    elif error=="label":row["answer"]=1
    elif error=="nan":row["logits"][0]=float("nan")
    elif error=="prediction":row["prediction"]=0
    elif error=="count":aggregate["arms"]["half_rank"]["correct"]+=1
    elif error=="paired_count":aggregate["arms"]["half_rank"]["wins_against_native"]+=1
    elif error=="confirmation":aggregate["confirmed"]=True
    elif error=="missing":del sources["mean_rank"]
    with pytest.raises(ValueError):JSpaceResidualDiagnosticsRecord.from_evidence(aggregate,sources,"a"*64)
