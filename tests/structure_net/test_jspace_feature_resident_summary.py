"""Reporting must never promote an incomplete or merely observed candidate."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.structure_net import jspace_feature_resident_summary as summary


def put(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(path, value):
    value=dict(value);value["fingerprint"]=summary.fingerprint(value);put(path,value)
    return value


def report(correct, ce=1.5, kl=.6):
    rows=[dict(identity=f"dev-{i}",family=("arc","boolq","hellaswag","winogrande")[i//16],
        answer=0,prediction=0 if i<correct else 1,logits=[2.,0.] if i<correct else [0.,2.]) for i in range(64)]
    metrics=dict(count=64,correct=correct,accuracy=correct/64,ce=ce,original_kl=kl)
    return dict(metrics=metrics,records=rows)


def phase(directory, name, fp, values, binaries=None):
    path=directory/name;artifacts={k:put(path/k,v) for k,v in values.items()}
    for k,v in (binaries or {}).items():
        (path/k).write_bytes(v);artifacts[k]=hashlib.sha256(v).hexdigest()
    put(path/"manifest.json",dict(status="completed",fingerprint=fp,artifacts=artifacts))
    return artifacts


def train(directory, fp, raw=46, best=48):
    values={};history=[]
    for epoch in range(-1,13):
        value=report(raw,1.5) if epoch==-1 else report(raw,1.4) if epoch==0 else report(best if epoch==8 else raw-2,1.7)
        metrics=value["metrics"];ce=metrics["ce"]<=1.5+1e-8
        name=f"epoch_{epoch:02d}_development.json";values[name]=value
        digest=hashlib.sha256(json.dumps(value,allow_nan=False).encode()).hexdigest()
        history.append(dict(epoch=epoch,metrics=metrics,development_sha256=digest,
            eligibility=dict(eligible=ce,finite=True,ce_pass=ce,original_kl_pass=True,tolerance=1e-8)))
    selected_sha=hashlib.sha256(b"scientific").hexdigest();best_sha=hashlib.sha256(b"best-task").hexdigest()
    trained=dict(status="completed",completed_epochs=12,updates=384,strict_reload_exact=True,fixed_state_unchanged=True,
        history=history,initial=history[0]["metrics"],warm_initial=history[1]["metrics"],selected=history[1]["metrics"],
        selected_epoch=0,selected_checkpoint_sha256=selected_sha,quality_pass=raw>=45)
    values["result.json"]=trained
    values["diagnostic_best_task.json"]=dict(epoch=8,metrics=history[9]["metrics"],checkpoint_sha256=best_sha,
        diagnostic_only=True,automatically_adopted=False)
    artifacts=phase(directory,"train",fp,values,{"selected.pt":b"scientific","diagnostic_best_task.pt":b"best-task"})
    fresh=dict(status="completed",passed=True,fixed_state_unchanged=True,**values["epoch_00_development.json"],selected_checkpoint_sha256=selected_sha)
    return trained,values,artifacts,fresh


def acceptance(directory, values, artifacts, *, seed=False):
    candidate=dict(kind="feature_best_task" if seed else "best_task",epoch=8,
        checkpoint=str(directory/"train/diagnostic_best_task.pt"),checkpoint_sha256=artifacts["diagnostic_best_task.pt"],
        report=str(directory/"train/epoch_08_development.json"),report_sha256=artifacts["epoch_08_development.json"],
        metrics=values["epoch_08_development.json"]["metrics"])
    passed=candidate["metrics"]["correct"]>=45
    return dict(candidate=candidate,accepted=passed,quality=dict(passed=passed,native_correct=48,minimum_correct=45,
        count=64,maximum_drop=.05,ce_kl_increase_is_veto=False))


@pytest.fixture
def campaign(tmp_path):
    root=tmp_path/"campaign";feature=tmp_path/"feature";proof=tmp_path/"proof"
    fp=plan(feature/"plan.json",dict(schema="feature"))["fingerprint"]
    _,values,artifacts,fresh=train(feature,fp)
    phase(feature,"verify",fp,{"result.json":fresh})
    chosen=acceptance(feature,values,artifacts,seed=True)
    pp=plan(proof/"plan.json",dict(candidate=chosen["candidate"]))
    pr=dict(status="completed",passed=True,strict_state_load=True,fixed_state_unchanged=True,
        checkpoint_sha256=chosen["candidate"]["checkpoint_sha256"],**values["epoch_08_development.json"])
    phase(proof,"run",pp["fingerprint"],{"result.json":pr})
    inputs={str(p.resolve()):summary.sha(p) for p in (feature/"plan.json",feature/"train/manifest.json",
        feature/"verify/manifest.json",proof/"plan.json")}
    p=plan(root/"plan.json",dict(schema=summary.PRODUCER_SCHEMA,seed_cut=23,phases=list(summary.PHASES),
        source_sha256={},input_sha256=inputs,feature_root=str(feature),seed_fingerprint=fp,seed_correct=48,
        seed_checkpoint_sha256=artifacts["diagnostic_best_task.pt"],seed_selection=chosen,
        seed_verification=dict(root=str(proof),manifest_sha256=summary.sha(proof/"run/manifest.json"))))
    return root,p


def add_stage(campaign, *, complete=True, raw=44, best=47):
    root,parent=campaign;directory=root/"stages/block_22"
    stage=plan(directory/"plan.json",dict(root_fingerprint=parent["fingerprint"],cut=22,parent_cut=23,
        parent_correct=48,parent_checkpoint_sha256=parent["seed_checkpoint_sha256"]))
    for name in summary.PHASES[:-2]:
        phase(directory,name,stage["fingerprint"],{"result.json":dict(status="completed")})
    _,values,artifacts,fresh=train(directory,stage["fingerprint"],raw,best)
    accepted=acceptance(directory,values,artifacts)
    reference=values["epoch_08_development.json"]
    accepted.update(strict_state_load=True,fixed_state_unchanged=True,exact_packed_replay=True,**reference)
    fresh["acceptance"]=accepted
    if complete:
        phase(directory,"verify",stage["fingerprint"],{"result.json":fresh,"acceptance.json":accepted,
            "accepted_development.json":reference},{"accepted.pt":b"best-task"} if best>=45 else {})
    else:
        put(directory/"verify/result.json",fresh)  # Provisional result is not a completed phase.
    return directory


def mutate_json(path, edit, *, rehash_manifest=False):
    value=json.loads(path.read_text());edit(value);put(path,value)
    if rehash_manifest:
        mp=path.parent/"manifest.json";m=json.loads(mp.read_text());m["artifacts"][path.name]=summary.sha(path);put(mp,m)


def test_scientific_quality_failure_does_not_veto_separate_verified_task_acceptance(campaign):
    add_stage(campaign)
    result=summary.summarize(campaign[0]);seed,row=result["rows"]
    assert (seed["compressed_blocks"],seed["scientific_correct"],seed["accepted_correct"])==(13,46,48)
    assert (row["raw_correct"],row["warm_correct"],row["scientific_correct"],row["best_task_correct"],row["accepted_correct"])==(44,44,44,47,47)
    assert row["scientific_checkpoint_sha256"]!=row["accepted_checkpoint_sha256"]
    assert result["accepted_blocks"]==14 and result["latest_accepted_correct"]==47
    assert result["minimum_correct"]==45 and not result["confirmed"]


def test_incomplete_fresh_verification_never_promotes_provisional_weights(campaign):
    directory=add_stage(campaign,complete=False)
    result=summary.summarize(campaign[0])
    assert result["accepted_blocks"]==13 and len(result["rows"])==1
    assert result["pending_stage"]["next_incomplete_phase"]=="verify"
    assert str((directory/"verify/result.json").resolve()) not in result["input_sha256"]


def test_rejected_completed_candidate_does_not_grow_accepted_depth(campaign):
    add_stage(campaign,raw=42,best=44)
    result=summary.summarize(campaign[0])
    assert result["status"]=="task_quality_stop" and result["accepted_blocks"]==13
    assert result["rows"][-1]["accepted_correct"]==44 and not result["rows"][-1]["accepted"]


@pytest.mark.parametrize("mutation",["manifest","accepted_hash","report","eligibility","parent","partial"])
def test_mismatched_or_inconsistent_stage_evidence_rejected(campaign,mutation):
    directory=add_stage(campaign)
    if mutation=="manifest":
        mutate_json(directory/"verify/manifest.json",lambda v:v.update(fingerprint="wrong"))
    elif mutation=="accepted_hash":
        mutate_json(directory/"verify/manifest.json",lambda v:v["artifacts"].update({"accepted.pt":"0"*64}))
    elif mutation=="report":
        mutate_json(directory/"verify/accepted_development.json",lambda v:v["metrics"].update(correct=60))
    elif mutation=="eligibility":
        mutate_json(directory/"train/result.json",lambda v:v["history"][9]["eligibility"].update(eligible=True),rehash_manifest=True)
    elif mutation=="parent":
        p=json.loads((directory/"plan.json").read_text());p["parent_checkpoint_sha256"]="0"*64;plan(directory/"plan.json",p)
    else:
        (directory/"prealign/manifest.json").unlink()
    with pytest.raises(ValueError):summary.summarize(campaign[0])


def test_only_small_json_and_source_are_opened_no_weights_or_heldout(campaign,monkeypatch,tmp_path):
    add_stage(campaign);original=Path.open
    def guarded(path,*args,**kwargs):
        assert path.suffix not in (".pt",".npy") and "heldout" not in str(path)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    output=tmp_path/"summary"
    receipt=summary.generate(campaign[0],output)
    assert receipt["accepted_blocks"]==14
    assert json.loads((output/"summary.json").read_text())["heldout_measured"] is False
    assert (output/summary.__file__.split("/")[-1]).is_file()
    with pytest.raises(ValueError):summary.generate(campaign[0],output)


def test_fresh_seed_proof_cannot_be_substituted(campaign):
    root,p=campaign;proof=Path(p["seed_verification"]["root"])
    mutate_json(proof/"run/result.json",lambda v:v.update(passed=False))
    with pytest.raises(ValueError):summary.summarize(root)
