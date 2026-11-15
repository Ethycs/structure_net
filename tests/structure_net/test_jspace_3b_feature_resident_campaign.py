"""Task acceptance stays distinct from immutable cautious selection."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.structure_net import jspace_3b_task_acceptance as accept
from experiments.structure_net import jspace_3b_feature_resident_campaign as study
from test_jspace_3b_staged_campaign import quality, write


def candidate(kind, correct, ce=1., kl=.2):
    return dict(kind=kind, metrics=quality(correct,ce,kl)["metrics"])


def test_accuracy_budget_does_not_reintroduce_ce_or_kl_veto():
    choices=[candidate("baseline",46,1.),candidate("feature_selected",44,.8),
             candidate("feature_best_task",47,1.7,2.)]
    value=accept.choose(choices)
    assert value["accepted"] and value["candidate"]["kind"]=="feature_best_task"
    assert not accept.native_gate(choices[1]["metrics"])["passed"]
    assert value["quality"]["ce_kl_increase_is_veto"] is False
    assert value["quality"]["minimum_correct"]==45


def test_seed_exact_tie_prefers_baseline_then_scientific_selection():
    choices=[candidate("baseline",47),candidate("feature_selected",47),candidate("feature_best_task",47)]
    assert accept.choose(choices)["candidate"]["kind"]=="baseline"
    choices[0]["metrics"]["ce"]=1.1
    assert accept.choose(choices)["candidate"]["kind"]=="feature_selected"
    choices[2]["metrics"]["ce"]=.9
    assert accept.choose(choices)["candidate"]["kind"]=="feature_best_task"


@pytest.mark.parametrize("field,value",[("ce",float("nan")),("original_kl",float("inf")),("count",63),("correct",True)])
def test_task_gate_rejects_nonfinite_or_wrong_population(field,value):
    item=candidate("baseline",46);item["metrics"][field]=value
    with pytest.raises(ValueError):accept.choose([item])


def checkpoint_fixture(directory,selected_correct=44,best_correct=47):
    directory.mkdir(parents=True)
    history=[]
    for epoch,report in [(0,quality(selected_correct,.8,.1)),(1,quality(best_correct,1.7,2.))]:
        path=directory/f"epoch_{epoch:02d}_development.json";write(path,report)
        history.append(dict(epoch=epoch,metrics=report["metrics"],development_sha256=accept.base.sha(path)))
    (directory/"selected.pt").write_bytes(b"original scientific checkpoint")
    (directory/"diagnostic_best_task.pt").write_bytes(b"separate best-task checkpoint")
    result=dict(selected_epoch=0,selected=history[0]["metrics"],history=history,
        selected_checkpoint_sha256=accept.base.sha(directory/"selected.pt"),quality_pass=False)
    write(directory/"result.json",result)
    write(directory/"diagnostic_best_task.json",dict(epoch=1,metrics=history[1]["metrics"],
        checkpoint_sha256=accept.base.sha(directory/"diagnostic_best_task.pt"),
        diagnostic_only=True,automatically_adopted=False))
    return result


def test_best_task_descriptor_is_saved_authenticated_and_never_relabels_science(tmp_path):
    trained=checkpoint_fixture(tmp_path/"train")
    before={p.name:accept.base.sha(p) for p in (tmp_path/"train").iterdir()}
    selected=accept.descriptor("feature_selected",tmp_path/"train",trained)
    best=accept.descriptor("feature_best_task",tmp_path/"train",trained,diagnostic=True)
    assert selected["checkpoint"]!=best["checkpoint"]
    assert selected["checkpoint_sha256"]!=best["checkpoint_sha256"]
    assert not trained["quality_pass"] and accept.choose([selected,best])["accepted"]
    assert before=={p.name:accept.base.sha(p) for p in (tmp_path/"train").iterdir()}


@pytest.mark.parametrize("change",["missing_weights","changed_weights","changed_rows","not_actual_winner"])
def test_best_task_admission_rejects_unsaved_or_changed_candidate(tmp_path,change):
    trained=checkpoint_fixture(tmp_path/"train");directory=tmp_path/"train"
    if change=="missing_weights":(directory/"diagnostic_best_task.pt").unlink()
    elif change=="changed_weights":(directory/"diagnostic_best_task.pt").write_bytes(b"changed")
    elif change=="changed_rows":
        path=directory/"epoch_01_development.json";report=accept.read(path)
        report["records"][0]["identity"]="another";write(path,report)
    else:
        path=directory/"diagnostic_best_task.json";value=accept.read(path);value["epoch"]=0;write(path,value)
    with pytest.raises((ValueError,FileNotFoundError)):
        accept.descriptor("feature_best_task",directory,trained,diagnostic=True)


def seed_verification(tmp_path,monkeypatch):
    directory=tmp_path/"training";trained=checkpoint_fixture(directory)
    best=accept.descriptor("feature_best_task",directory,trained,diagnostic=True)
    root=tmp_path/"fresh";root.mkdir();write(root/"reference.json",accept.read(best["report"]))
    plan=dict(candidate=best,fingerprint="verification")
    monkeypatch.setattr(accept,"verify_plan",lambda path:(plan,{}))
    path=root/"run";path.mkdir()
    report=accept.read(root/"reference.json")
    result=dict(status="completed",passed=True,strict_state_load=True,fixed_state_unchanged=True,
        checkpoint_sha256=best["checkpoint_sha256"],metrics=report["metrics"],records=report["records"])
    write(path/"result.json",result)
    write(path/"manifest.json",dict(status="completed",fingerprint=plan["fingerprint"],
        artifacts={"result.json":accept.base.sha(path/"result.json")}))
    return root,best


@pytest.mark.parametrize("change",["none","incomplete","failed_result","wrong_hash","wrong_prediction"])
def test_accepted_best_task_requires_own_fresh_exact_checkpoint_receipt(tmp_path,monkeypatch,change):
    root,best=seed_verification(tmp_path,monkeypatch);path=root/"run"
    if change=="none":
        assert accept.checked(root,best)["metrics"]["correct"]==47
        return
    if change=="incomplete":
        (path/"manifest.json").unlink()
    else:
        result=accept.read(path/"result.json")
        if change=="failed_result":result["status"]="failed"
        elif change=="wrong_hash":result["checkpoint_sha256"]="scientific selected hash"
        else:result["records"][0]["prediction"]=1
        write(path/"result.json",result);manifest=accept.read(path/"manifest.json")
        manifest["artifacts"]["result.json"]=accept.base.sha(path/"result.json");write(path/"manifest.json",manifest)
    with pytest.raises((ValueError,FileNotFoundError)):accept.checked(root,best)


def test_existing_baseline_verification_does_not_create_a_fake_feature_seed(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    best=candidate("baseline",46)
    monkeypatch.setattr(accept,"seed_candidates",lambda *a:{"selection":{"candidate":best}})
    result=accept.prepare(tmp_path/"new",tmp_path/"baseline",tmp_path/"feature")
    assert result["status"]=="existing_fresh_verification_sufficient" and result["kind"]=="baseline"
    assert not (tmp_path/"new").exists()


def test_seed_candidates_require_completed_feature_proof_and_same_baseline(tmp_path,monkeypatch):
    baseline=tmp_path/"baseline";directory=accept.source.stage_dir(baseline,23)
    old=checkpoint_fixture(directory/"train",selected_correct=46)
    feature=tmp_path/"feature";new=checkpoint_fixture(feature/"train")
    parent={"fingerprint":"original"};stage={"fingerprint":"stage23"}
    plan={"fingerprint":"feature","source_root":str(baseline)}
    monkeypatch.setattr(accept.resident,"SEED_SHA",old["selected_checkpoint_sha256"])
    monkeypatch.setattr(accept.source,"verify_stage",lambda *a:(parent,stage))
    monkeypatch.setattr(accept.source,"checked",lambda path,phase,s:old if phase=="train" else {"passed":True})
    monkeypatch.setattr(accept.feature,"verify_plan",lambda root:(plan,parent,stage))
    calls=[]
    def checked(root,phase,p):
        calls.append(phase)
        return new if phase=="train" else {"passed":True}
    monkeypatch.setattr(accept.feature,"checked",checked)
    evidence=accept.seed_candidates(baseline,feature)
    assert calls==["pilot","train","verify"]
    assert evidence["selection"]["candidate"]["kind"]=="feature_best_task"
    assert [x["kind"] for x in evidence["candidates"]]==["baseline","feature_selected","feature_best_task"]
    def partial(root,phase,p):
        if phase=="verify":raise ValueError("Fresh feature verification incomplete")
        return checked(root,phase,p)
    monkeypatch.setattr(accept.feature,"checked",partial)
    with pytest.raises(ValueError,match="verification incomplete"):
        accept.seed_candidates(baseline,feature)
    monkeypatch.setattr(accept.feature,"checked",checked)
    plan["source_root"]=str(tmp_path/"different")
    with pytest.raises(ValueError,match="same authenticated"):
        accept.seed_candidates(baseline,feature)


def test_best_task_verifier_does_not_touch_cuda_before_reservation(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_full_lease
    monkeypatch.setattr(accept,"verify_plan",lambda root:({},{}))
    def denied(*a):raise RuntimeError("lease denied")
    def forbidden(*a,**k):pytest.fail("GPU/model touched before lease")
    monkeypatch.setattr(jspace_full_lease,"wait_ready",denied)
    monkeypatch.setattr(accept.source,"cap_allocator",forbidden)
    monkeypatch.setattr(accept.source,"make_stack",forbidden)
    with pytest.raises(RuntimeError,match="lease denied"):accept.run(tmp_path)
    assert accept.read(tmp_path/"run/failure.json")["lease_id"] is None


def test_feature_prepare_routes_exact_baseline_tie_to_resident_only(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    choices=[candidate("baseline",46),candidate("feature_selected",46),candidate("feature_best_task",46)]
    monkeypatch.setattr(study,"qualification",lambda root:{"qualified":True})
    monkeypatch.setattr(study.acceptance,"seed_candidates",lambda *a:{"selection":accept.choose(choices)})
    value=study.prepare(tmp_path/"new",tmp_path/"baseline",tmp_path/"replay",tmp_path/"feature")
    assert value["status"]=="resident_only_fallback" and value["runner"]==study.resident.MODULE
    assert value["root_created"] is False and not (tmp_path/"new").exists()


def accepted_fixture(directory):
    trained=checkpoint_fixture(directory/"train")
    chosen=accept.choose(study.task_candidates(directory,trained));best=chosen["candidate"]
    reference=accept.read(best["report"]);path=directory/"verify";path.mkdir()
    value=dict(chosen,strict_state_load=True,fixed_state_unchanged=True,exact_packed_replay=True,
        metrics=reference["metrics"],records=reference["records"])
    write(path/"acceptance.json",value);write(path/"accepted_development.json",reference)
    (path/"accepted.pt").write_bytes(Path(best["checkpoint"]).read_bytes())
    return trained,dict(acceptance=value)


def test_accepted_artifact_is_separate_from_scientific_failure_and_replay_bound(tmp_path):
    trained,result=accepted_fixture(tmp_path)
    before={p.name:accept.base.sha(p) for p in (tmp_path/"train").iterdir()}
    value=study.check_acceptance(tmp_path,trained,result)
    assert not trained["quality_pass"] and trained["selected"]["correct"]==44
    assert value["accepted"] and value["metrics"]["correct"]==47
    assert before=={p.name:accept.base.sha(p) for p in (tmp_path/"train").iterdir()}
    (tmp_path/"verify/accepted.pt").write_bytes((tmp_path/"train/selected.pt").read_bytes())
    with pytest.raises(ValueError,match="Accepted parent bytes"):
        study.check_acceptance(tmp_path,trained,result)


def test_new_stage_uses_accepted_hash_and_reference_not_failed_scientific_choice(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    directory=study.stage_dir(tmp_path,22);directory.mkdir(parents=True)
    trained,verified=accepted_fixture(directory);verified["passed"]=True
    fit=tmp_path/"fit";fit.mkdir()
    for name in ("fit.pt","fit.json","relation.pt","relations.json","interface.json"):
        (fit/name).write_bytes(b"fixed fit")
    parent=dict(cut=22,root_fingerprint="campaign",fits={str(i):str(fit) for i in range(22,36)})
    parent["fingerprint"]=study.base.digest(parent);write(directory/"plan.json",parent)
    for phase in ("train","verify"):write(directory/phase/"manifest.json",{})
    monkeypatch.setattr(study,"verify_root",lambda root:{"fingerprint":"campaign","items_path":"items.json"})
    monkeypatch.setattr(study,"checked",lambda path,phase,stage:trained if phase=="train" else verified)
    stage=study.prepare_stage(tmp_path,21)
    assert stage["parent_correct"]==47
    assert stage["parent_checkpoint"].endswith("verify/accepted.pt")
    assert stage["parent_records"].endswith("verify/accepted_development.json")
    assert stage["parent_checkpoint_sha256"]==verified["acceptance"]["candidate"]["checkpoint_sha256"]
    assert stage["parent_checkpoint_sha256"]!=trained["selected_checkpoint_sha256"]


def test_target_bundle_checks_parent_boundary_and_original_training_population(tmp_path,monkeypatch):
    items=[{"split":"train"} for _ in range(512)]+[{"split":"development"} for _ in range(64)]
    path=tmp_path/"items.json";write(path,{"items":items})
    stage=dict(items_path=str(path),parent_checkpoint_sha256="a"*64,parent_cut=23)
    receipt=dict(shape=[512,1024],training_count=512,parent_cut=23,parent_layers=list(range(23,36)))
    calls=[]
    def load(directory,rows,sha):
        calls.append((directory,rows,sha));return "targets",receipt
    monkeypatch.setattr(study.features,"load_targets",load)
    assert study.target_bundle(tmp_path,stage)[0]=="targets"
    assert calls[0][2]=="a"*64 and calls[0][1]==items
    receipt["parent_cut"]=24
    with pytest.raises(ValueError,match="actual accepted parent"):
        study.target_bundle(tmp_path,stage)
    receipt["parent_cut"]=23;receipt["shape"]=[576,1024]
    with pytest.raises(ValueError,match="actual accepted parent"):
        study.target_bundle(tmp_path,stage)


def test_seven_phase_schedule_uses_feature_pilot_and_accepts_task_winner(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","");calls=[]
    monkeypatch.setattr(study,"verify_root",lambda root:{"fingerprint":"campaign"})
    def prepare(root,cut):
        path=study.stage_dir(root,cut);path.mkdir(parents=True);write(path/"plan.json",{"cut":cut})
    monkeypatch.setattr(study,"prepare_stage",prepare)
    monkeypatch.setattr(study,"verify_stage",lambda root,cut:({},study.read(study.stage_dir(root,cut)/"plan.json")))
    chosen=accept.choose([candidate("scientific_selected",44,.8),candidate("best_task",47,1.7)])
    def checked(path,phase,stage):
        assert (path/phase/"manifest.json").exists()
        return dict(quality_pass=False,selected={"correct":44},quality={"passed":False},
            passed=True,acceptance=chosen,storage={})
    monkeypatch.setattr(study,"checked",checked)
    def run(argv,**kwargs):
        calls.append(argv);phase=argv[argv.index("-m")+2];cut=int(argv[argv.index("--cut")+1])
        path=study.stage_dir(tmp_path,cut)/phase;path.mkdir();write(path/"manifest.json",{})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(study.subprocess,"run",run)
    result=study.campaign(tmp_path,stop_cut=22)
    assert result["status"]=="requested_prefix_completed" and result["compressed_blocks"]==14
    assert [c[c.index("-m")+2] for c in calls]==["capture","fit","pilot","prealign","feature-pilot","train","verify"]
    for argv in calls:
        phase=argv[argv.index("-m")+2];assert argv.count("--allocation")==1
        expected="GPU-0370fdb0=11264"+(",GPU-7b1db3cb=7168" if phase in ("pilot","feature-pilot","train") else "")
        assert argv[argv.index("--allocation")+1]==expected
        assert argv[argv.index("-m")+1]==study.MODULE
    prepare(tmp_path,21);(study.stage_dir(tmp_path,21)/"capture").mkdir();calls.clear()
    with pytest.raises(RuntimeError,match="Incomplete phase"):
        study.campaign(tmp_path,stop_cut=21)
    assert not calls


def test_training_cannot_reach_gpu_without_completed_warm_feature_pilot(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_full_lease
    monkeypatch.setattr(study,"verify_stage",lambda *a:({},{}));calls=[]
    def checked(directory,phase,stage):
        calls.append(phase)
        if phase=="feature-pilot":raise ValueError("Warm feature pilot incomplete")
        return {}
    monkeypatch.setattr(study,"checked",checked)
    monkeypatch.setattr(jspace_full_lease,"wait_ready",lambda *a:pytest.fail("lease requested before warm alpha"))
    with pytest.raises(ValueError,match="Warm feature pilot incomplete"):
        study.run_phase(tmp_path,22,"train")
    assert calls==["capture","fit","pilot","prealign","feature-pilot"]


@pytest.mark.parametrize("change",["none","alpha","development","target_energy","placement"])
def test_warm_feature_pilot_binds_train_calibration_and_resident_cap(tmp_path,monkeypatch,change):
    from test_jspace_3b_resident_campaign import resident_receipt
    from test_jspace_3b_staged_campaign import write_phase,reseal
    items=[dict(family=family,split="train") for family in ("arc","boolq","hellaswag","winogrande") for _ in range(128)]
    items += [dict(family="arc",split="development") for _ in range(64)]
    indices=study.features.feature.calibration_indices(items)
    groups=[dict(indices=indices[i:i+4],endpoint_gradient_squared=4.,feature_gradient_squared=4.) for i in range(0,32,4)]
    receipt=dict(normalization_energy=2.)
    calibration=dict(groups=groups,indices=indices,endpoint_gradient_squared=32.,feature_gradient_squared=32.,
        ratio=.1,alpha=.1,normalization_energy=2.,unchanged_warm_state=True,state_before={"sha":"warm"},
        state_after={"sha":"warm"},development_used=False,optimizer_updates=0)
    report=quality(47);stage=dict(cut=22,fingerprint="stage")
    value=dict(status="completed",passed=True,systems_only=True,discarded=True,raw_replay_exact=True,warm_restore_exact=True,
        raw=report["metrics"],warm=report["metrics"],warm_records=report["records"],calibration=calibration,
        warm_state_fingerprint={"sha":"warm"},target_receipt=receipt,
        probe=dict(passed=True,layer_indices=list(range(22,36)),fp32_master_changed_by_block=[True]*14,
            norm_fp32_master_changed=True,fixed_state_unchanged=True,update=dict(update=1,alpha=.1)))
    if change=="alpha":calibration["alpha"]=.2
    elif change=="development":calibration["development_used"]=True
    elif change=="target_energy":calibration["normalization_energy"]=3.
    path=tmp_path/"feature-pilot"
    write_phase(tmp_path,"feature-pilot",value,stage,{"calibration.json":calibration})
    resident_receipt(path);placement=study.read(path/"resident_optimizer.json")
    placement["optimizer"]=study.OPTIMIZER
    if change=="placement":placement["memory"]["resident_planned_peak_bytes"]=6144*1024**2+1
    write(path/"resident_optimizer.json",placement);reseal(path)
    monkeypatch.setattr(study,"target_bundle",lambda *a:(None,receipt,items))
    real_checked=study.checked
    monkeypatch.setattr(study,"checked",lambda *a:dict(initial=report["metrics"]))
    if change=="none":assert real_checked(tmp_path,"feature-pilot",stage)["calibration"]["alpha"]==.1
    else:
        with pytest.raises(ValueError):real_checked(tmp_path,"feature-pilot",stage)
