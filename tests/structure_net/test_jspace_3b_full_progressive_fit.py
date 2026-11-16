"""All-depth continuation preserves failed/pending quality without bypassing systems."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from experiments.structure_net import jspace_3b_full_progressive_fit as study


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def plan(path,value):
    value=dict(value);value["fingerprint"]=study.dataset.data.digest(value);put(path,value);return value


def data(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    root=tmp_path/"data";records=[]
    for family in range(8):
        for i in range(20):
            records.append(SimpleNamespace(example_id=f"{family}-{i}",group_id=f"g{family}-{i}",split="train",
                prompt_ids=(1,2),response_ids=(3,4),input_ids=(1,2,3,4),finish_reason="eos",task=str(family),provenance={}))
    p=dict(model={"path":"native","revision":"pinned"},records_sha256="records",rows=[{"id":"original"}],cuts=list(range(36)),
        seed_data={"fingerprint":"native-v2-bank"},
        calibration_indices=[f*20+i for f in range(8) for i in range(16)],
        diagnostic_indices=[f*20+i for f in range(8) for i in range(16,20)],input_sha256={})
    p=plan(root/"plan.json",p)
    result=put(root/"native/result.json",dict(full_teacher_bit_exact=True,existing_boundary_bit_exact=True,teacher_unchanged=True,
        seed_boundary35_bit_exact=True,seed_data_fingerprint="native-v2-bank"))
    put(root/"native/manifest.json",dict(status="completed",fingerprint=p["fingerprint"],artifacts={"result.json":study.sha(result)},
        artifact_identity={"result.json":study.identity(result)}))
    monkeypatch.setattr(study.dataset,"verify",lambda *_a,**_k:(p,records))
    source=study.ROOT/"experiments/structure_net/jspace_3b_full_progressive_fit.py"
    monkeypatch.setattr(study,"_sources",lambda:{str(source):study.sha(source)})
    return root,p,records


def make_fits(root,cut):
    fits={};inputs={}
    for i in range(cut,36):
        directory=root/f"fits/block_{i:02d}";directory.mkdir(parents=True)
        fits[str(i)]=str(directory)
        for name in ("fit.pt","fit.json","relation.pt","relations.json","interface.json"):
            path=directory/name;path.write_text(str(i)+name);inputs[str(path)]=study.sha(path)
    return fits,inputs


def parent(root,dp,cut,*,quality=None,systems=True):
    fits,inputs=make_fits(root,cut)
    checkpoint=root/"run/checkpoints/update_0046.pt";checkpoint.parent.mkdir(parents=True);checkpoint.write_text("checkpoint")
    p=plan(root/"plan.json",dict(schema=study.RECOVERY_SCHEMA,cut=cut,model=dp["model"],
        data_fingerprint=dp["fingerprint"],corpus_fingerprint=dp["records_sha256"],fits=fits,input_sha256=inputs))
    result=put(root/"run/gpu_result.json",dict(status="completed",fingerprint=p["fingerprint"],
        systems_passed=systems,quality_status="pending" if quality is None else "completed",
        relative_retention_pass=quality,continuation_checkpoint=str(checkpoint),continuation_checkpoint_sha256=study.sha(checkpoint)))
    put(root/"run/gpu_manifest.json",dict(status="completed",fingerprint=p["fingerprint"],
        artifacts={"gpu_result.json":study.sha(result)},checkpoints={str(checkpoint):study.sha(checkpoint)}))
    return p,checkpoint


@pytest.mark.parametrize("cut",[34,17,0])
@pytest.mark.parametrize("quality",[False,None])
def test_every_depth_continues_failed_or_pending_quality_but_pins_actual_parent(tmp_path,monkeypatch,cut,quality):
    data_root,dp,_=data(tmp_path,monkeypatch);parent_root=tmp_path/"parent"
    _,checkpoint=parent(parent_root,dp,cut+1,quality=quality)
    result=study.prepare(tmp_path/"fit",data_root,cut=cut,parent_root=parent_root)
    p=study.verify(tmp_path/"fit")
    assert p["parent"]["relative_retention_pass"] is quality
    assert p["parent"]["checkpoint"]==str(checkpoint)
    assert result["cut"]==cut and len(p["parent"]["fits"])==35-cut
    assert study.check(tmp_path/"fit")["status"]=="prepared"
    # A later CPU grading result is neither read nor frozen as model selection.
    put(parent_root/"run/task_result.json",{"relative_retention_pass":False})
    assert study.verify(tmp_path/"fit")["fingerprint"]==p["fingerprint"]


def test_quality_does_not_bypass_failed_systems_wrong_data_or_changed_checkpoint(tmp_path,monkeypatch):
    _,dp,_=data(tmp_path,monkeypatch);root=tmp_path/"parent"
    _,checkpoint=parent(root,dp,34,quality=False,systems=False)
    with pytest.raises(ValueError,match="systems"):
        study.continuation_parent(root,33,dp)
    with pytest.raises(ValueError,match="adjacent"):
        study.continuation_parent(root,32,dp)
    with pytest.raises(ValueError,match="exact native"):
        study.continuation_parent(root,33,dict(dp,records_sha256="changed"))
    result=study.read(root/"run/gpu_result.json");result["systems_passed"]=True
    put(root/"run/gpu_result.json",result)
    manifest=study.read(root/"run/gpu_manifest.json");manifest["artifacts"]["gpu_result.json"]=study.sha(root/"run/gpu_result.json")
    put(root/"run/gpu_manifest.json",manifest);checkpoint.write_text("tampered")
    with pytest.raises(ValueError,match="changed"):
        study.continuation_parent(root,33,dp)


def test_existing_update46_seed_retains_old_nll_failure_and_cannot_refit35(tmp_path,monkeypatch):
    data_root,dp,_=data(tmp_path,monkeypatch);source=tmp_path/"v2/blocks_01/recovery"
    fits,inputs=make_fits(source,35);old_data=source.parents[1]/"data"
    oldp=plan(old_data/"plan.json",{k:dp[k] for k in ("model","records_sha256","rows","calibration_indices","diagnostic_indices")})
    p=plan(source/"plan.json",dict(schema="jspace.3b-progressive-response-recovery.v1",cut=35,
        data_root=str(old_data),data_fingerprint=oldp["fingerprint"],model=dp["model"],
        corpus_fingerprint=dp["records_sha256"],fits=fits,input_sha256=inputs))
    checkpoint=source/"run/checkpoints/update_0046.pt";checkpoint.parent.mkdir(parents=True);checkpoint.write_text("original-v2")
    candidate=dict(update=46,checkpoint=dict(path=str(checkpoint),sha256=study.sha(checkpoint)),
        packed_reload_passed=True,packed_reload=dict(packed_state_exact=True),nll=2.950581,mc_accuracy=.7734375,
        cheap_gate={"checks":{"response_nll":False,"multiple_choice":True},"passed":False},
        eligibility={"passed":False},grading_status="not_run_failed_cheap_retention")
    result=put(source/"run/result.json",dict(status="completed",fingerprint=p["fingerprint"],
        relative_retention_pass=False,candidates=[candidate]))
    put(source/"run/manifest.json",dict(status="completed",fingerprint=p["fingerprint"],
        artifacts={"result.json":study.sha(result)},checkpoints={str(checkpoint):study.sha(checkpoint)}))
    receipt=study.seed(tmp_path/"seed",data_root,source,checkpoint)
    assert receipt["source_quality"]["cheap_gate"]["checks"]["response_nll"] is False
    assert receipt["relative_retention_pass"] is False and receipt["source_data_fingerprint"]!=dp["fingerprint"]
    admitted,_=study.continuation_parent(tmp_path/"seed",34,dp)
    assert admitted["checkpoint_sha256"]==study.sha(checkpoint)
    with pytest.raises(ValueError,match="never refit35"):
        study.prepare(tmp_path/"bad",data_root,cut=35,parent_root=tmp_path/"seed")


def test_full_depth_adapter_reuses_exact_geometry_and_all_response_admission(tmp_path,monkeypatch):
    _,dp,records=data(tmp_path,monkeypatch);calls=[]
    monkeypatch.setattr(study.fitting,"fit_stage",lambda *a,**kw:calls.append(kw) or {"completed":True})
    p=dict(cut=0,configuration=asdict(study.fitting.FitConfig()),model=dp["model"],
        calibration_indices=dp["calibration_indices"],diagnostic_indices=dp["diagnostic_indices"],
        teacher_fingerprint="authenticated",parent=dict(fits={str(i):f"fit{i}" for i in range(1,36)},checkpoint="parent.pt"))
    result=study.fit_stage(tmp_path,SimpleNamespace(records=records),plan=p,guard=lambda:None,device="cpu")
    assert result["completed"] and calls[0]["cut"]==0 and calls[0]["production"] is False
    assert calls[0]["config"]==study.fitting.FitConfig() and len(calls[0]["parent_fits"])==35
    assert calls[0]["calibration_indices"]==dp["calibration_indices"]
