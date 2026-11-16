"""Frozen lifecycle and accepted-adjacent provenance without a GPU allocation."""
from contextlib import nullcontext
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import jspace_3b_rebuild_fit_campaign as study


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def prepared_data(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    root=tmp_path/"data";records=[]
    for family in range(8):
        for i in range(20):
            records.append(SimpleNamespace(example_id=f"{family}-{i}",group_id=f"g{family}-{i}",split="train",
                prompt_ids=(1,2),response_ids=(3,4),input_ids=(1,2,3,4),finish_reason="eos",task=str(family),provenance={}))
    p=dict(fingerprint="data-fp",cuts=[35,34],model={"path":"native"},records_sha256="records",
        calibration_indices=[f*20+i for f in range(8) for i in range(16)],
        diagnostic_indices=[f*20+i for f in range(8) for i in range(16,20)],input_sha256={})
    put(root/"plan.json",p)
    result=put(root/"native/result.json",dict(full_teacher_bit_exact=True,existing_boundary_bit_exact=True,teacher_unchanged=True))
    put(root/"native/manifest.json",dict(status="completed",fingerprint="data-fp",artifacts={"result.json":study.sha(result)}))
    monkeypatch.setattr(study.dataset,"verify",lambda *_a,**_k:(p,records))
    monkeypatch.setattr(study.dataset.cache.prior.source.inherited,"local_import_closure",lambda _: [study.ROOT/"experiments/structure_net/jspace_3b_rebuild_fit_campaign.py"])
    return root,p


def test_native_prepare_freezes_configuration_inputs_and_source(tmp_path,monkeypatch):
    data,_=prepared_data(tmp_path,monkeypatch);root=tmp_path/"fit"
    result=study.prepare(root,data,cut=35)
    p=study.verify(root)
    assert p["parent"] is None and result["fingerprint"]==p["fingerprint"]
    assert p["configuration"]==asdict(study.fitting.FitConfig())
    assert study.check(root)["status"]=="prepared"
    (data/"native/result.json").write_text("changed")
    with pytest.raises(ValueError,match="input changed"):study.verify(root)


def parent(tmp_path,*,passed=True):
    root=tmp_path/"parent";fits=root/"fit"
    inputs={}
    for name in ("fit.pt","fit.json","relation.pt","relations.json","interface.json"):
        path=fits/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name);inputs[str(path)]=study.sha(path)
    checkpoint=root/"run/checkpoints/update_0001.pt";checkpoint.parent.mkdir(parents=True);checkpoint.write_text("selected")
    p=dict(schema="jspace.3b-progressive-response-recovery.v1",cut=35,data_fingerprint="data-fp",
           fits={"35":str(fits)},input_sha256=inputs)
    p["fingerprint"]=study.dataset.data.digest(p);put(root/"plan.json",p)
    result=put(root/"run/result.json",dict(status="completed",relative_retention_pass=passed,fingerprint=p["fingerprint"],
        selected_checkpoint=str(checkpoint),selected_checkpoint_sha256=study.sha(checkpoint)))
    put(root/"run/manifest.json",dict(status="completed",fingerprint=p["fingerprint"],artifacts={"result.json":study.sha(result)}))
    return root,checkpoint


def test_stage34_requires_authenticated_passed_adjacent_parent(tmp_path,monkeypatch):
    data,_=prepared_data(tmp_path,monkeypatch);p,checkpoint=parent(tmp_path)
    result=study.prepare(tmp_path/"fit",data,cut=34,parent_checkpoint=checkpoint)
    plan=study.verify(tmp_path/"fit")
    assert result["cut"]==34 and plan["parent"]["checkpoint"]==str(checkpoint)
    assert plan["parent"]["fits"].keys()=={"35"}
    with pytest.raises(ValueError,match="adjacent"):
        study.accepted_parent(p,33)
    with pytest.raises(ValueError,match="adjacent"):
        study.accepted_parent(p,34,data_fingerprint="different-corpus")
    checkpoint.write_text("changed")
    with pytest.raises(ValueError,match="changed"):
        study.accepted_parent(p,34)


def test_retention_failure_cannot_become_next_geometry(tmp_path):
    root,_=parent(tmp_path,passed=False)
    with pytest.raises(ValueError,match="accepted adjacent"):
        study.accepted_parent(root,34)


def test_run_uses_guard_frozen_cap_and_checks_completed_artifacts(tmp_path,monkeypatch):
    data,_=prepared_data(tmp_path,monkeypatch);root=tmp_path/"fit";study.prepare(root,data,cut=35)
    calls=[]
    monkeypatch.setattr(torch.cuda,"_lazy_init",lambda:pytest.fail("Mock lifecycle initialized CUDA"))
    monkeypatch.setattr(study.dataset.cache,"reservation",lambda n:({"id":"fixture","allocation":{"3060":11264}},lambda:calls.append("guard")))
    monkeypatch.setattr(study.dataset,"RebuildDataset",lambda *_a,**_k:SimpleNamespace(close=lambda:calls.append("close")))
    monkeypatch.setattr(study.dataset.cache.execution,"arithmetic_policy",nullcontext)
    monkeypatch.setattr(torch.cuda,"max_memory_allocated",lambda:12)
    monkeypatch.setattr(torch.cuda,"max_memory_reserved",lambda:16)
    def fit(out,dataset,**kwargs):
        kwargs["guard"]();out.mkdir()
        receipt=dict(status="completed",teacher_fingerprint=kwargs["teacher_fingerprint"],cut=35,
            configuration=asdict(kwargs["config"]),native_seam_bit_exact=True,saved_fit_compilation_passed=True,
            fit_directory=str(out/"fits/block_35"))
        put(out/"fit_receipt.json",receipt);return receipt
    monkeypatch.setattr(study.fitting,"fit_stage",fit)
    with pytest.raises(ValueError,match="Allocator"):
        study.run(root,limit_mib=9999)
    assert calls==[]
    result=study.run(root)
    assert result["status"]=="completed" and calls==["guard","guard","close"]
    assert study.check(root)["lease_id"]=="fixture"
    with pytest.raises(FileExistsError):study.run(root)
    (root/"work/fit_receipt.json").write_text("changed")
    with pytest.raises(ValueError,match="receipt changed"):study.check(root)
