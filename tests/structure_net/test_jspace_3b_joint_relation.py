"""Rank-admitted fixed-width fitting and immutable historical-library reuse."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net import jspace_3b_joint_relation as module


def calibration():
    items=[dict(identity=f"{f}-train{i}",split="train",family=f)
        for f in ("arc","boolq","hellaswag","winogrande") for i in range(32)]
    return dict(status="completed",calibration_indices=list(range(128))),items


def test_calibration_rejects_development_and_unbalanced_families():
    meta,items=calibration()
    assert module.check_calibration(meta,items)["count"]==128
    items[0]["split"]="development"
    with pytest.raises(ValueError,match="TRAIN"):module.check_calibration(meta,items)
    items[0]["split"]="train";items[0]["family"]="boolq"
    with pytest.raises(ValueError,match="TRAIN"):module.check_calibration(meta,items)


def spectrum():
    singular=torch.cat([torch.linspace(2.,1.,15),torch.ones(15)*1e-8])
    cutoff=float(singular[0]*30*torch.finfo(torch.float32).eps)
    state=dict(singular=singular,small=torch.diag(singular),vh=torch.eye(30),scale=torch.ones(30))
    report=dict(seed=20863,spectral_state_sha256="prior-spectrum",diagnostics=dict(
        original_shape=[90,30],requested_anchors=24,admitted=False,
        small_svd_gate=dict(rank=15,absolute_cutoff=cutoff),
        diagnostic_anchor_indices=list(range(24)),diagnostic_pivot_norms=list(range(24,0,-1))))
    return report,state


def test_saved_spectrum_uses_original_cutoff_and_pivot_prefix_without_new_sketch():
    report,state=spectrum();before=deepcopy(report);captured=[]
    choose=module.saved_selector(report,state,on_spectrum=lambda d,s:captured.append((d,s)))
    rows=torch.empty(90,30)
    anchors,info=choose(rows,state["scale"],15,seed=20863,svd_rank=64,oversample=16,power_iterations=1,chunk_rows=256)
    assert anchors==list(range(15)) and info["all_rows_used"] and info["normalization_scale_bit_exact"]
    assert captured[0][0]["small_svd_gate"]==report["diagnostics"]["small_svd_gate"]
    assert captured[0][0]["admitted"] and captured[0][1] is state and report==before
    wrong=state["scale"].clone();wrong[0]=torch.nextafter(wrong[0],torch.tensor(float("inf")))
    with pytest.raises(ValueError,match="TRAIN normalization"):
        choose(rows,wrong,15,seed=20863,svd_rank=64,oversample=16,power_iterations=1,chunk_rows=256)
    with pytest.raises(ValueError,match="TRAIN normalization"):
        choose(rows,state["scale"],24,seed=20863,svd_rank=64,oversample=16,power_iterations=1,chunk_rows=256)


def test_saved_spectrum_cannot_lower_rank_cutoff_or_admit_fewer_directions():
    report,state=spectrum();state["singular"][14]=0.
    choose=module.saved_selector(report,state,on_spectrum=lambda *a:pytest.fail("Invalid spectrum published"))
    with pytest.raises(ValueError,match="numerical-rank cutoff"):
        choose(torch.empty(90,30),state["scale"],15,seed=20863,svd_rank=64,oversample=16,power_iterations=1,chunk_rows=256)


def test_saved_and_fresh_selectors_give_identical_relation_fit_for_same_anchors():
    torch.manual_seed(49)
    train,dev,J,weight=torch.randn(64,30),torch.randn(24,30),torch.randn(90,30),torch.randn(4,30)
    mean,scale=module.fitter._normalization(train,256)
    report,state=spectrum();state["scale"]=scale.clone()
    saved=module.saved_selector(report,state,on_spectrum=lambda *a:None)
    def fixed(rows,scale,max_anchors,**kwargs):return list(range(max_anchors)),{}
    options=dict(anchor_widths=(15,),seed=20863,remove_count=5,svd_rank=64,oversample=16,power_iterations=1,chunk_rows=256)
    args=(train,["a"]*32+["b"]*32,J,weight,dev,["a"]*12+["b"]*12)
    a=module.fitter.fit_benchmark_groebner(*args,selector=saved,**options)
    b=module.fitter.fit_benchmark_groebner(*args,selector=fixed,**options)
    for key in ("mean","scale","removed"):assert torch.equal(a["fits"]["15"][key],b["fits"]["15"][key])
    assert torch.equal(a["fits"]["15"]["coefficients"]["2"],b["fits"]["15"]["coefficients"]["2"])
    assert a["metrics"]==b["metrics"]


def write_manifest(library,path,paths,fingerprint):
    module.base.write(path,dict(status="completed",fingerprint=fingerprint,
        artifacts={str(p.relative_to(library)):module.base.sha(p) for p in paths}))


def library_fixture(root):
    root.mkdir();parent=dict(schema="jspace.qwen-parallel.v2",systems_only=False,
        config=dict(rank=1024,head_dim=64,anchors=24,removed_channels=1024),model={"path":"native"},
        groups=[list(range(30-6*g,36-6*g)) for g in range(6)])
    parent["fingerprint"]=module.protocol_digest(parent);module.base.write(root/"plan.json",parent)
    for g,layers in enumerate(parent["groups"]):
        directory=root/"groups"/f"group_{g:02}";capture=directory/"capture";capture.mkdir(parents=True)
        meta,items=calibration();meta["teacher_fingerprint"]=f"teacher{g}"
        module.base.write(capture/"metadata.json",meta);module.base.write(capture/"items.json",dict(items=items))
        write_manifest(root,capture/"manifest.json",[capture/"metadata.json",capture/"items.json"],parent["fingerprint"])
        basisfiles=[]
        for layer in layers:
            d=directory/f"block_{layer:02}";d.mkdir();(d/"fit.pt").write_bytes(b"native-fit")
            module.base.write(d/"fit.json",dict(rank=1024,head_dim=64,teacher_fingerprint=f"teacher{g}"))
            module.base.write(d/"interface.json",dict(layer=layer,teacher_fingerprint=f"teacher{g}",input_sha256={}))
            basisfiles += [d/"fit.pt",d/"fit.json",d/"interface.json"]
            if layer in module.WIDTHS:
                if layer==2:
                    (d/"rank_spectrum.pt").write_bytes(b"original-spectrum")
                    module.base.write(d/"rank_diagnostic.json",dict(layer=2,seed=20863,teacher_fingerprint=f"teacher{g}",
                        diagnostics=dict(requested_anchors=24,admitted=False,small_svd_gate=dict(rank=15)),
                        spectral_state_sha256=module.base.sha(d/"rank_spectrum.pt"),
                        input_sha256={str(d/"interface.json"):module.base.sha(d/"interface.json")}))
            else:
                (d/"relation.pt").write_bytes(b"relation")
                module.base.write(d/"relations.json",dict(fixed_anchor_width=24,removed_channels=1024,teacher_fingerprint=f"teacher{g}"))
                write_manifest(root,d/"fit_result/manifest.json",[d/"relation.pt",d/"relations.json"],parent["fingerprint"])
        write_manifest(root,directory/"basis/manifest.json",basisfiles,parent["fingerprint"])
    return root


def test_prepare_pins_36_distinct_fit_paths_and_preserves_failed_parent(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    source=library_fixture(tmp_path/"library");original=(source/"groups/group_05/block_02/rank_diagnostic.json").read_bytes()
    monkeypatch.setattr(module,"local_import_closure",lambda p:[Path(module.__file__).resolve()])
    out=tmp_path/"new";module.prepare(out,source);plan=module.verify(out)
    assert len(plan["layers"])==36 and sum(r["new_relation"] for r in plan["layers"].values())==2
    assert plan["anchor_widths"]=={"2":15,"5":24}
    assert Path(plan["layers"]["2"]["directory"])==out/"block_02"
    assert Path(plan["layers"]["3"]["directory"])==source/"groups/group_05/block_03"
    assert (source/"groups/group_05/block_02/rank_diagnostic.json").read_bytes()==original
    with pytest.raises(FileExistsError):module.prepare(out,source)
    (out/"block_02/fit.pt").write_bytes(b"changed")
    with pytest.raises(ValueError,match="Copied fitting"):module.verify(out)


def test_failed_fit_is_preserved_and_not_retried(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    block=tmp_path/"block_05";block.mkdir()
    plan=dict(fingerprint="fp",layers={"5":dict(directory=str(block),source="old")})
    monkeypatch.setattr(module,"verify",lambda root:plan)
    def fail(*a,**kw):raise ValueError("Rank below24")
    monkeypatch.setattr(module.fitter,"fit_relation",fail)
    with pytest.raises(ValueError,match="Rank below24"):module.run(tmp_path,5)
    assert module.read(block/"failure.json")["status"]=="failed"
    with pytest.raises(FileExistsError):module.run(tmp_path,5)


def test_non_cpu_environment_rejected_before_input_access(tmp_path,monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES",raising=False)
    with pytest.raises(ValueError,match="CUDA hidden"):module.prepare(tmp_path/"new",tmp_path/"missing")


def test_saved_spectrum_is_bound_to_exact_raw_jacobian_hash_not_only_shape_scale(tmp_path):
    path=tmp_path/"interface.json";interface=dict(input_sha256={str(tmp_path/"jacobian.npy"):"original-J"})
    module.base.write(path,interface)
    recorded=dict(input_sha256={**interface["input_sha256"],str(path):module.base.sha(path)})
    module.validate_spectrum_provenance(recorded,interface,path)
    recorded["input_sha256"][str(tmp_path/"jacobian.npy")]="different-J-same-shape"
    with pytest.raises(ValueError,match="exact Jacobian"):
        module.validate_spectrum_provenance(recorded,interface,path)
