"""Offline common functional coordinates and all-layer polynomial compilation."""
import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from . import jspace_full_capture as capture
from .jspace_bench_data import DATASETS,sha,write
from .jspace_groebner_tensor import anchor_indices,fit_coefficients,features,graph_certificate

DEFAULT_ROOT=capture.ROOT/"data/experiments/jspace_full_compact/20261113_fit_v1"


def derive_root(root,parent,paths,inputs,**extra):
    capture.cpu_only()
    if root.exists():raise FileExistsError(root)
    previous=capture.verify(parent)
    value={**previous,"source_sha256":{**previous["source_sha256"],**{str(p):sha(p) for p in paths}},
           "input_sha256":{**previous["input_sha256"],**{str(p):sha(p) for p in inputs}},
           "parent_root":str(parent.resolve()),"created_at":datetime.now(timezone.utc).isoformat(),**extra}
    value.pop("scientific_fingerprint");value["scientific_fingerprint"]=capture.digest(value)
    root.mkdir(parents=True)
    for name in value["source_sha256"]:
        path=Path(name);target=root/"producer"/path.relative_to(capture.ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    shutil.copy2(parent/"tasks.json",root/"tasks.json");shutil.copy2(parent/"user_recipe.txt",root/"user_recipe.txt")
    write(root/"plan.json",value)
    return {"fingerprint":value["scientific_fingerprint"],"root":str(root),"parent":str(parent)}


def prepare(root,parent):
    capture.checked(parent,"capture")
    return derive_root(root,parent,[Path(__file__)],[parent/"capture_manifest.json"],capture_root=str(parent.resolve()))


def reuse_report(errors,own_errors,anchors):
    good=(errors.detach().cpu()<=capture.CONFIG["shared_relation_nrmse"])
    good[:,anchors]=False
    own=(own_errors.detach().cpu()<=capture.CONFIG["shared_relation_nrmse"]);own[:,anchors]=False
    counts=good.sum(0);total=int(counts.sum());unique=int((counts>0).sum())
    return {"threshold_nrmse":capture.CONFIG["shared_relation_nrmse"],"family_order":list(DATASETS),
            "common_basis_eligible_per_task":good.sum(1).tolist(),"task_specific_fit_eligible_per_task":own.sum(1).tolist(),
            "task_uses":total,"unique_elements":unique,"redundant_occurrences":total-unique,
            "redundant_fraction":(total-unique)/total if total else 0.,
            "shared_at_least_two":int((counts>=2).sum()),"shared_all_four":int((counts==4).sum()),
            "overlap_counts":(good.long()@good.long().T).tolist()}


def fit(root):
    parent=Path(capture.verify(root)["capture_root"]);capture.checked(parent,"capture")
    stage=capture.Stage(root,"fit")
    # Fitting does not use the teacher weights; release them before the eigensolve.
    del stage.model;gc.collect();torch.cuda.empty_cache()
    stats=torch.load(parent/"capture/statistics.pt",map_location="cpu",weights_only=True)
    meta=json.loads((parent/"capture/items.json").read_text())
    total=torch.zeros(2048,2048,device="cuda:0")
    for gram in stats["grams"]:
        g=gram.cuda();total.add_(g/g.trace().clamp_min(1e-30))
    eigenvalues,q=torch.linalg.eigh(total)
    q=q.flip(1).contiguous();eigenvalues=eigenvalues.flip(0)
    orthogonality=float((q.T@q-torch.eye(2048,device=q.device)).abs().max())
    if orthogonality>1e-4 or not torch.isfinite(q).all():raise ValueError("Invalid common orthogonal basis")
    rank=capture.CONFIG["functional_rank"];fits={};summaries={};fractions=[]
    for layer in range(36):
        gram=stats["grams"][layer].cuda()
        fractions.append(float(((q[:,:rank].T@gram)*q[:,:rank].T).sum()/gram.trace().clamp_min(1e-30)))
        train=capture.read_bits(np.load(parent/f"capture/mlp_calibration_{layer:02d}.npy",mmap_mode="r"),slice(None)).float()
        dev=capture.read_bits(np.load(parent/f"capture/mlp_development_{layer:02d}.npy",mmap_mode="r"),slice(None)).float()
        jac=torch.from_numpy(np.array(np.load(parent/f"capture/mlp_jacobian_{layer:02d}.npy",mmap_mode="r"))).cuda()
        mean=train.mean(0);scale=train.std(0).clamp_min(1e-5)
        z=(train-mean)/scale;zd=(dev-mean)/scale
        anchors,singular,bank=anchor_indices(jac,scale,capture.CONFIG["anchors"])
        coefficient=fit_coefficients(z,anchors,2,capture.CONFIG["ridge"])
        predicted=features(zd[:,anchors],2)@coefficient.T
        errors=[];own_errors=[]
        for family in DATASETS:
            mask=torch.tensor([f==family for f in meta["development_families"]],device=q.device)
            trainmask=torch.tensor([f==family for f in meta["calibration_families"]],device=q.device)
            normalizer=zd[mask].square().mean(0).clamp_min(1e-6)
            errors.append(((predicted[mask]-zd[mask]).square().mean(0)/normalizer).sqrt())
            own=fit_coefficients(z[trainmask],anchors,2,capture.CONFIG["ridge"])
            own_errors.append(((features(zd[mask][:,anchors],2)@own.T-zd[mask]).square().mean(0)/normalizer).sqrt())
        errors=torch.stack(errors);own_errors=torch.stack(own_errors)
        sensitivity=(jac*scale).square().mean(0).sqrt()
        score=errors.max(0).values*sensitivity.clamp_min(1e-8);score[anchors]=torch.inf
        removed=score.argsort()[:capture.CONFIG["removed_per_layer"]].cpu().tolist()
        group=int(stats["group_energy"][layer].argmin())
        pairs=stats["pair_energy"][layer,group].topk(capture.CONFIG["rope_pairs"]).indices.sort().values.tolist()
        count=stats["count"];vmean=stats["value_sum"][layer,group]/count
        covariance=stats["value_cov"][layer,group]/count-vmean[:,None]*vmean[None,:]
        ve,s=torch.linalg.eigh(covariance.cuda());s=s[:, -2*len(pairs):].flip(1).contiguous()
        entry={"anchors":anchors,"mean":mean.cpu(),"scale":scale.cpu(),"coefficients":{"2":coefficient.cpu()},
               "removed":removed,"jacobian_singular_values":singular.cpu(),"attention_group":group,
               "rope_pairs":pairs,"value_basis":s.cpu(),"mean_coordinates":(stats["mean"][layer].cuda()@q).cpu()}
        fits[str(layer)]=entry
        certificate=graph_certificate(coefficient,anchors,removed,2)
        summaries[str(layer)]={"anchors":anchors,"removed_count":len(removed),"attention_group":group,"rope_pairs":pairs,
            "common_core_jacobian_energy_fraction":fractions[-1],"redundancy":reuse_report(errors,own_errors,anchors),
            "selected_per_task_median_nrmse":errors[:,removed].median(1).values.cpu().tolist(),"certificate":certificate,
            "retained_value_variance_fraction":float(ve[-2*len(pairs):].sum()/ve.sum()),
            "calibration_tokens":len(z),"development_tokens":len(zd),"jacobian_rows":len(jac)}
        stage.event("layer_complete",layer=layer,core_energy=fractions[-1],removed=len(removed),
                    redundant=summaries[str(layer)]["redundancy"]["redundant_occurrences"])
        del train,dev,jac,z,zd,predicted,coefficient,own,bank,gram,errors,own_errors
        gc.collect();torch.cuda.empty_cache()
    capture.save(root/"fit.pt",{"basis":q.cpu(),"layers":fits,"eigenvalues":eigenvalues.cpu()})
    write(root/"fit.json",{"layers":summaries,"basis_orthogonality_max_abs":orthogonality,
        "functional_rank":rank,"retained_residual_rank":2048-rank,"jacobian_energy_fractions":fractions,
        "global_core_energy_fraction":float(eigenvalues[:rank].sum()/eigenvalues.sum()),
        "scope":"offline common functional basis plus rational graph compilation; relation errors remain empirical"})
    return stage.finish(["fit.pt","fit.json"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","fit"))
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT);p.add_argument("--capture-root",type=Path,default=capture.DEFAULT_ROOT)
    a=p.parse_args();print(json.dumps(prepare(a.root,a.capture_root) if a.stage=="prepare" else fit(a.root),allow_nan=False),flush=True)
