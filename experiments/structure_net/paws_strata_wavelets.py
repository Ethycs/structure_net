#!/usr/bin/env python3
"""Experiments 08/09: operational Whitney diagnostics and task diffusion wavelets."""
from __future__ import annotations

import argparse, asyncio, hashlib, json, time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csgraph
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner
try:
    from experiments.structure_net.paws_embedding_router import lexical_features
    from experiments.structure_net.paws_teacher_annotation import eligible_rows, sha256
except ModuleNotFoundError:
    from paws_embedding_router import lexical_features
    from paws_teacher_annotation import eligible_rows, sha256

SCHEMA_VERSION="nal.paws-strata-wavelets.v1"


def carrier(atlas:Any,rows:list[dict[str,Any]])->tuple[np.ndarray,np.ndarray,np.ndarray]:
    construction=np.isin(atlas["partition"],[1,2]); embedding=atlas["embedding"].astype(np.float32)
    pca=PCA(n_components=16,random_state=17).fit(embedding[construction]); semantic=pca.transform(embedding)
    lexical=np.stack([lexical_features(row["sentence1"],row["sentence2"]) for row in rows]); base=np.concatenate((semantic,lexical),axis=1); scaler=StandardScaler().fit(base[construction]); base=scaler.transform(base)
    success=np.stack([atlas["a_prediction"]==atlas["labels"],atlas["b_prediction"]==atlas["labels"],atlas["c_prediction"]==atlas["labels"]],axis=1).astype(float)
    atlas_mask=atlas["partition"]==1;k=min(31,int(atlas_mask.sum()));distance,index=NearestNeighbors(n_neighbors=k).fit(base[atlas_mask]).kneighbors(base); weights=1/(distance+1e-6); probability=(weights[:,:,None]*success[atlas_mask][index]).sum(1)/weights.sum(1)[:,None]
    logits=np.log(np.clip(probability,1e-4,1-1e-4)/np.clip(1-probability,1e-4,1)); return np.concatenate((base,logits),axis=1),probability,success


def stratum_keys(labels:np.ndarray,success:np.ndarray)->np.ndarray:
    return np.asarray([f"label={label}|family=paws_wiki_unknown|signature={''.join(str(int(bit)) for bit in row)}" for label,row in zip(labels,success)])


def strata_diagnostics(x:np.ndarray,keys:np.ndarray,construction:np.ndarray,audit:np.ndarray)->dict[str,Any]:
    result={}; tangents={}; rng=np.random.default_rng(17)
    for key in sorted(set(keys[construction])):
        values=x[construction & (keys==key)]; centered=values-values.mean(0); _,singular,vh=np.linalg.svd(centered,full_matrices=False); total=float(np.sum(singular**2)); energy=np.cumsum(singular**2)/max(1e-12,total); rank=0 if total<=1e-12 else int(np.searchsorted(energy,.9)+1); tangents[key]=vh[:rank]
        ranks=[]
        if len(values)>=10:
            for _ in range(32):
                sample=values[rng.integers(0,len(values),len(values))]; s=np.linalg.svd(sample-sample.mean(0),compute_uv=False); total_sample=float(np.sum(s**2)); e=np.cumsum(s**2)/max(1e-12,total_sample); ranks.append(0 if total_sample<=1e-12 else int(np.searchsorted(e,.9)+1))
        result[key]={"rows":len(values),"tangent_rank_90":rank,"bootstrap_rank_mode_fraction":max((ranks.count(value) for value in set(ranks)),default=0)/max(1,len(ranks))}
    train_x=x[construction]; train_keys=keys[construction]; audit_x=x[audit]; audit_keys=keys[audit]; _,neighbor=NearestNeighbors(n_neighbors=8).fit(train_x).kneighbors(audit_x)
    frontier={}; residual=[]
    for i,indices in enumerate(neighbor):
        for j in indices:
            if audit_keys[i]==train_keys[j]: continue
            edge=" -> ".join(sorted((audit_keys[i],train_keys[j]))); frontier[edge]=frontier.get(edge,0)+1
            tangent=tangents.get(train_keys[j]); secant=audit_x[i]-train_x[j]; norm=np.linalg.norm(secant)
            if tangent is not None and norm>0: residual.append(float(np.linalg.norm(secant-tangent.T@(tangent@secant))/norm))
    return {"strata":result,"frontier_incidence":frontier,"audit_cross_frontier_pairs":len(residual),"whitney_b_secant_residual_median":float(np.median(residual)) if residual else None,"whitney_b_secant_residual_p95":float(np.quantile(residual,.95)) if residual else None,"claim_boundary":"operational finite-sample diagnostics; not a Whitney-stratification proof"}


def varimax(phi:np.ndarray,iterations:int=64,tolerance:float=1e-7)->np.ndarray:
    if phi.shape[1]<=1:return phi
    rotation=np.eye(phi.shape[1]); previous=0.0
    for _ in range(iterations):
        value=phi@rotation; u,s,vh=np.linalg.svd(phi.T@(value**3-value@(np.diag(np.sum(value**2,axis=0))/phi.shape[0])),full_matrices=False); rotation=u@vh; objective=s.sum()
        if objective-previous<tolerance:break
        previous=objective
    return phi@rotation


def wavelet_basis(x:np.ndarray,probability:np.ndarray,keys:np.ndarray,indices:np.ndarray)->tuple[dict[str,Any],dict[str,np.ndarray]]:
    x=x[indices]; probability=probability[indices]; keys=keys[indices]; n=len(x); distance,neighbor=NearestNeighbors(n_neighbors=min(13,n)).fit(x).kneighbors(x); sigma=float(np.median(distance[:,1:]))
    weights=np.zeros((n,n),dtype=np.float64)
    for i in range(n):
        for d,j in zip(distance[i,1:],neighbor[i,1:]):
            midpoint=np.clip((probability[i]+probability[j])/2,1e-4,1-1e-4); fisher=float(np.sum((probability[i]-probability[j])**2/(midpoint*(1-midpoint)))); value=np.exp(-(d/max(sigma,1e-8))**2)/(1+fisher)
            if keys[i]!=keys[j]:value=max(value,.02)
            weights[i,j]=max(weights[i,j],value)
    weights=np.maximum(weights,weights.T); bridges=[]
    components,component_label=csgraph.connected_components(weights>0,directed=False)
    pair_distance=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2)
    while components>1:
        masked=np.where(component_label[:,None]!=component_label[None,:],pair_distance,np.inf); i,j=np.unravel_index(np.argmin(masked),masked.shape);weights[i,j]=weights[j,i]=max(weights[i,j],.02);bridges.append((int(i),int(j)));components,component_label=csgraph.connected_components(weights>0,directed=False)
    components=int(components); degree=weights.sum(1); normalized=weights/np.sqrt(degree[:,None]*degree[None,:]); diffusion=(np.eye(n)+normalized)/2
    eigenvalue,eigenvector=np.linalg.eigh(diffusion); order=np.argsort(eigenvalue)[::-1]; eigenvalue=eigenvalue[order]; eigenvector=eigenvector[:,order]
    band=np.full(n,6,dtype=np.int8)
    for column,value in enumerate(np.abs(eigenvalue)):
        for scale in range(6):
            if value**(2**scale)<.1:band[column]=scale;break
    columns=[]; bands=[]
    for scale in [6,5,4,3,2,1,0]:
        block=eigenvector[:,band==scale]
        if block.shape[1]: columns.append(varimax(block)); bands.extend([scale]*block.shape[1])
    basis=np.concatenate(columns,axis=1); orth=float(np.linalg.norm(basis.T@basis-np.eye(n),ord=2)); signal=(probability-probability.mean(0)); coefficients=basis.T@signal; energy=np.sum(coefficients**2,axis=1); needed=int(np.searchsorted(np.cumsum(np.sort(energy)[::-1]),.95*energy.sum())+1)
    truncated=basis[:,:min(128,n)]; reconstruction=truncated@(truncated.T@signal); error=float(np.linalg.norm(signal-reconstruction)/max(1e-12,np.linalg.norm(signal)))
    boundary=np.asarray([any(keys[i]!=keys[j] for j in neighbor[i,1:]) for i in range(n)],dtype=float); boundary_coeff=basis.T@(boundary-boundary.mean()); keep=np.argsort(boundary_coeff**2)[::-1][:max(1,n//4)]; boundary_reconstruction=basis[:,keep]@boundary_coeff[keep]; recall=float(np.mean(boundary[np.argsort(boundary_reconstruction)[-max(1,int(boundary.sum())):]])) if boundary.sum() else 1.0
    metrics={"landmarks":n,"connected_components":components,"seam_bridges":len(bridges),"orthogonality_error":orth,"diffusion_eigenvalue_min":float(eigenvalue.min()),"diffusion_eigenvalue_max":float(eigenvalue.max()),"competence_reconstruction_error_128":error,"coefficients_for_95pct_competence_energy":needed,"boundary_recall_top_mass":recall,"bands":{str(scale):bands.count(scale) for scale in set(bands)}}
    return metrics,{"basis":basis.astype(np.float32),"diffusion_eigenvalues":eigenvalue.astype(np.float32),"band":np.asarray(bands,dtype=np.int8),"landmark_indices":indices,"weights":weights.astype(np.float32)}


def deterministic_landmarks(keys:np.ndarray,mask:np.ndarray,count:int=512)->np.ndarray:
    candidates=np.flatnonzero(mask); groups={key:candidates[keys[candidates]==key] for key in sorted(set(keys[candidates]))}; chosen=[]
    if count<len(groups) or count>len(candidates):raise ValueError("landmark count cannot preserve every observed stratum")
    target={key:count*len(indices)/len(candidates) for key,indices in groups.items()};quota={key:1 for key in groups}
    for _ in range(count-len(groups)):
        available=[key for key,indices in groups.items() if quota[key]<len(indices)];key=sorted(available,key=lambda value:(-(target[value]-quota[value]),value))[0];quota[key]+=1
    for key,indices in groups.items():
        order=sorted(indices,key=lambda i:hashlib.sha256(f"17:{i}".encode()).hexdigest());chosen.extend(order[:quota[key]])
    return np.asarray(chosen,dtype=np.int64)


def worker(experiment:Experiment,_device_id:int)->ExperimentResult:
    started=time.perf_counter(); p=experiment.parameters; atlas=np.load(p["atlas"]); rows=eligible_rows(Path("data/datasets/paws-wiki/labeled/dev.csv")); x,probability,success=carrier(atlas,rows); keys=stratum_keys(atlas["labels"],success); construction=np.isin(atlas["partition"],[1,2]); audit=atlas["partition"]==3; output=Path(p["output"]); output.mkdir(parents=True,exist_ok=True)
    if p["stage"]=="strata":
        payload={"schema_version":SCHEMA_VERSION,"complete":True,"carrier_dimension":x.shape[1],"construction_rows":int(construction.sum()),"audit_rows":int(audit.sum()),**strata_diagnostics(x,keys,construction,audit),"atlas_sha256":sha256(Path(p["atlas"]))}; path=output/"strata.json"; path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); metrics={"strata":float(len(payload["strata"])),"audit_pairs":float(payload["audit_cross_frontier_pairs"])}; primary=1.0
    else:
        indices=deterministic_landmarks(keys,construction); metrics,arrays=wavelet_basis(x,probability,keys,indices); complete=metrics["connected_components"]==1 and metrics["orthogonality_error"]<=1e-5; np.savez_compressed(output/"wavelet_basis.npz",**arrays); payload={"schema_version":SCHEMA_VERSION,"complete":complete,**metrics,"atlas_sha256":sha256(Path(p["atlas"])),"basis_sha256":sha256(output/"wavelet_basis.npz")}; path=output/"wavelets.json"; path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); primary=1-metrics["competence_reconstruction_error_128"]
    return ExperimentResult(experiment_id=experiment.id,hypothesis_id=p["hypothesis"],metrics={key:float(value) for key,value in metrics.items() if isinstance(value,(int,float))},primary_metric=primary,model_architecture=[x.shape[1]],model_parameters=0,training_time=time.perf_counter()-started,observations=[f"result={path}"],error=None if payload["complete"] else "construction gates failed")


async def run(stage:str,output:Path)->dict[str,Any]:
    hypothesis=f"paws-operational-{'whitney-strata' if stage=='strata' else 'competence-wavelets'}-v1"; experiment=Experiment(id=f"paws-{stage}",hypothesis_id=hypothesis,name=f"PAWS {stage}",parameters={"stage":stage,"hypothesis":hypothesis,"atlas":"data/experiments/paws_abc_routing/2026-08-16_experiment_06/competence_atlas.npz","output":str(output)}); runner=AsyncExperimentRunner(LabConfig(project_name=f"paws_{stage}",results_dir=str(output/"nal"),device_ids=[-1],max_parallel_experiments=1,min_experiments_per_hypothesis=1,require_statistical_significance=False,enable_wandb=False),worker); result=(await runner.run_experiments([experiment]))[0]; return {"complete":result.error is None,"metrics":result.metrics,"error":result.error}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("strata","wavelets"),required=True);parser.add_argument("--output",type=Path);args=parser.parse_args();number="08" if args.stage=="strata" else "09";output=args.output or Path(f"data/experiments/paws_abc_routing/2026-08-16_experiment_{number}");result=asyncio.run(run(args.stage,output));print(json.dumps(result,indent=2));raise SystemExit(0 if result["complete"] else 1)
if __name__=="__main__":main()
