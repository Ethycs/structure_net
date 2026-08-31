#!/usr/bin/env python3
"""Experiment 07: lexical and embedding-proximity A/B/C routers under NAL."""
from __future__ import annotations

import argparse, asyncio, json, time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner

try:
    from experiments.structure_net.paws_teacher_annotation import eligible_rows, sha256
except ModuleNotFoundError:
    from paws_teacher_annotation import eligible_rows, sha256

SCHEMA_VERSION="nal.paws-embedding-router.v1"; HYPOTHESIS_ID="paws-embedding-proximity-router-v1"; COSTS=np.asarray([1.0,360/51,8000/51])


def lexical_features(left:str,right:str)->np.ndarray:
    a=left.casefold().split(); b=right.casefold().split(); sa,sb=set(a),set(b); union=sa|sb; shared=sa&sb
    positions={token:index/max(1,len(b)-1) for index,token in enumerate(b)}
    displacement=np.mean([abs(index/max(1,len(a)-1)-positions[token]) for index,token in enumerate(a) if token in positions]) if shared else 1.0
    return np.asarray([len(shared)/max(1,len(union)),min(len(a),len(b))/max(1,max(len(a),len(b))),abs(len(a)-len(b)),len(shared)/max(1,min(len(a),len(b))),displacement],dtype=np.float64)


def lower_bound(success:np.ndarray,weights:np.ndarray)->np.ndarray:
    total=weights.sum(1); probability=np.clip((weights[:,:,None]*success).sum(1)/total[:,None],0,1); effective=total**2/(weights**2).sum(1)
    return np.clip(probability-1.96*np.sqrt(probability*(1-probability)/effective[:,None]),0,1)


def route(lcb:np.ndarray,tau:float)->np.ndarray:
    result=np.full(len(lcb),2,dtype=np.int8)
    result[lcb[:,1]>=tau]=1; result[lcb[:,0]>=tau]=0
    return result


def metrics(routes:np.ndarray,success:np.ndarray,probability:np.ndarray)->dict[str,float]:
    index=np.arange(len(routes)); correct=success[index,routes]; oracle=np.asarray([next((m for m in range(3) if row[m]),2) for row in success])
    confidence=probability[index,routes]; ece=sum(abs(correct[mask].mean()-confidence[mask].mean())*mask.mean() for low in np.linspace(0,1,10,endpoint=False) if (mask:=(confidence>=low)&(confidence<low+.1)).any())
    return {"accuracy":float(correct.mean()),"mean_cost":float(COSTS[routes].mean()),"escalation_rate":float((routes>0).mean()),"routing_regret":float((COSTS[routes]-COSTS[oracle]).mean()),"ece":float(ece)}


def worker(experiment:Experiment,_device_id:int)->ExperimentResult:
    started=time.perf_counter(); p=experiment.parameters; atlas=np.load(p["atlas"]); rows=eligible_rows(Path("data/datasets/paws-wiki/labeled/dev.csv"))
    groups=np.asarray([row["group_id"] for row in rows]); assert np.array_equal(groups,atlas["groups"])
    success=np.stack([atlas["a_prediction"]==atlas["labels"],atlas["b_prediction"]==atlas["labels"],atlas["c_prediction"]==atlas["labels"]],axis=1).astype(float)
    train=atlas["partition"]==1; validation=atlas["partition"]==2
    pca=PCA(n_components=32,random_state=17).fit(atlas["embedding"][train].astype(np.float32)); scaler=StandardScaler().fit(pca.transform(atlas["embedding"][train].astype(np.float32)))
    z_train=scaler.transform(pca.transform(atlas["embedding"][train].astype(np.float32))); z_validation=scaler.transform(pca.transform(atlas["embedding"][validation].astype(np.float32)))
    candidates=[]
    for k in (5,15,31,63):
        distance,index=NearestNeighbors(n_neighbors=k).fit(z_train).kneighbors(z_validation); neighbor_success=success[train][index]
        for weighting in ("unweighted","inverse_distance"):
            weights=np.ones_like(distance) if weighting=="unweighted" else 1/(distance+1e-6); total=weights.sum(1); probability=(weights[:,:,None]*neighbor_success).sum(1)/total[:,None]; lcb=lower_bound(neighbor_success,weights)
            for tau in (.80,.85,.90,.95):
                routes=route(lcb,tau); candidates.append({"family":"embedding_knn","k":k,"weighting":weighting,"tau":tau,**metrics(routes,success[validation],probability)})
    lexical=np.stack([lexical_features(row["sentence1"],row["sentence2"]) for row in rows]); lexical_scaler=StandardScaler().fit(lexical[train]); x_train=lexical_scaler.transform(lexical[train]); x_validation=lexical_scaler.transform(lexical[validation]); lexical_probability=[]; lexical_models=[]
    for model in range(3):
        fitted=LogisticRegression(random_state=17,max_iter=1000).fit(x_train,success[train,model]); lexical_models.append(fitted); lexical_probability.append(fitted.predict_proba(x_validation)[:,1])
    lexical_probability=np.stack(lexical_probability,axis=1)
    for tau in (.80,.85,.90,.95): candidates.append({"family":"lexical_logistic","tau":tau,**metrics(route(lexical_probability,tau),success[validation],lexical_probability)})
    always_c=float(success[validation,2].mean()); feasible=[item for item in candidates if item["accuracy"]>=always_c-.01]
    selected=sorted(feasible,key=lambda item:(item["mean_cost"],-item["accuracy"],item["ece"],json.dumps(item,sort_keys=True)))[0] if feasible else sorted(candidates,key=lambda item:(-item["accuracy"],item["mean_cost"],item["ece"]))[0]
    output=Path(p["output"]); output.mkdir(parents=True,exist_ok=True)
    support_distance=NearestNeighbors(n_neighbors=2).fit(z_train).kneighbors(z_train)[0][:,1]
    artifact={"schema_version":SCHEMA_VERSION,"complete":True,"always_c_accuracy":always_c,"selection_floor":always_c-.01,"selected":selected,"candidates":candidates,"atlas_sha256":sha256(Path(p["atlas"])),"pca_components":pca.components_.tolist(),"pca_mean":pca.mean_.tolist(),"scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist(),"support_distance_p99":float(np.quantile(support_distance,.99)),"lexical_scaler_mean":lexical_scaler.mean_.tolist(),"lexical_scaler_scale":lexical_scaler.scale_.tolist(),"lexical_models":[{"coef":model.coef_.tolist(),"intercept":model.intercept_.tolist()} for model in lexical_models]}
    (output/"router.json").write_text(json.dumps(artifact,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return ExperimentResult(experiment_id=experiment.id,hypothesis_id=HYPOTHESIS_ID,metrics={key:selected[key] for key in ("accuracy","mean_cost","ece","escalation_rate","routing_regret")},primary_metric=selected["accuracy"],model_architecture=[32,3],model_parameters=0,training_time=time.perf_counter()-started,observations=[f"router={output/'router.json'}"])


async def run(output:Path)->dict[str,Any]:
    experiment=Experiment(id="paws-embedding-proximity-router",hypothesis_id=HYPOTHESIS_ID,name="PAWS embedding proximity router",parameters={"atlas":"data/experiments/paws_abc_routing/2026-08-16_experiment_06/competence_atlas.npz","output":str(output)})
    runner=AsyncExperimentRunner(LabConfig(project_name="paws_embedding_router",results_dir=str(output/"nal"),device_ids=[-1],max_parallel_experiments=1,min_experiments_per_hypothesis=1,require_statistical_significance=False,enable_wandb=False),worker)
    result=(await runner.run_experiments([experiment]))[0]; return {"complete":result.error is None,"metrics":result.metrics,"error":result.error}


def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=Path("data/experiments/paws_abc_routing/2026-08-16_experiment_07")); args=parser.parse_args(); result=asyncio.run(run(args.output)); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["complete"] else 1)
if __name__=="__main__":main()
