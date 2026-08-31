#!/usr/bin/env python3
"""Experiment 10: gate-protected frozen PAWS A/B/C routing evaluation."""
from __future__ import annotations

import argparse, asyncio, json, math, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner
try:
    from experiments.structure_net.paws_competence_atlas import best_checkpoint, local_probabilities, qwen_dev
    from experiments.structure_net.paws_distillation import encode_rows
    from experiments.structure_net.paws_embedding_router import COSTS, lexical_features, lower_bound, metrics, route
    from experiments.structure_net.paws_teacher_annotation import eligible_rows, sha256
except ModuleNotFoundError:
    from paws_competence_atlas import best_checkpoint, local_probabilities, qwen_dev
    from paws_distillation import encode_rows
    from paws_embedding_router import COSTS, lexical_features, lower_bound, metrics, route
    from paws_teacher_annotation import eligible_rows, sha256

SCHEMA_VERSION="nal.paws-frozen-campaign.v1";HYPOTHESIS_ID="paws-frozen-end-to-end-routing-v1"


def completion_gates()->dict[str,bool]:
    checks={}
    paths={"02":Path("data/experiments/paws_abc_routing/2026-08-16_experiment_02/campaign_results.json"),"03":Path("data/experiments/paws_abc_routing/2026-08-16_experiment_03/summary.json"),"04":Path("data/experiments/paws_abc_routing/2026-08-16_experiment_04/campaign_results.json"),"05":Path("data/experiments/paws_abc_routing/2026-08-16_experiment_05/campaign_results.json"),"06":Path("data/experiments/paws_abc_routing/2026-08-16_experiment_06/summary.json"),"07":Path("data/experiments/paws_abc_routing/2026-08-16_experiment_07/router.json"),"08":Path("data/experiments/paws_abc_routing/2026-08-16_experiment_08/strata.json"),"09":Path("data/experiments/paws_abc_routing/2026-08-16_experiment_09/wavelets.json")}
    try:
        experiment_01=json.loads(Path("data/experiments/paws_abc_routing/2026-08-16_experiment_01/nal_result.json").read_text(encoding="utf-8"));checks["01"]=experiment_01["status"]=="completed" and experiment_01["primary_metric"]==1.0
    except (FileNotFoundError,KeyError,json.JSONDecodeError):checks["01"]=False
    for number,path in paths.items():
        try: checks[number]=bool(json.loads(path.read_text(encoding="utf-8"))["complete"])
        except (FileNotFoundError,KeyError,json.JSONDecodeError): checks[number]=False
    return checks


def predict_head(features:np.ndarray,checkpoint_path:Path)->np.ndarray:
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True);x=torch.from_numpy(features.astype(np.float32));head=torch.nn.Linear(x.shape[1],2);head.load_state_dict(checkpoint["head_state"]);head.eval()
    with torch.no_grad():return torch.softmax(head((x-checkpoint["mean"].float())/checkpoint["std"].float()),1).numpy()


def sigmoid(value:np.ndarray)->np.ndarray:return 1/(1+np.exp(-value))


def worker(experiment:Experiment,device_id:int)->ExperimentResult:
    started=time.perf_counter(); gates=completion_gates()
    if not all(gates.values()):raise RuntimeError(f"test reveal blocked by incomplete gates: {gates}")
    # This is intentionally the first test-file access in the program.
    rows=eligible_rows(Path("data/datasets/paws-wiki/labeled/test.csv"));output=Path(experiment.parameters["output"]);output.mkdir(parents=True,exist_ok=True);device=torch.device(f"cuda:{device_id}")
    feature_path=output/"test_features.npz"
    if not feature_path.exists():
        tiny=encode_rows("tinyllm",rows,device,64);smol=encode_rows("smollm",rows,device,16);np.savez_compressed(feature_path,tiny=tiny,smol=smol,groups=np.asarray([row["group_id"] for row in rows]))
    features=np.load(feature_path);tiny_checkpoint=best_checkpoint(Path("data/experiments/paws_abc_routing/2026-08-16_experiment_05/campaign_results.json"));smol_checkpoint=best_checkpoint(Path("data/experiments/paws_abc_routing/2026-08-16_experiment_04/campaign_results.json"));a_prob=predict_head(features["tiny"],tiny_checkpoint);b_prob=predict_head(features["smol"],smol_checkpoint);a=a_prob.argmax(1);b=b_prob.argmax(1)
    qwen=qwen_dev(rows,output/"qwen_test.jsonl",experiment.parameters["concurrency"]);groups=features["groups"];c=np.asarray([qwen[group]["teacher_label"] for group in groups]);labels=np.asarray([row["label"] for row in rows]);model_prediction=np.stack((a,b,c),axis=1);success=(model_prediction==labels[:,None]).astype(float)
    atlas=np.load("data/experiments/paws_abc_routing/2026-08-16_experiment_06/competence_atlas.npz");router_artifact=json.loads(Path("data/experiments/paws_abc_routing/2026-08-16_experiment_07/router.json").read_text());selected=router_artifact["selected"]
    atlas_success=np.stack((atlas["a_prediction"]==atlas["labels"],atlas["b_prediction"]==atlas["labels"],atlas["c_prediction"]==atlas["labels"]),axis=1).astype(float);train=atlas["partition"]==1
    components=np.asarray(router_artifact["pca_components"]);pca_mean=np.asarray(router_artifact["pca_mean"]);scale_mean=np.asarray(router_artifact["scaler_mean"]);scale=np.asarray(router_artifact["scaler_scale"]);z_train=((atlas["embedding"][train].astype(float)-pca_mean)@components.T-scale_mean)/scale;z_test=((features["smol"].astype(float)-pca_mean)@components.T-scale_mean)/scale
    support_distance=NearestNeighbors(n_neighbors=1).fit(z_train).kneighbors(z_test)[0][:,0];ood=support_distance>router_artifact["support_distance_p99"]
    if selected["family"]=="embedding_knn":
        distance,index=NearestNeighbors(n_neighbors=selected["k"]).fit(z_train).kneighbors(z_test);weights=np.ones_like(distance) if selected["weighting"]=="unweighted" else 1/(distance+1e-6);neighbors=atlas_success[train][index];probability=(weights[:,:,None]*neighbors).sum(1)/weights.sum(1)[:,None];routes=route(lower_bound(neighbors,weights),selected["tau"])
    else:
        lexical=np.stack([lexical_features(row["sentence1"],row["sentence2"]) for row in rows]);x=(lexical-np.asarray(router_artifact["lexical_scaler_mean"]))/np.asarray(router_artifact["lexical_scaler_scale"]);probability=np.stack([sigmoid(x@np.asarray(model["coef"])[0]+np.asarray(model["intercept"])[0]) for model in router_artifact["lexical_models"]],axis=1);routes=route(probability,selected["tau"])
    routes[ood]=2;routed=model_prediction[np.arange(len(rows)),routes];base_metrics=metrics(routes,success,probability);accuracy=float((routed==labels).mean());balanced=float(sum((routed[labels==value]==value).mean() for value in (0,1))/2)
    lexical=np.stack([lexical_features(row["sentence1"],row["sentence2"]) for row in rows]);overlap=lexical[:,0];displacement=lexical[:,-1];subgroups={}
    for name,mask in {"low_lexical_overlap":overlap<=np.quantile(overlap,.25),"high_lexical_overlap":overlap>=np.quantile(overlap,.75),"high_word_order_displacement":displacement>=np.quantile(displacement,.75)}.items():subgroups[name]={"rows":int(mask.sum()),"accuracy":float((routed[mask]==labels[mask]).mean())}
    records=[{"group_id":str(groups[i]),"label":int(labels[i]),"a_prediction":int(a[i]),"b_prediction":int(b[i]),"c_prediction":int(c[i]),"route":"ABC"[routes[i]],"routed_prediction":int(routed[i]),"ood_fallback":bool(ood[i])} for i in range(len(rows))];(output/"predictions.jsonl").write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in records),encoding="utf-8")
    payload={"schema_version":SCHEMA_VERSION,"complete":True,"gates":gates,"rows":len(rows),"accuracy":accuracy,"balanced_accuracy":balanced,"standalone_accuracy":{"A":float(success[:,0].mean()),"B":float(success[:,1].mean()),"C":float(success[:,2].mean())},**base_metrics,"ood_fallback_rate":float(ood.mean()),"within_one_point_of_c":accuracy>=float(success[:,2].mean())-.01,"cost_reduction_vs_c":1-base_metrics["mean_cost"]/COSTS[2],"subgroups":subgroups,"router_sha256":sha256(Path("data/experiments/paws_abc_routing/2026-08-16_experiment_07/router.json")),"wavelets_sha256":sha256(Path("data/experiments/paws_abc_routing/2026-08-16_experiment_09/wavelet_basis.npz")),"predictions_sha256":sha256(output/"predictions.jsonl"),"test_features_sha256":sha256(feature_path),"qwen_sha256":sha256(output/"qwen_test.jsonl")};(output/"summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return ExperimentResult(experiment_id=experiment.id,hypothesis_id=HYPOTHESIS_ID,metrics={"accuracy":accuracy,"balanced_accuracy":balanced,**base_metrics,"ood_fallback_rate":float(ood.mean())},primary_metric=accuracy,model_architecture=[],model_parameters=0,training_time=time.perf_counter()-started,observations=[f"summary={output/'summary.json'}"])


async def run(output:Path,concurrency:int)->dict[str,Any]:
    experiment=Experiment(id="paws-frozen-end-to-end",hypothesis_id=HYPOTHESIS_ID,name="PAWS frozen A/B/C routing",parameters={"output":str(output),"concurrency":concurrency});runner=AsyncExperimentRunner(LabConfig(project_name="paws_frozen_campaign",results_dir=str(output/"nal"),device_ids=[0],max_parallel_experiments=1,min_experiments_per_hypothesis=1,require_statistical_significance=False,enable_wandb=False),worker);result=(await runner.run_experiments([experiment]))[0];return {"complete":result.error is None,"metrics":result.metrics,"error":result.error}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=Path("data/experiments/paws_abc_routing/2026-08-16_experiment_10"));parser.add_argument("--concurrency",type=int,default=8);args=parser.parse_args();result=asyncio.run(run(args.output,args.concurrency));print(json.dumps(result,indent=2));raise SystemExit(0 if result["complete"] else 1)
if __name__=="__main__":main()
