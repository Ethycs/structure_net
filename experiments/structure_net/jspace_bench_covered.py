"""Strict common-relation follow-up with a vectorized, factored operation tensor."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import torch
from torch.nn import functional as F

from . import jspace_bench_tensor as base
from .jspace_bench_tensor_v2 import DEFAULT_ROOT as PARENT
from .jspace_bench_data import DATASETS, sha, write
from .jspace_groebner_tensor import PolynomialResidualMLP, features, graph_certificate


DEFAULT_ROOT=base.ROOT/"data/experiments/jspace_bench_tensor/20261113_covered_v1"
PROTOCOL=base.ROOT/"docs/07 - Status Reports/jspace/2026-11-13_jspace-bench-covered-plan.md"
FOLLOWUP={"selection":"all four task NRMSE <=0.5; exclude anchors", "threshold":.5,
          "arms":["factorized","dense_tensor","zero"],"reuses_evaluation_pool":True,
          "new_independent_confirmation":False,"runtime":"vectorized quadratic features",
          "known_before_freeze":"shared counts 20/34/19; v2 development budgets missed KL gate; no held-out accuracy inspected"}


class PackedResidualMLP(PolynomialResidualMLP):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        pairs=torch.triu_indices(len(self.anchor_positions),len(self.anchor_positions),device=self.anchor_positions.device)
        self.register_buffer("pair_i",pairs[0]);self.register_buffer("pair_j",pairs[1])

    def forward(self,x):
        hidden=self.act_fn(self.gate_proj(x))*self.up_proj(x)
        z=(hidden[...,self.anchor_positions].to(self.compute_dtype)-self.anchor_mean)/self.anchor_scale
        value=self.down_proj(hidden).to(self.compute_dtype)+self.output_bias
        quadratic=z[...,self.pair_i]*z[...,self.pair_j] if self.degree==2 else None
        if self.compiled:
            if self.degree==2:value+=F.linear(quadratic,self.quadratic_weight)
        else:
            pieces=[torch.ones_like(z[...,:1]),z]
            if quadratic is not None:pieces.append(quadratic)
            predicted=F.linear(torch.cat(pieces,-1),self.coefficient)*self.dropped_scale+self.dropped_mean
            value+=F.linear(predicted,self.dropped_weight)
        return value.to(x.dtype)


def prepare(root):
    base.cpu_only()
    if root.exists():raise FileExistsError(root)
    parent=base.verify(PARENT);base.checked(PARENT,"fit")
    extra=[Path(__file__),PROTOCOL,base.ROOT/"tests/structure_net/test_jspace_bench_covered.py"]
    value={**parent,"followup_config":FOLLOWUP,"parent_fit_fingerprint":parent["scientific_fingerprint"],
           "sources":{**parent["sources"],**{str(p):sha(p) for p in extra}},
           "input_sha256":{**parent["input_sha256"],**{str(PARENT/n):sha(PARENT/n) for n in ("plan.json","fit_manifest.json","fit.pt","fit.json")}}}
    value.pop("scientific_fingerprint");value["scientific_fingerprint"]=base.digest(value)
    root.mkdir(parents=True)
    for name in value["sources"]:
        p=Path(name);target=root/"producer"/p.relative_to(base.ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    shutil.copy2(PARENT/"tasks.json",root/"tasks.json")
    write(root/"plan.json",value)
    return {"fingerprint":value["scientific_fingerprint"],"followup":FOLLOWUP}


def select(fit,development):
    selected={};details={}
    for key,item in fit.items():
        data=development["activations"][key].float().cuda()
        z=(data-item["mean"].cuda())/item["scale"].cuda()
        predicted=features(z[:,item["anchors"]],2)@item["coefficients"]["2"].cuda().T
        errors=[]
        for family in DATASETS:
            mask=torch.tensor([f==family for f in development["families"]],device="cuda:0")
            errors.append(((predicted[mask]-z[mask]).square().mean(0)/z[mask].square().mean(0).clamp_min(1e-6)).sqrt())
        error=torch.stack(errors)
        valid=(error<=FOLLOWUP["threshold"]).all(0);valid[item["anchors"]]=False
        indices=valid.nonzero().flatten().cpu().tolist()
        if not indices:raise ValueError("No jointly covered component in selected layer")
        selected[key]=indices
        details[key]={"count":len(indices),"indices":indices,"maximum_task_nrmse":error[:,indices].max(1).values.cpu().tolist(),
                      "certificate":graph_certificate(item["coefficients"]["2"],item["anchors"],indices,2)}
    return selected,details


def run(root):
    if base.verify(root)["followup_config"]!=FOLLOWUP:raise ValueError("Follow-up configuration changed")
    stage=base.Stage(root,"covered")
    fit=torch.load(PARENT/"fit.pt",map_location="cpu",weights_only=True)
    captured=torch.load(PARENT/"capture/features.pt",map_location="cpu",weights_only=True)
    selected,details=select(fit,captured["development"])
    del captured;gc.collect();torch.cuda.empty_cache()
    write(root/"coverage.json",details)
    originals={k:stage.model.model.layers[k].mlp for k in base.CONFIG["layers"]}
    def install(arm):
        for layer in base.CONFIG["layers"]:
            stage.model.model.layers[layer].mlp.to("cpu")
            if arm=="baseline":replacement=originals[layer]
            else:replacement=PackedResidualMLP(originals[layer],fit[str(layer)],selected[str(layer)],
                    "zero" if arm=="zero" else "quadratic",compiled=arm!="factorized")
            stage.model.model.layers[layer].mlp=replacement.cuda()
        gc.collect();torch.cuda.empty_cache()
    baseline_dev=base.run_cases(stage,stage.tasks["development"],"baseline_development")
    install("factorized")
    changed_dev=base.run_cases(stage,stage.tasks["development"],"factorized_development")
    development={"passed":base.dev_pass(baseline_dev,changed_dev),"accuracy":base.accuracy(changed_dev),
                 "comparison":base.comparison(baseline_dev,changed_dev)}
    write(root/"development.json",development)
    install("baseline")
    original=base.run_cases(stage,stage.tasks["evaluation"],"baseline_evaluation")
    metrics={"baseline":base.accuracy(original)};timings={"baseline":base.timing(stage,"baseline_before")}
    for arm in FOLLOWUP["arms"]:
        install(arm)
        rows=base.run_cases(stage,stage.tasks["evaluation"],arm+"_evaluation")
        metrics[arm]={"accuracy":base.accuracy(rows),"comparison":base.comparison(original,rows),
                      "model_bytes":base.model_bytes(stage.model)}
        if arm!="zero":timings[arm]=base.timing(stage,arm)
        if arm=="factorized":
            base.save(root/"compressed_modules.pt",{str(k):{n:v.cpu() for n,v in stage.model.model.layers[k].mlp.state_dict().items()}
                                                   for k in base.CONFIG["layers"]})
    install("baseline");timings["baseline_after"]=base.timing(stage,"baseline_after")
    write(root/"evaluation.json",{"coverage":details,"development":development,"metrics":metrics,"timings":timings})
    return stage.finish(["coverage.json","development.json","evaluation.json","compressed_modules.pt"],followup_config=FOLLOWUP)


def analyze(root):
    base.cpu_only();plan=base.verify(root);base.checked(root,"covered")
    result=json.loads((root/"evaluation.json").read_text())
    base_acc=result["metrics"]["baseline"];new=result["metrics"]["factorized"]
    quality=(result["development"]["passed"] and new["accuracy"]["correct"]>=base_acc["correct"]-1
             and new["comparison"]["agreement"]>=base.CONFIG["evaluation_agreement"]
             and all(new["accuracy"]["by_task"][f]["correct"]>=base_acc["by_task"][f]["correct"]-1 for f in DATASETS))
    t=result["timings"];rate=(t["baseline"]["tokens_per_second"]+t["baseline_after"]["tokens_per_second"])/2
    speedup=t["factorized"]["tokens_per_second"]/rate
    reduction=1-new["model_bytes"]/t["baseline"]["resident_model_tensor_bytes"]
    value={"study_id":base.STUDY_ID,"followup":"strict_cross_task_coverage","status":"completed","confirmed":False,
           "scientific_fingerprint":plan["scientific_fingerprint"],"quality_gate":quality,"speedup":speedup,
           "model_tensor_byte_reduction":reduction,"result":result,
           "classification":"covered_quality_and_speed_pass" if quality and speedup>=1.02 and reduction>0 else
                            "covered_quality_preserved_no_speed_gate" if quality else "covered_quality_gate_failed",
           "scope":"exploratory follow-up on the same held-out pool; not an independent confirmation",
           "tensorization":"normal-form coefficient tensor retained as exact W_elim times polynomial-coefficient factors; no weight SVD"}
    write(root/"result.json",value);write(root/"campaign_results.json",{**value,"result_sha256":sha(root/"result.json")})
    return {k:v for k,v in value.items() if k!="result"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","run","analyze"));parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args()
    print(json.dumps({"prepare":prepare,"run":run,"analyze":analyze}[args.stage](args.root),allow_nan=False),flush=True)
