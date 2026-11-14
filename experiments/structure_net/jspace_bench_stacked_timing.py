"""Counterbalanced timing of unchanged packed and factored implementations.

Known before this timing-only extension: packed and factored models both score
209/256; first rates are 12.34 and 11.49 tok/s, while the matched stacked dense
control is 11.95. Isolated packed/dense layers are nearly equal, so repeat timing
in reverse orders to check whether ordering explains the whole-model difference.
No accuracy, basis, weight, kernel, prompt or precision changes are permitted.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import torch

from . import jspace_bench_tensor as base
from .jspace_bench_stacked import DEFAULT_ROOT as PARENT, FIT_ROOT, COVERED_ROOT
from .jspace_bench_covered import PackedResidualMLP
from .jspace_packed_core import StackedCoreMLP
from .jspace_bench_data import sha, write


DEFAULT_ROOT = base.ROOT / "data/experiments/jspace_bench_tensor/20261113_stacked_timing_v1"
ORDER = ["previous_factorized", "stacked_polynomial_core", "stacked_dense_original",
         "stacked_dense_original", "stacked_polynomial_core", "previous_factorized"]


def prepare(root):
    base.cpu_only()
    if root.exists():raise FileExistsError(root)
    parent=base.verify(PARENT);base.checked(PARENT,"stacked")
    value={**parent,"timing_order":ORDER,"timing_scope":"counterbalanced repeat; no new accuracy evidence",
           "sources":{**parent["sources"],str(Path(__file__)):sha(__file__)},
           "input_sha256":{**parent["input_sha256"],str(PARENT/"result.json"):sha(PARENT/"result.json")}}
    value.pop("scientific_fingerprint");value["scientific_fingerprint"]=base.digest(value)
    root.mkdir(parents=True)
    for name in value["sources"]:
        path=Path(name);target=root/"producer"/path.relative_to(base.ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    shutil.copy2(PARENT/"tasks.json",root/"tasks.json")
    write(root/"plan.json",value)
    return {"fingerprint":value["scientific_fingerprint"],"order":ORDER}


def run(root):
    if base.verify(root)["timing_order"]!=ORDER:raise ValueError("Timing order changed")
    stage=base.Stage(root,"timing")
    fit=torch.load(FIT_ROOT/"fit.pt",map_location="cpu",weights_only=True)
    removed={int(k):v["indices"] for k,v in json.loads((COVERED_ROOT/"coverage.json").read_text()).items()}
    originals={i:layer.mlp for i,layer in enumerate(stage.model.model.layers)}
    measurements=[]
    for index,arm in enumerate(ORDER):
        for layer in originals:
            stage.model.model.layers[layer].mlp.to("cpu");original=originals[layer].cpu()
            if arm=="previous_factorized":
                replacement=PackedResidualMLP(original,fit[str(layer)],removed[layer],compiled=False) if layer in removed else original
            else:
                covered=arm=="stacked_polynomial_core" and layer in removed
                replacement=StackedCoreMLP(original,fit[str(layer)] if covered else None,removed[layer] if covered else [])
            stage.model.model.layers[layer].mlp=replacement.cuda()
        gc.collect();torch.cuda.empty_cache()
        value=base.timing(stage,f"{index}_{arm}")
        measurements.append({"arm":arm,"order_index":index,**value})
        print(json.dumps({"arm":arm,"index":index,"tokens_per_second":value["tokens_per_second"]}),flush=True)
    summaries={}
    for arm in set(ORDER):
        runs=[v for v in measurements if v["arm"]==arm]
        tokens=sum(s["new_tokens"] for v in runs for s in v["samples"])
        seconds=sum(s["seconds"] for v in runs for s in v["samples"])
        summaries[arm]={"tokens_per_second":tokens/seconds,"new_tokens":tokens,"seconds":seconds,
                        "run_rates":[v["tokens_per_second"] for v in runs]}
    result={"study_id":base.STUDY_ID,"status":"completed","confirmed":False,
            "scientific_fingerprint":stage.plan["scientific_fingerprint"],
            "scope":"same model/input paired engineering timing; no independent quality evidence",
            "measurements":measurements,"summary":summaries}
    write(root/"result.json",result)
    write(root/"campaign_results.json",{**result,"result_sha256":sha(root/"result.json")})
    return stage.finish(["result.json","campaign_results.json"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","run"));parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args()
    print(json.dumps({"prepare":prepare,"run":run}[args.stage](args.root),allow_nan=False),flush=True)
