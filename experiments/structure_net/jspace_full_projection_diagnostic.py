"""CPU diagnostic: Jacobian trace retention versus data-weighted remainder."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from . import jspace_full_capture as base
from .jspace_full_fit_v2 import DEFAULT_ROOT as FIT
from .jspace_bench_data import sha,write

ROOT=base.ROOT/"data/experiments/jspace_full_compact/20261113_projection_diagnostic_v1"


def run():
    base.cpu_only();torch.set_num_threads(4)
    if ROOT.exists():raise FileExistsError(ROOT)
    source=base.DEFAULT_ROOT/"capture";ROOT.mkdir()
    inputs=[FIT/"fit.pt",source/"statistics.pt",source/"items.json"]
    write(ROOT/"plan.json",{"scope":"posthoc CPU proxy on 128 calibration final-token states, not a finite neural intervention",
        "known":"v1 affine value-mean omission; corrected v2 adaptation running",
        "source_sha256":{str(Path(__file__)):sha(__file__)},"input_sha256":{str(p):sha(p) for p in inputs}})
    shutil.copy2(__file__,ROOT/"producer.py")
    fit=torch.load(FIT/"fit.pt",map_location="cpu",weights_only=True)
    stats=torch.load(source/"statistics.pt",map_location="cpu",weights_only=True)
    meta=json.loads((source/"items.json").read_text());indices=[meta["items"][i]["stop"]-1 for i in meta["calibration_indices"]]
    u=fit["basis"][:,:1024];rows=[];boundaries={}
    for i in range(36):
        path=source/f"boundary_{i:02d}.npy"
        # Parent capture manifest supplies provenance for the large unchanged arrays.
        manifest=json.loads((base.DEFAULT_ROOT/"capture_manifest.json").read_text())
        boundaries[str(path)]=manifest["artifact_sha256"][str(path.relative_to(base.DEFAULT_ROOT))]
        x=base.read_bits(np.load(path,mmap_mode="r"),indices,"cpu").float()-stats["mean"][i]
        residual=x-(x@u)@u.T;g=stats["grams"][i]
        weighted=((residual@g)*residual).sum();total=((x@g)*x).sum()
        if weighted<0 or total<=0:raise ValueError("Invalid PSD quadratic form")
        entry=fit["layers"][str(i)];group=entry["attention_group"]
        mean=stats["value_sum"][i,group]/stats["count"];s=entry["value_basis"]
        mean_remainder=mean-s@(s.T@mean)
        rows.append({"layer":i,"input_remainder_energy_fraction":float(residual.square().sum()/x.square().sum()),
            "jacobian_weighted_remainder_ratio":float(weighted/total),
            "discarded_value_mean_energy_fraction":float(mean_remainder.square().sum()/mean.square().sum().clamp_min(1e-12))})
    value={"scope":"Linearized quadratic-form proxy using each layer's pooled task-Jacobian Gramian and calibration final-token inputs",
        "definition":"ratio = sum(delta^T G delta) / sum((h-mu)^T G (h-mu)); delta=(I-UU^T)(h-mu)",
        "rows":rows,"boundary_sha256_from_capture_manifest":boundaries,"capture_manifest_sha256":sha(base.DEFAULT_ROOT/"capture_manifest.json")}
    write(ROOT/"result.json",value)
    print(json.dumps({"root":str(ROOT),"weighted_remainder_min":min(r["jacobian_weighted_remainder_ratio"] for r in rows),
        "weighted_remainder_max":max(r["jacobian_weighted_remainder_ratio"] for r in rows)},allow_nan=False))


if __name__=="__main__":run()
