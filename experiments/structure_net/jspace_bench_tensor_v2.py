"""Versioned systems fix: cross-task Boolean overlap counting runs on CPU.

The frozen v1 scientific producer is unchanged. Reuse its successful capture,
retain its failed fit log, and declare the new producer in a new artifact root.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from . import jspace_bench_tensor as base
from .jspace_bench_data import sha, write


DEFAULT_ROOT = base.ROOT / "data/experiments/jspace_bench_tensor/20261113_v2"
AMENDMENT = base.ROOT / "docs/07 - Status Reports/jspace/2026-11-13_jspace-bench-tensor-systems-amendment.md"
_original_redundancy_report = base.redundancy_report


def redundancy_report(errors,separate_errors,config):
    return _original_redundancy_report(errors.cpu(),separate_errors.cpu(),config)


def prepare(root):
    base.cpu_only()
    if root.exists():raise FileExistsError(root)
    parent=base.verify(base.DEFAULT_ROOT)
    stages={name:base.checked(base.DEFAULT_ROOT,name) for name in ("shakedown","capture")}
    extras=[Path(__file__),AMENDMENT,base.ROOT/"tests/structure_net/test_jspace_bench_tensor_v2.py"]
    value={**parent,"sources":{**parent["sources"],**{str(p):sha(p) for p in extras}},
           "input_sha256":{**parent["input_sha256"],**{str(base.DEFAULT_ROOT/name):sha(base.DEFAULT_ROOT/name)
                for name in ("plan.json","shakedown_manifest.json","capture_manifest.json","fit_driver.log")}},
           "parent_fingerprint":parent["scientific_fingerprint"],"systems_revision":2,
           "systems_fix":"Only Boolean redundancy reporting tensors move to CPU; no changed fitting or selection rule",
           "created_at":datetime.now(timezone.utc).isoformat()}
    value.pop("scientific_fingerprint")
    value["scientific_fingerprint"]=base.digest(value)
    root.mkdir(parents=True)
    for name in value["sources"]:
        p=Path(name);target=root/"producer"/p.relative_to(base.ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    shutil.copy2(base.DEFAULT_ROOT/"tasks.json",root/"tasks.json")
    write(root/"plan.json",value)
    for name,original in stages.items():
        (root/name).symlink_to(base.DEFAULT_ROOT/name,target_is_directory=True)
        write(root/f"{name}_manifest.json",{**original,"fingerprint":value["scientific_fingerprint"],
            "reused_from":str(base.DEFAULT_ROOT/f"{name}_manifest.json"),
            "original_producer_fingerprint":parent["scientific_fingerprint"],"execution_reused":True})
    base.verify(root)
    return {"fingerprint":value["scientific_fingerprint"],"reused_stages":list(stages)}


def main(stage,root):
    base.redundancy_report=redundancy_report
    if stage=="fit":
        from structure_net_resources.client import require_gpu_lease
        import torch
        require_gpu_lease([0])
        probe=torch.tensor([[.1,.7],[.1,.1],[.1,.7],[.1,.7]],device="cuda:0")
        checked=redundancy_report(probe,probe,base.CONFIG)
        assert checked["redundant_occurrences"]==3 and checked["shared_by_all_tasks"]==1
    return {"prepare":prepare,"fit":base.fit,"evaluate":base.evaluate,"analyze":base.analyze}[stage](root)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","fit","evaluate","analyze"))
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args()
    print(json.dumps(main(args.stage,args.root),allow_nan=False),flush=True)
