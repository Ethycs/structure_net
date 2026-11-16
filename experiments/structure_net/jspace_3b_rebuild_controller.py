"""Serial reservation client for the bounded, native-first progressive pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
PIXI="/home/rabbit/.pixi/bin/pixi"
PROTOCOL=ROOT/"docs/07 - Status Reports/jspace/2026-11-16_qwen3b-progressive-response-rebuild.md"
MODULE="experiments.structure_net.jspace_3b_rebuild_"


def read(path): return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def selected_parent(stage):
    stage=Path(stage);p=read(stage/"plan.json");result=read(stage/"run/result.json")
    manifest=read(stage/"run/manifest.json")
    if (manifest.get("status")!="completed" or manifest.get("fingerprint")!=p["fingerprint"]
            or manifest.get("artifacts",{}).get("result.json")!=sha(stage/"run/result.json")
            or result.get("fingerprint")!=p["fingerprint"] or result.get("status")!="completed"):
        raise ValueError("Stage is missing authenticated completed evidence")
    if not result.get("relative_retention_pass"):
        return None
    selected=result.get("selected") or {}
    checkpoint=selected.get("checkpoint") or {}
    if not checkpoint.get("path") or sha(checkpoint["path"])!=checkpoint.get("sha256"):
        raise ValueError("Passed stage has no authenticated selected checkpoint")
    return checkpoint["path"]


def run(base):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ValueError("CPU orchestration requires CUDA hidden")
    base=Path(base).resolve();out=base/"controller";out.mkdir(exist_ok=False)
    paths=[Path(__file__),PROTOCOL,base/"data/plan.json"]
    paths += [ROOT/"experiments/structure_net"/("jspace_3b_rebuild_"+name+".py")
              for name in ("data","fit","fit_campaign","recovery")]
    pins={str(path):sha(path) for path in paths}
    write(out/"plan.json",dict(schema="jspace.3b-response-rebuild-controller.v1",pins=pins,
        cuts=[35,34],gpu="GPU-0370fdb0",allocation_mib=11264,allocator_mib=11008,
        all36_policy="Preserve existing worker and reservation; never signal it",confirmed=False))
    env=dict(os.environ,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",
        PIXI_CACHE_DIR="/tmp/structure-net-pixi-cache",UV_CACHE_DIR="/tmp/structure-net-uv-cache")
    def event(phase,**values):
        row=dict(utc=datetime.now(timezone.utc).isoformat(),phase=phase,**values)
        with (out/"progress.jsonl").open("a") as stream: stream.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)
    def verify_pins():
        if any(sha(path)!=digest for path,digest in pins.items()):
            raise ValueError("Prepared pilot controller or producing source changed")
    def invoke(label,module,phase,root,*,extra=(),minutes=None):
        verify_pins()
        command=[PIXI,"run","--locked","--offline","-e","ncm","python","-m",MODULE+module,
                 phase,"--root",str(root),*map(str,extra)]
        if minutes is not None:
            command += ["--limit-mib","11008"]
            command=["/usr/bin/python3","/usr/local/bin/inference-ctl","lease","run",
                "--allocation","GPU-0370fdb0=11264","--minutes",str(minutes),
                "--reason",f"Broad-response progressive pilot: {label}; all36 preserved","--",*command]
        event("launch",label=label,command=command)
        with (out/(label+".log")).open("x") as log:
            child=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        event("finished",label=label,returncode=child.returncode)
        if child.returncode:
            raise RuntimeError(f"{label} failed with code {child.returncode}; preserve its receipt and stop")
    start=time.monotonic();stages=[]
    try:
        event("waiting_for_progressive_transition")
        deadline=time.monotonic()+3*3600
        while not (base/"transition/result.json").exists():
            verify_pins()
            if time.monotonic()>deadline: raise TimeoutError("Progressive transition did not complete")
            time.sleep(20)
        transition=read(base/"transition/result.json")
        if transition["status"] not in ("worker_exited","worker_already_exited"):
            raise ValueError("Old progressive worker has not exited cleanly")
        # The reservation wrapper releases after its child exit. Allow its
        # finally handler to finish; actual admission still belongs to lease run.
        time.sleep(5)
        invoke("native_capture","data","capture",base/"data",minutes=60)
        parent=None
        for count,cut in enumerate((35,34),1):
            stage=base/f"blocks_{count:02d}"
            stage.mkdir(exist_ok=False)
            extra=["--data-root",base/"data","--cut",cut]
            if parent is not None: extra += ["--parent-checkpoint",parent]
            invoke(f"blocks{count}_fit_prepare","fit_campaign","prepare",stage/"fit",extra=extra)
            invoke(f"blocks{count}_fit","fit_campaign","run",stage/"fit",minutes=90)
            extra=["--data-root",base/"data","--fit-root",stage/"fit","--cut",cut]
            if parent is not None: extra += ["--parent-checkpoint",parent]
            invoke(f"blocks{count}_recovery_prepare","recovery","prepare",stage/"recovery",extra=extra)
            invoke(f"blocks{count}_pilot","recovery","pilot",stage/"recovery",minutes=60)
            invoke(f"blocks{count}_recovery","recovery","run",stage/"recovery",minutes=180)
            parent=selected_parent(stage/"recovery")
            stages.append(dict(cut=cut,root=str(stage),selected_checkpoint=parent,
                               relative_retention_pass=parent is not None))
            if parent is None:
                event("stopped_at_quality_gate",cut=cut)
                break
        write(out/"result.json",dict(status="completed",stages=stages,
            outcome="two_block_relative_retention" if len(stages)==2 and parent else "retention_not_established",
            elapsed_seconds=time.monotonic()-start,all36_untouched=True,confirmed=False))
    except BaseException as error:
        write(out/"failure.json",dict(status="execution_failed",error_type=type(error).__name__,error=str(error),
            stages=stages,elapsed_seconds=time.monotonic()-start,all36_untouched=True,confirmed=False))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--root",type=Path,required=True)
    run(parser.parse_args().root)
