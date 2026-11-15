"""CPU-only watcher: accepted full36 campaign -> reserved final gate, without retries."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
MODULE="experiments.structure_net.jspace_3b_final_gate"
PHASES=("compact_export","compact_validate","native_eager_before","native_sdpa_before","compact","native_sdpa_after")
SOURCES=(Path(__file__),ROOT/"experiments/structure_net/jspace_3b_final_gate.py",
    ROOT/"experiments/structure_net/jspace_3b_final_evaluate.py",
    ROOT/"experiments/structure_net/jspace_allblocks_artifact.py",
    ROOT/"experiments/structure_net/jspace_qwen_parallel_evaluate.py",
    ROOT/"experiments/structure_net/jspace_recovery_data.py",
    ROOT/"tests/structure_net/test_jspace_3b_completion.py",
    ROOT/"tests/structure_net/test_jspace_3b_final_gate.py",
    ROOT/"tests/structure_net/test_jspace_3b_final_evaluate.py",
    ROOT/"docs/07 - Status Reports/jspace/2026-11-15_qwen3b-full-model-gate.md",ROOT/"pyproject.toml",ROOT/"pixi.lock")
read=lambda p:json.loads(Path(p).read_text())
def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()
def write(path,value):
    with Path(path).open("x") as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")
def event(state,kind,**values):
    with (Path(state)/"events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(time=time.time(),pid=os.getpid(),kind=kind,**values),allow_nan=False)+"\n")
def cpu_env():
    return {**os.environ,"CUDA_VISIBLE_DEVICES":"","CUDA_DEVICE_ORDER":"PCI_BUS_ID",
        "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
        "PIXI_CACHE_DIR":"/tmp/structure-net-pixi-cache","UV_CACHE_DIR":"/tmp/structure-net-uv-cache"}
def pixi(*args):return ["pixi","run","--locked","--offline","-e","ncm","python",*args]
def prepare(state,campaign,final_root,*,poll_seconds=15,stale_hours=3):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Supervisor preparation must hide CUDA")
    if not 1<=poll_seconds<=15 or not math.isfinite(stale_hours) or stale_hours<3:raise ValueError("Poll1..15s and stale timeout>=3h required")
    state,campaign,final_root=(Path(p).resolve() for p in (state,campaign,final_root))
    if state.exists():raise FileExistsError(state)
    sources=dict(read(campaign/"plan.json").get("source_sha256",{}))
    sources.update({str(p.resolve()):sha(p) for p in SOURCES})
    for path,digest in sources.items():
        if sha(path)!=digest:raise ValueError("Prepared campaign source changed: "+path)
    configuration=dict(schema="jspace.3b-completion.v1",campaign=str(campaign),final_root=str(final_root),
        campaign_plan_sha256=sha(campaign/"plan.json"),poll_seconds=poll_seconds,stale_seconds=3600*stale_hours,
        created_at=time.time(),source_sha256=sources)
    state.mkdir(parents=True);(state/"logs").mkdir();write(state/"plan.json",configuration)
    write(state/"plan.sha256.json",{"sha256":sha(state/"plan.json")})
    return dict(status="prepared",state_root=str(state))
def verify(state):
    state=Path(state)
    if sha(state/"plan.json")!=read(state/"plan.sha256.json")["sha256"]:raise ValueError("Supervisor launch configuration changed")
    plan=read(state/"plan.json")
    if plan.get("schema")!="jspace.3b-completion.v1" or sha(Path(plan["campaign"])/"plan.json")!=plan["campaign_plan_sha256"]:
        raise ValueError("Campaign/launch identity changed")
    for path,digest in plan["source_sha256"].items():
        if sha(path)!=digest:raise ValueError("Prepared final producer changed: "+path)
    return plan

def observe(plan,started,*,now=None):
    root=Path(plan["campaign"]);now=time.time() if now is None else now
    for path in sorted(root.glob("stages/*/*/failure.json")):
        try:failure=read(path)
        except json.JSONDecodeError:continue  # A live writer may not have closed its receipt yet.
        return dict(status="campaign_execution_failure",failure_path=str(path),failure=failure)
    outcome=root/"outcome.json"
    if outcome.exists():
        try:value=read(outcome)
        except json.JSONDecodeError:value=None
        if value is not None:
            if value.get("status")=="all_blocks_recovered" and value.get("last_accepted_cut")==0 and value.get("compressed_blocks")==36:
                return dict(status="ready",outcome=value)
            return dict(status="quality_stop" if value.get("status")=="quality_stop" else "campaign_terminal_not_full36",outcome=value)
    progress=root/"progress.jsonl";last=max(started,progress.stat().st_mtime if progress.exists() else started)
    if now-last>=plan["stale_seconds"]:return dict(status="stalled",seconds_without_progress=now-last,quality_failure=False)
    return None

def child(state,label,command):
    verify(state);log=Path(state)/"logs"/f"{time.time_ns()}_{label}.log"
    event(state,"child_started",label=label,command=command,log=str(log))
    with log.open("xb") as stream:
        result=subprocess.run(command,cwd=ROOT,env=cpu_env(),stdout=stream,stderr=subprocess.STDOUT,check=False)
    event(state,"child_finished",label=label,returncode=result.returncode,log=str(log))
    if result.returncode:raise RuntimeError(f"{label} exited{result.returncode}; log={log}")

def validate_final(state,phase=None):
    plan=verify(state)
    script=("import sys; from experiments.structure_net import jspace_3b_final_gate as m; "
        "p=m.verify(sys.argv[1]); assert p['campaign_root']==sys.argv[2]; "
        "m.checked(sys.argv[1],sys.argv[3],p) if len(sys.argv)>3 else None")
    child(state,"validate_"+(phase or "plan"),pixi("-c",script,plan["final_root"],plan["campaign"],*([phase] if phase else [])))

def finish(state):
    plan=verify(state);final=Path(plan["final_root"])
    if not final.exists():
        child(state,"prepare",pixi("-m",MODULE,"prepare","--root",str(final),"--campaign",plan["campaign"]))
    elif not (final/"plan.json").exists():raise RuntimeError("Incomplete final preparation; preserve it and use a new root")
    validate_final(state)
    for phase in PHASES:
        path=final/phase
        if (path/"manifest.json").exists():validate_final(state,phase);continue
        if path.exists():raise RuntimeError("Incomplete final phase cannot be retried: "+str(path))
        command=["/usr/bin/python3","/usr/local/bin/inference-ctl","lease","run","--allocation","GPU-0370fdb0=11264",
            "--minutes","120","--reason",f"JSPACE-QWEN3B-FULLMODEL-GATE-01 {phase}","--",
            *pixi("-m",MODULE,phase,"--root",str(final))]
        child(state,phase,command);validate_final(state,phase)
    if not (final/"result.json").exists():child(state,"finalize",pixi("-m",MODULE,"finalize","--root",str(final)))
    result=read(final/"result.json");final_plan=read(final/"plan.json")
    if result.get("status")!="completed" or result.get("fingerprint")!=final_plan["fingerprint"]:
        raise ValueError("Final aggregate is incomplete or mismatched")
    return dict(status="completed",final_result=str(final/"result.json"),final_result_sha256=sha(final/"result.json"),
        classification=result.get("classification"),confirmed=False)

def watch(state):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Watcher must hide CUDA")
    state=Path(state);plan=verify(state)
    with (state/"watcher.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.seek(0);lock.truncate();lock.write(str(os.getpid())+"\n");lock.flush()
        if (state/"status.json").exists():return read(state/"status.json")
        event(state,"watcher_started",campaign=plan["campaign"])
        started=time.time()
        try:
            while True:
                outcome=observe(plan,started)
                if outcome is not None:break
                time.sleep(plan["poll_seconds"])
            event(state,"campaign_observed",**outcome)
            result=finish(state) if outcome["status"]=="ready" else outcome
        except Exception as error:
            result=dict(status="completion_execution_failure",error_type=type(error).__name__,error=str(error),quality_failure=False)
        result.update(pid=os.getpid(),finished_at=time.time());write(state/"status.json",result)
        event(state,"watcher_finished",result=result);return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=("prepare","watch"));parser.add_argument("--state-root",required=True,type=Path)
    parser.add_argument("--campaign",type=Path);parser.add_argument("--final-root",type=Path);parser.add_argument("--poll-seconds",type=float,default=15)
    parser.add_argument("--stale-hours",type=float,default=3);args=parser.parse_args()
    result=prepare(args.state_root,args.campaign,args.final_root,poll_seconds=args.poll_seconds,stale_hours=args.stale_hours) if args.command=="prepare" else watch(args.state_root)
    print(json.dumps(result,allow_nan=False),flush=True)
    if result["status"] in ("completion_execution_failure","campaign_execution_failure","stalled"):raise SystemExit(1)
