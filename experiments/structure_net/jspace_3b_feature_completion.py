"""Durable CPU watcher for the accepted feature/resident full36 final gate.

The watcher never trains, opens confirmation data or initializes CUDA. It only
launches frozen final-gate commands; model work receives normal reservations.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from . import jspace_3b_completion as previous

ROOT=Path(__file__).resolve().parents[2]
SCHEMA="jspace.3b-feature-completion.v1"
CAMPAIGN_SCHEMA="jspace.3b-feature-resident-campaign.v1"
MODULE="experiments.structure_net.jspace_3b_feature_resident_final_gate"
SMOKE_MODULE="experiments.structure_net.jspace_3b_infer"
SMOKE=dict(prompt="Reply with one short sentence describing what a tree is.",max_new_tokens=16,
    mode="pinned_chat_template",scope="Tokenize, generate and decode interface smoke only; no quality or speed claim")
PHASES=previous.PHASES
read,sha,write,event,pixi=previous.read,previous.sha,previous.write,previous.event,previous.pixi
observe=previous.observe
SOURCES=(Path(__file__),Path(previous.__file__),
    ROOT/"experiments/structure_net/jspace_3b_feature_resident_final_gate.py",
    ROOT/"experiments/structure_net/jspace_3b_final_gate.py",
    ROOT/"experiments/structure_net/jspace_3b_final_evaluate.py",
    ROOT/"experiments/structure_net/jspace_allblocks_artifact.py",
    ROOT/"experiments/structure_net/jspace_qwen_parallel_evaluate.py",
    ROOT/"experiments/structure_net/jspace_recovery_data.py",
    ROOT/"experiments/structure_net/jspace_3b_infer.py",
    ROOT/"tests/structure_net/test_jspace_3b_feature_completion.py",
    ROOT/"tests/structure_net/test_jspace_3b_feature_resident_final_gate.py",
    ROOT/"tests/structure_net/test_jspace_3b_final_gate.py",
    ROOT/"tests/structure_net/test_jspace_3b_final_evaluate.py",
    ROOT/"tests/structure_net/test_jspace_3b_infer.py",
    ROOT/"docs/07 - Status Reports/jspace/2026-11-15_qwen3b-parent-feature-resident-continuation.md",
    ROOT/"docs/07 - Status Reports/jspace/2026-11-15_qwen3b-full-model-gate.md",
    ROOT/"pyproject.toml",ROOT/"pixi.lock")


def cpu_env():
    return {**previous.cpu_env(),"PYTHONUNBUFFERED":"1"}


def prepare(state,campaign,final_root,*,poll_seconds=15,stale_hours=3):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Supervisor preparation must hide CUDA")
    if not 1<=poll_seconds<=15 or not math.isfinite(stale_hours) or stale_hours<3:
        raise ValueError("Poll1..15s and stale timeout>=3h required")
    state,campaign,final_root=(Path(p).resolve() for p in (state,campaign,final_root))
    if state.exists():raise FileExistsError(state)
    if len({state,campaign,final_root})!=3:raise ValueError("Watcher, campaign and final roots must be distinct")
    source=read(campaign/"plan.json")
    if source.get("schema")!=CAMPAIGN_SCHEMA or not source.get("fingerprint"):
        raise ValueError("Prepared feature/resident campaign required")
    sources=dict(source["source_sha256"])
    for path,digest in sources.items():
        if sha(path)!=digest:raise ValueError("Pinned campaign source changed: "+path)
    for path in SOURCES:
        path=path.resolve();digest=sha(path)
        if str(path) in sources and sources[str(path)]!=digest:
            raise ValueError("Final producer conflicts with pinned campaign source")
        sources[str(path)]=digest
    plan=dict(schema=SCHEMA,campaign=str(campaign),final_root=str(final_root),campaign_schema=CAMPAIGN_SCHEMA,
        campaign_fingerprint=source["fingerprint"],campaign_plan_sha256=sha(campaign/"plan.json"),
        final_module=MODULE,phases=list(PHASES),raw_text_smoke=SMOKE,allocation="GPU-0370fdb0=11264",lease_minutes=120,
        poll_seconds=poll_seconds,stale_seconds=3600*stale_hours,created_at=time.time(),source_sha256=sources,
        admission="Frozen feature final preparation authenticates all36 accepted checkpoints and seven phases before confirmation reads",
        scope="Completion does not imply passing quality, storage or runtime gates",confirmed=False)
    state.mkdir(parents=True);(state/"logs").mkdir()
    for path,digest in sources.items():
        target=state/"producer"/Path(path).relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        if sha(target)!=digest:raise ValueError("Watcher producer archive changed during preparation")
    write(state/"plan.json",plan);write(state/"plan.sha256.json",dict(sha256=sha(state/"plan.json")))
    return dict(status="prepared",state_root=str(state),campaign_fingerprint=source["fingerprint"])


def verify(state):
    state=Path(state)
    if sha(state/"plan.json")!=read(state/"plan.sha256.json")["sha256"]:
        raise ValueError("Watcher launch configuration changed")
    plan=read(state/"plan.json")
    if (plan.get("schema")!=SCHEMA or plan.get("campaign_schema")!=CAMPAIGN_SCHEMA or
            plan.get("final_module")!=MODULE or plan.get("phases")!=list(PHASES) or plan.get("raw_text_smoke")!=SMOKE or
            plan.get("allocation")!="GPU-0370fdb0=11264" or plan.get("lease_minutes")!=120 or
            sha(Path(plan["campaign"])/"plan.json")!=plan["campaign_plan_sha256"]):
        raise ValueError("Campaign/final-gate identity or reservation changed")
    for path,digest in plan["source_sha256"].items():
        if sha(path)!=digest or sha(state/"producer"/Path(path).relative_to(ROOT))!=digest:
            raise ValueError("Frozen watcher/final producer changed: "+path)
    return plan


def child(state,label,command):
    verify(state);log=Path(state)/"logs"/f"{time.time_ns()}_{label}.log"
    event(state,"child_started",label=label,command=command,log=str(log))
    with log.open("xb") as stream:
        result=subprocess.run(command,cwd=ROOT,env=cpu_env(),stdout=stream,stderr=subprocess.STDOUT,check=False)
    event(state,"child_finished",label=label,returncode=result.returncode,log=str(log))
    if result.returncode:raise RuntimeError(f"{label} exited {result.returncode}; log={log}")
    return log


def reserved_command(plan,label,command):
    return ["/usr/bin/python3","/usr/local/bin/inference-ctl","lease","run","--allocation",plan["allocation"],
        "--minutes",str(plan["lease_minutes"]),"--reason",f"JSPACE-3B-FEATURE-FINAL-GATE {label}","--",*command]


def validate_smoke_result(value,*,fingerprint):
    ids=value.get("generated_token_ids",[]);reservation=value.get("reservation",{})
    if (value.get("source_fingerprint")!=fingerprint or value.get("device")!="cuda:0" or value.get("greedy") is not True or
            value.get("position_capacity")!=416 or type(value.get("input_tokens")) is not int or value["input_tokens"]<=0 or
            not isinstance(value.get("text"),str) or not 1<=len(ids)<=SMOKE["max_new_tokens"] or
            any(type(token) is not int or token<0 for token in ids) or value.get("generated_tokens")!=len(ids) or
            value["input_tokens"]+SMOKE["max_new_tokens"]>416 or
            not str(reservation.get("physical_uuid","")).startswith("GPU-0370fdb0") or not reservation.get("lease_id") or
            reservation.get("lease_bytes")!=11264*1024**2 or reservation.get("allocator_limit_bytes")!=11008*1024**2):
        raise ValueError("Reserved raw-text smoke output or bundle identity invalid")
    return value


def raw_text_smoke(state):
    plan=verify(state);state=Path(state);output=state/"raw_text_smoke"
    final=Path(plan["final_root"]);final_plan=read(final/"plan.json");bundle=final/"compact_export/model"
    validate_final(state,"compact_validate")
    bundle_sha=sha(bundle/"manifest.json")
    if (output/"manifest.json").exists():
        receipt=read(output/"manifest.json")
        if (receipt.get("status")!="completed" or receipt.get("watcher_plan_sha256")!=sha(state/"plan.json") or
                receipt.get("bundle_manifest_sha256")!=bundle_sha or receipt.get("configuration")!=SMOKE or
                receipt.get("result_sha256")!=sha(output/"result.json") or receipt.get("log_sha256")!=sha(receipt["log"])):
            raise ValueError("Completed raw-text smoke provenance changed")
        return validate_smoke_result(read(output/"result.json"),fingerprint=final_plan["fingerprint"])
    if output.exists():raise RuntimeError("Incomplete raw-text smoke cannot be retried; preserve its evidence")
    output.mkdir()
    command=reserved_command(plan,"raw_text_smoke",pixi("-m",SMOKE_MODULE,"--bundle",str(bundle),
        "--prompt",SMOKE["prompt"],"--max-new-tokens",str(SMOKE["max_new_tokens"]),"--device","cuda:0","--json"))
    log=child(state,"raw_text_smoke",command)
    value=None
    for line in reversed(log.read_text().splitlines()):
        try:parsed=json.loads(line)
        except json.JSONDecodeError:continue
        if isinstance(parsed,dict) and "generated_token_ids" in parsed:value=parsed;break
    if value is None:raise ValueError("Raw-text CLI did not emit a JSON generation receipt")
    validate_smoke_result(value,fingerprint=final_plan["fingerprint"])
    if sha(bundle/"manifest.json")!=bundle_sha:raise ValueError("Bundle changed during raw-text smoke")
    write(output/"result.json",value)
    write(output/"manifest.json",dict(status="completed",watcher_plan_sha256=sha(state/"plan.json"),configuration=SMOKE,
        bundle_manifest_sha256=bundle_sha,result_sha256=sha(output/"result.json"),log=str(log),log_sha256=sha(log)))
    return value


def validate_final(state,phase=None):
    plan=verify(state)
    # The explicit wrapper owns admission; checked() remains on its unchanged
    # numerical module. Do not dispatch through the old campaign's admission.
    script=("import sys; from experiments.structure_net import jspace_3b_feature_resident_final_gate as m; "
        "p=m.verify(sys.argv[1]); assert p['campaign_root']==sys.argv[2]; "
        "assert p['campaign_fingerprint']==sys.argv[3]; "
        "m.numerical.checked(sys.argv[1],sys.argv[4],p) if len(sys.argv)>4 else None")
    child(state,"validate_"+(phase or "plan"),pixi("-c",script,plan["final_root"],plan["campaign"],
        plan["campaign_fingerprint"],*([phase] if phase else [])))


def finish(state):
    plan=verify(state);final=Path(plan["final_root"])
    ready=observe(plan,time.time())
    if not ready or ready["status"]!="ready":raise ValueError("Full36 accepted outcome required before final preparation")
    if not final.exists():
        # This CPU command checks the full accepted chain before it can open the
        # untouched confirmation population. A marker alone is insufficient.
        child(state,"prepare",pixi("-m",MODULE,"prepare","--root",str(final),"--campaign",plan["campaign"]))
    elif not (final/"plan.json").exists():
        raise RuntimeError("Incomplete final preparation; preserve it and use a new root")
    validate_final(state)
    for phase in PHASES:
        if phase=="native_eager_before":raw_text_smoke(state)
        path=final/phase
        if (path/"manifest.json").exists():validate_final(state,phase);continue
        if path.exists():raise RuntimeError("Incomplete final phase cannot be retried: "+str(path))
        command=reserved_command(plan,phase,pixi("-m",MODULE,phase,"--root",str(final)))
        child(state,phase,command);validate_final(state,phase)
    if not (final/"result.json").exists():child(state,"finalize",pixi("-m",MODULE,"finalize","--root",str(final)))
    result=read(final/"result.json");final_plan=read(final/"plan.json")
    if result.get("status")!="completed" or result.get("fingerprint")!=final_plan["fingerprint"]:
        raise ValueError("Final aggregate is incomplete or mismatched")
    return dict(status="completed",final_result=str(final/"result.json"),final_result_sha256=sha(final/"result.json"),
        raw_text_smoke_manifest=str(Path(state)/"raw_text_smoke/manifest.json"),
        raw_text_smoke_manifest_sha256=sha(Path(state)/"raw_text_smoke/manifest.json"),
        classification=result.get("classification"),confirmed=False)


def watch(state):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Watcher must hide CUDA")
    state=Path(state);plan=verify(state)
    with (state/"watcher.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.seek(0);lock.truncate();lock.write(str(os.getpid())+"\n");lock.flush()
        if (state/"status.json").exists():return read(state/"status.json")
        event(state,"watcher_started",campaign=plan["campaign"]);started=time.time()
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","watch"));parser.add_argument("--state-root",required=True,type=Path)
    parser.add_argument("--campaign",type=Path);parser.add_argument("--final-root",type=Path)
    parser.add_argument("--poll-seconds",type=float,default=15);parser.add_argument("--stale-hours",type=float,default=3)
    args=parser.parse_args()
    if args.command=="prepare" and (args.campaign is None or args.final_root is None):parser.error("prepare requires campaign and final roots")
    result=prepare(args.state_root,args.campaign,args.final_root,poll_seconds=args.poll_seconds,stale_hours=args.stale_hours) if args.command=="prepare" else watch(args.state_root)
    print(json.dumps(result,allow_nan=False),flush=True)
    if result["status"] in ("completion_execution_failure","campaign_execution_failure","stalled"):raise SystemExit(1)


if __name__=="__main__":main()
