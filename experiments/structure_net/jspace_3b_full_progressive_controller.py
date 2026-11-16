"""Reservation-backed full36 continuation with independent serial CPU judging."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time

import torch

from . import jspace_3b_full_progressive_data as data
from . import jspace_3b_full_progressive_fit as fit
from . import jspace_3b_full_progressive_recovery as recovery

ROOT=Path(__file__).resolve().parents[2]
PIXI="/home/rabbit/.pixi/bin/pixi"
MODULE="experiments.structure_net.jspace_3b_full_progressive_"
SCHEMA="jspace.3b-full-progressive-controller.v1"
V2=Path("/data/structure_net_experiments/jspace_3b_progressive_response_rebuild/20261116_v2/blocks_01/recovery")
SEED_CHECKPOINT=V2/"run/checkpoints/update_0046.pt"
GPU="GPU-0370fdb0"
CUTS=tuple(range(34,-1,-1))
MINIMUM_FREE_BYTES=64*1024**3
read,write,sha=data.read,data.write,data.sha


def authenticated_stage(root,cut,data_fingerprint):
    """Bind receipts, actual retained checkpoint identity and its layer inventory."""
    root=Path(root).resolve();p,_,_=recovery.verify(root,live=False)
    r=recovery.gpu_result(root,p)
    if (p["cut"]!=cut or p["data_fingerprint"]!=data_fingerprint or r.get("cut")!=cut
            or r.get("compressed_blocks")!=36-cut or r.get("all36_compressed") is not (cut==0)):
        raise ValueError("Completed GPU stage has the wrong cut, block count, or corpus")
    checkpoint=torch.load(r["continuation_checkpoint"],map_location="cpu",weights_only=True,mmap=True)
    extra=checkpoint.get("stack",{}).get("_extra_state",{})
    identities=checkpoint.get("identities",{})
    if (list(extra.get("layer_indices",[]))!=list(range(cut,36))
            or extra.get("packed") is not False or extra.get("stripped") is not False
            or identities.get("study")!=p["fingerprint"] or identities.get("data")!=data_fingerprint
            or identities.get("teacher")!=p["model"]["revision"]):
        raise ValueError("Actual continuation checkpoint does not contain the declared compact suffix")
    del checkpoint
    return r


def transient_prelaunch(text):
    message=text.lower()
    network=("connection refused","connection reset","connection aborted","read timed out","connecttimeout",
             "connectionerror","temporary failure","network is unreachable","service unavailable","bad gateway")
    return (any(term in message for term in network)
            or "409" in message and any(term in message for term in ("conflict","allocation","reservation","lease"))
            or "infisical" in message and any(term in message for term in ("timeout","timed out","502","503","504")))


def run_process(command,*,cwd,env,stdout,stderr,timeout):
    """Keep the reservation wrapper and its worker in one owned process group.

    SIGTERM gives inference-ctl its forwarding/finally lease cleanup. A bounded
    grace period precedes group-wide cleanup, so an interrupted controller does
    not orphan a producer after killing only its reservation wrapper.
    """
    with subprocess.Popen(command,cwd=cwd,env=env,stdout=stdout,stderr=stderr,start_new_session=True) as process:
        try:
            returncode=process.wait(timeout=timeout)
        except BaseException:
            def signal_group(kind):
                try:os.killpg(process.pid,kind)
                except ProcessLookupError:pass
            signal_group(signal.SIGTERM)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
            finally:
                # Include any descendant that ignored termination even if its
                # wrapper has already exited and released the lease.
                signal_group(signal.SIGKILL)
                process.wait(timeout=10)
            raise
        return subprocess.CompletedProcess(command,returncode)


def invoke(command,log_root,label,*,env,markers=(),timeout=3600,runner=None,
           sleeper=time.sleep,event=lambda *_a,**_k:None,verify=lambda:None):
    """Retry only transient reservation/network failures before producer entry."""
    runner=run_process if runner is None else runner
    for attempt in range(1,4):
        verify();log=Path(log_root)/f"{label}.attempt{attempt}.log"
        event("launch",label=label,attempt=attempt,command=command)
        with log.open("x") as stream:
            result=runner(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        event("finished",label=label,attempt=attempt,returncode=result.returncode)
        if result.returncode==0:return
        entered=any(Path(path).exists() for path in markers)
        with log.open("rb") as stream:
            stream.seek(max(0,log.stat().st_size-65536));tail=stream.read().decode("utf-8",errors="replace")
        if entered or attempt==3 or not transient_prelaunch(tail):
            raise RuntimeError(f"{label} failed; producer_started={entered}; receipt/log preserved at {log}")
        event("prelaunch_transient_retry",label=label,attempt=attempt,wait_seconds=5*attempt)
        sleeper(5*attempt)


def execute_chain(advance,grade,*,event=lambda *_a,**_k:None):
    """Every remaining depth is scheduled; task quality never selects the next cut."""
    stages=[];jobs=[]
    def safe_grade(stage):
        try:return grade(stage)
        except Exception as error:
            return dict(cut=stage["cut"],quality_status="unavailable",relative_retention_pass=None,
                        error_type=type(error).__name__,error=str(error),continuation_unchanged=True)
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix="full-progressive-judge") as pool:
        for cut in CUTS:
            stage=advance(cut)
            if stage.get("cut")!=cut or stage.get("compressed_blocks")!=36-cut:
                raise ValueError("Progressive scheduler received an incorrect depth")
            stages.append(stage);jobs.append(pool.submit(safe_grade,stage))
            event("gpu_stage_completed",cut=cut,compressed_blocks=36-cut,quality_status=stage.get("quality_status","pending"))
        grades=[job.result() for job in jobs]
    return dict(stages=stages,task_results=grades,all36_compressed=len(stages)==35 and stages[-1]["cut"]==0,
        final_checkpoint=stages[-1]["continuation_checkpoint"],
        final_checkpoint_sha256=stages[-1]["continuation_checkpoint_sha256"],final_quality=grades[-1],
        unavailable_judge_cuts=[r["cut"] for r in grades if r.get("quality_status") in ("unavailable","pending")],
        quality_failure_stops_progression=False,confirmed=False)


class Controller:
    def __init__(self,base,out,plan,*,retry=False):
        self.base,self.out,self.plan=Path(base),Path(out),plan
        self.retry=retry;self.data_root=self.base/"data";self.parent=self.base/"seed"
        self.lock=threading.Lock()
        self.env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",
            PIXI_CACHE_DIR="/tmp/structure-net-pixi-cache",UV_CACHE_DIR="/tmp/structure-net-uv-cache")

    def event(self,phase,**values):
        row=dict(utc=datetime.now(timezone.utc).isoformat(),phase=phase,**values)
        with self.lock:
            with (self.out/"progress.jsonl").open("a") as stream:stream.write(json.dumps(row,allow_nan=False)+"\n")
            print(json.dumps(row,allow_nan=False),flush=True)

    def verify(self):
        for path,digest in self.plan["pins"].items():
            if sha(path)!=digest:raise ValueError("Controller input/source changed: "+path)
        for path,digest in self.plan["source_sha256"].items():
            if sha(self.out/"producer"/Path(path).relative_to(ROOT))!=digest:raise ValueError("Archived controller producer changed")

    def command(self,label,module,phase,root,*,extra=(),minutes=None,markers=()):
        command=[PIXI,"run","--locked","--offline","-e","ncm","python","-m",MODULE+module,
                 phase,"--root",str(root),*map(str,extra)]
        if minutes is not None:
            command += ["--limit-mib","11008"]
            command=["/usr/bin/python3","/usr/local/bin/inference-ctl","lease","run","--allocation",GPU+"=11264",
                "--minutes",str(minutes),"--reason",f"Full36 progressive compression: {label}; all36 independent worker preserved",
                "--",*command]
        invoke(command,self.out,label,env=self.env,markers=markers,timeout=minutes*60+300 if minutes else 3600,
               event=self.event,verify=self.verify)

    def native_and_seed(self):
        dp,_=data.verify(self.data_root,live=False)
        native=self.data_root/"native"
        if not (native/"manifest.json").exists():
            if native.exists():raise RuntimeError("Native capture is active or partial; it will not be duplicated")
            self.command("native_capture","data","capture",self.data_root,minutes=180,markers=[native])
        data._manifest_files(native,dp["fingerprint"],hash_payloads=False)
        r=read(native/"result.json")
        if not all(r.get(k) for k in ("full_teacher_bit_exact","existing_boundary_bit_exact","teacher_unchanged","seed_boundary35_bit_exact")):
            raise ValueError("Full native cache controls did not pass")
        if not self.parent.exists():
            self.command("seed_prepare","fit","seed",self.parent,
                extra=["--data-root",self.data_root,"--source-root",self.plan["seed_source_root"],
                       "--checkpoint",self.plan["seed_checkpoint"]],markers=[self.parent])
        parent,_=fit.continuation_parent(self.parent,34,dp,checkpoint=self.plan["seed_checkpoint"])
        self.event("seed_authenticated",checkpoint_sha256=parent["checkpoint_sha256"],
                   relative_retention_pass=parent["relative_retention_pass"],quality=parent["quality"])

    def advance(self,cut):
        self.verify();free=shutil.disk_usage(self.base).free
        if free<MINIMUM_FREE_BYTES:raise OSError(f"Stage needs64GiB temporary disk headroom; available={free}")
        self.event("stage_disk_admission",cut=cut,available_bytes=free,minimum_bytes=MINIMUM_FREE_BYTES)
        stage=self.base/f"blocks_{36-cut:02d}";f=stage/"fit";r=stage/"recovery"
        if stage.exists() and not self.retry:raise FileExistsError("Prior stage artifacts require explicit controller_retry_N")
        stage.mkdir(exist_ok=True)
        if not f.exists():
            self.command(f"blocks{36-cut}_fit_prepare","fit","prepare",f,
                extra=["--data-root",self.data_root,"--cut",cut,"--parent-root",self.parent],markers=[f])
        fp=fit.verify(f)
        if (fp["cut"]!=cut or fp["data_fingerprint"]!=self.plan["data_fingerprint"]
                or Path(fp["parent"]["root"]).resolve()!=self.parent.resolve()):
            raise ValueError("Prepared fit differs from this ordered continuation")
        if (f/"result.json").exists():
            if fit.check(f).get("status")!="completed":raise ValueError("Fit completion is not authenticated")
            self.event("reuse_completed_fit",cut=cut)
        else:
            if (f/"work").exists() or (f/"failure.json").exists():raise RuntimeError("Partial scientific fit preserved; no automatic rerun")
            self.command(f"blocks{36-cut}_fit","fit","run",f,minutes=180,markers=[f/"work",f/"failure.json"])
            if fit.check(f).get("status")!="completed":raise ValueError("Fit did not complete")
        if not r.exists():
            self.command(f"blocks{36-cut}_recovery_prepare","recovery","prepare",r,
                extra=["--data-root",self.data_root,"--fit-root",f,"--cut",cut],markers=[r])
        rp,_,_=recovery.verify(r)
        if rp["cut"]!=cut or rp["data_fingerprint"]!=self.plan["data_fingerprint"]:
            raise ValueError("Prepared recovery differs from this stage")
        if not (r/"run/gpu_manifest.json").exists():
            if (r/"run").exists():raise RuntimeError("Partial scientific recovery preserved; no automatic rerun")
            if (r/"pilot/result.json").exists():
                recovery.base.pilot_receipt(r,rp);self.event("reuse_completed_pilot",cut=cut)
            else:
                if (r/"pilot").exists():raise RuntimeError("Partial systems pilot preserved; no automatic rerun")
                self.command(f"blocks{36-cut}_pilot","recovery","pilot",r,minutes=120,markers=[r/"pilot"])
                recovery.base.pilot_receipt(r,rp)
            self.command(f"blocks{36-cut}_recovery","recovery","run",r,minutes=360,markers=[r/"run"])
        result=authenticated_stage(r,cut,self.plan["data_fingerprint"])
        self.parent=r
        return dict(cut=cut,compressed_blocks=result["compressed_blocks"],root=str(r),
            continuation_checkpoint=result["continuation_checkpoint"],continuation_checkpoint_sha256=result["continuation_checkpoint_sha256"],
            quality_status=result.get("quality_status","pending"),relative_retention_pass=result.get("relative_retention_pass"),
            gpu_result_sha256=sha(r/"run/gpu_result.json"),gpu_manifest_sha256=sha(r/"run/gpu_manifest.json"),
            response_tokens=result.get("response_tokens"),selected=result.get("selected"))

    def grade(self,stage):
        # The seed task probe is launched separately by the root process using
        # /usr/bin/flock on this same path. One executor plus this shared lock
        # keeps the judge route globally serial across both producers.
        self.event("waiting_for_cpu_judge_lock",cut=stage["cut"])
        with (self.base/".cpu_judge.lock").open("a") as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            return self._grade_locked(stage)

    def _grade_locked(self,stage):
        root=Path(stage["root"]);cut=stage["cut"]
        if not (root/"run/task_manifest.json").exists():
            if (root/"run/task_judge").exists():
                self.event("judge_unavailable_preserved_attempt",cut=cut)
                return dict(cut=cut,quality_status="unavailable",relative_retention_pass=None,
                    reason="Existing incomplete judge attempt preserved",continuation_unchanged=True)
            try:self.command(f"blocks{36-cut}_judge","recovery","judge",root,markers=[root/"run/task_judge"])
            except Exception as error:
                self.event("judge_unavailable",cut=cut,error_type=type(error).__name__)
                return dict(cut=cut,quality_status="unavailable",relative_retention_pass=None,
                    error_type=type(error).__name__,error=str(error),continuation_unchanged=True)
        result=recovery.check(root).get("task_result")
        if result is None:raise ValueError("Judge exited without authenticated task evidence")
        if result.get("continuation_checkpoint_sha256")!=stage["continuation_checkpoint_sha256"]:
            raise ValueError("Judge attempted to change the GPU continuation checkpoint")
        self.event("judge_completed",cut=cut,quality_status=result["quality_status"])
        return result


def run(root,*,attempt_name="controller",seed_source_root=V2,seed_checkpoint=SEED_CHECKPOINT):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("CPU controller requires CUDA hidden")
    if re.fullmatch(r"controller(?:_retry_[1-9][0-9]*)?",attempt_name) is None:raise ValueError("Explicit controller or controller_retry_N required")
    root=Path(root).resolve();out=root/attempt_name
    if not root.is_dir():raise ValueError("Prepared study root required")
    with (root/".full_progressive_controller.lock").open("a") as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError("Another full-progressive controller is active") from error
        if out.exists():raise FileExistsError(out)
        dp,_=data.verify(root/"data",live=False)
        files=data.cache.prior.source.inherited.local_import_closure([Path(__file__),data.PROTOCOL,
            ROOT/"tests/structure_net/test_jspace_3b_full_progressive_controller.py",
            ROOT/"tests/structure_net/test_jspace_3b_full_progressive_fit.py",
            ROOT/"tests/structure_net/test_jspace_3b_full_progressive_recovery.py",
            ROOT/"tests/structure_net/test_jspace_3b_full_progressive_data.py"])
        sources={str(path):sha(path) for path in sorted(files)}
        pins={**sources,str(root/"data/plan.json"):sha(root/"data/plan.json")}
        p=dict(schema=SCHEMA,cuts=list(CUTS),data_fingerprint=dp["fingerprint"],seed_source_root=str(Path(seed_source_root).resolve()),
            seed_checkpoint=str(Path(seed_checkpoint).resolve()),allocation_mib=11264,allocator_mib=11008,gpu=GPU,
            phase_minutes=dict(fit=180,pilot=120,recovery=360,capture=180),judge_workers=1,
            minimum_free_bytes=MINIMUM_FREE_BYTES,pins=pins,source_sha256=sources,
            quality_failure_stops_progression=False,all36_existing_worker_untouched=True,confirmed=False)
        p["fingerprint"]=data.data.digest(p);out.mkdir();write(out/"plan.json",p);fit._archive(out,sources)
        controller=Controller(root,out,p,retry=attempt_name!="controller");started=time.monotonic()
        try:
            controller.native_and_seed()
            result=execute_chain(controller.advance,controller.grade,event=controller.event)
            result.update(status="completed",fingerprint=p["fingerprint"],elapsed_seconds=time.monotonic()-started,
                compressed_blocks=36,seed_compressed_blocks=1,new_stages_completed=len(result["stages"]),
                all36_existing_worker_untouched=True)
            write(out/"result.json",result)
            write(out/"manifest.json",dict(status="completed",fingerprint=p["fingerprint"],artifacts={
                name:sha(out/name) for name in ("plan.json","progress.jsonl","result.json")}))
            return result
        except BaseException as error:
            write(out/"failure.json",dict(status="execution_failed",fingerprint=p["fingerprint"],
                error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started,
                partial_scientific_artifacts_preserved=True,requires_explicit_retry_attempt=True,confirmed=False))
            raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",required=True)
    parser.add_argument("--attempt-name",default="controller")
    parser.add_argument("--seed-source-root",default=str(V2));parser.add_argument("--seed-checkpoint",default=str(SEED_CHECKPOINT))
    args=vars(parser.parse_args());result=run(**args)
    print(json.dumps({k:result[k] for k in ("status","all36_compressed","compressed_blocks","unavailable_judge_cuts")}),flush=True)
