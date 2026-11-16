"""Full-depth scheduling, immutable reuse and reservation retry boundaries."""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import fcntl
import threading
import signal
import subprocess

import pytest
import torch

from experiments.structure_net import jspace_3b_full_progressive_controller as study


def stage(cut):
    return dict(cut=cut,compressed_blocks=36-cut,continuation_checkpoint=f"cut{cut}.pt",
        continuation_checkpoint_sha256=str(cut),quality_status="pending",relative_retention_pass=False)


def test_all35_remaining_stages_run_even_when_every_quality_result_fails():
    scheduled=[];graded=[]
    def advance(cut):scheduled.append(cut);return stage(cut)
    def grade(row):
        graded.append(row["cut"])
        return dict(cut=row["cut"],quality_status="fail",relative_retention_pass=False)
    result=study.execute_chain(advance,grade)
    assert scheduled==graded==list(range(34,-1,-1))
    assert len(result["stages"])==35 and result["all36_compressed"]
    assert result["final_checkpoint"]=="cut0.pt" and result["final_quality"]["relative_retention_pass"] is False


def test_unavailable_judge_and_exception_never_stop_gpu_progression():
    scheduled=[]
    def advance(cut):scheduled.append(cut);return stage(cut)
    def grade(row):
        if row["cut"]==34:raise ConnectionError("Judge unavailable")
        return dict(cut=row["cut"],quality_status="unavailable",relative_retention_pass=None)
    result=study.execute_chain(advance,grade)
    assert scheduled==list(study.CUTS) and result["all36_compressed"]
    assert result["unavailable_judge_cuts"]==list(study.CUTS)
    assert result["task_results"][0]["error_type"]=="ConnectionError"


def actual_checkpoint(tmp_path,monkeypatch,cut=0):
    root=tmp_path/"recovery";run=root/"run";run.mkdir(parents=True)
    checkpoint=run/"checkpoint.pt"
    torch.save(dict(stack={"_extra_state":{"layer_indices":list(range(cut,36)),"packed":False,"stripped":False},
                           "fixture":torch.ones(2)},norm={},
        identities={"study":"fp","data":"data","teacher":"revision"}),checkpoint)
    p=dict(fingerprint="fp",cut=cut,data_fingerprint="data",model={"revision":"revision"})
    r=dict(status="completed",systems_passed=True,fingerprint="fp",cut=cut,compressed_blocks=36-cut,
        all36_compressed=cut==0,continuation_checkpoint=str(checkpoint),continuation_checkpoint_sha256=study.sha(checkpoint))
    study.write(run/"gpu_result.json",r)
    study.write(run/"gpu_manifest.json",dict(status="completed",fingerprint="fp",
        artifacts={"gpu_result.json":study.sha(run/"gpu_result.json")},checkpoints={str(checkpoint):study.sha(checkpoint)}))
    monkeypatch.setattr(study.recovery,"verify",lambda *_a,**_k:(p,{},[]))
    return root,checkpoint


def test_actual_checkpoint_layers_and_corruption_are_checked(tmp_path,monkeypatch):
    root,checkpoint=actual_checkpoint(tmp_path,monkeypatch)
    assert study.authenticated_stage(root,0,"data")["compressed_blocks"]==36
    with checkpoint.open("ab") as stream:stream.write(b"tampered")
    with pytest.raises(ValueError,match="checkpoint changed"):
        study.authenticated_stage(root,0,"data")


def test_deceptive_count_receipt_cannot_replace_actual_layer_inventory(tmp_path,monkeypatch):
    root,checkpoint=actual_checkpoint(tmp_path,monkeypatch,cut=1)
    # Keep model file/hash valid while supplying a false claimed depth.
    with pytest.raises(ValueError,match="wrong cut"):
        study.authenticated_stage(root,0,"data")


def test_transient_prelaunch_retry_uses_distinct_logs_and_is_bounded(tmp_path):
    calls=[];waits=[]
    def runner(*_a,**kwargs):
        calls.append(1);kwargs["stdout"].write("HTTP409 allocation reservation conflict\n")
        return SimpleNamespace(returncode=1 if len(calls)==1 else 0)
    study.invoke(["fixture"],tmp_path,"phase",env={},runner=runner,sleeper=waits.append)
    assert len(calls)==2 and waits==[5]
    assert (tmp_path/"phase.attempt1.log").exists() and (tmp_path/"phase.attempt2.log").exists()
    calls.clear();waits.clear()
    def unavailable(*_a,**kwargs):
        calls.append(1);kwargs["stdout"].write("Infisical connection timeout\n");return SimpleNamespace(returncode=1)
    with pytest.raises(RuntimeError):
        study.invoke(["fixture"],tmp_path,"unavailable",env={},runner=unavailable,sleeper=waits.append)
    assert len(calls)==3 and waits==[5,10]


def test_started_producer_is_never_retried_even_for_network_error(tmp_path):
    work=tmp_path/"scientific";calls=[]
    def runner(*_a,**kwargs):
        calls.append(1);work.mkdir();kwargs["stdout"].write("ConnectionError during training\n")
        return SimpleNamespace(returncode=1)
    with pytest.raises(RuntimeError,match="producer_started=True"):
        study.invoke(["fixture"],tmp_path,"phase",env={},markers=[work],runner=runner,
                     sleeper=lambda _:pytest.fail("Retried a started scientific phase"))
    assert len(calls)==1 and work.exists()


def test_explicit_retry_reuses_completed_stages_but_rejects_partial_fit(tmp_path,monkeypatch):
    base=tmp_path/"study";out=base/"controller_retry_1";out.mkdir(parents=True)
    f=base/"blocks_02/fit";r=base/"blocks_02/recovery";(r/"run").mkdir(parents=True);f.mkdir()
    (f/"result.json").write_text("{}");(r/"run/gpu_manifest.json").write_text("{}")
    controller=study.Controller(base,out,{"pins":{},"source_sha256":{},"data_fingerprint":"data"},retry=True)
    controller.command=lambda *_a,**_k:pytest.fail("Reran an authenticated completed scientific phase")
    monkeypatch.setattr(study.shutil,"disk_usage",lambda _:SimpleNamespace(free=study.MINIMUM_FREE_BYTES+1))
    monkeypatch.setattr(study.fit,"verify",lambda _:dict(cut=34,data_fingerprint="data",parent={"root":str(base/"seed")}))
    monkeypatch.setattr(study.fit,"check",lambda _:dict(status="completed"))
    monkeypatch.setattr(study.recovery,"verify",lambda _:({"cut":34,"data_fingerprint":"data"},{},[]))
    gpu=dict(stage(34),response_tokens=25000)
    monkeypatch.setattr(study,"authenticated_stage",lambda *_a:gpu)
    (r/"run/gpu_result.json").write_text("{}")
    assert controller.advance(34)["relative_retention_pass"] is False
    assert controller.parent==r
    controller.parent=base/"seed";controller.retry=False
    with pytest.raises(FileExistsError,match="explicit"):
        controller.advance(34)
    controller.retry=True;(f/"result.json").unlink();(f/"work").mkdir()
    with pytest.raises(RuntimeError,match="Partial scientific fit"):
        controller.advance(34)


def test_gpu_command_is_external_reservation_with_hidden_cpu_controller(tmp_path,monkeypatch):
    controller=study.Controller(tmp_path,tmp_path,{"pins":{},"source_sha256":{}},retry=False)
    calls=[]
    monkeypatch.setattr(study,"invoke",lambda command,*a,**kw:calls.append((command,kw)))
    controller.command("blocks36_recovery","recovery","run",tmp_path/"recovery",minutes=360)
    command,kwargs=calls[0]
    assert command[:5]==["/usr/bin/python3","/usr/local/bin/inference-ctl","lease","run","--allocation"]
    assert command[5]==study.GPU+"=11264"
    assert command[command.index("--minutes")+1]=="360"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"]==""
    assert command[-2:]==["--limit-mib","11008"]


def test_cpu_judge_waits_for_separate_seed_probe_file_lock(tmp_path):
    controller=study.Controller(tmp_path,tmp_path,{"pins":{},"source_sha256":{}},retry=False)
    waiting=threading.Event();entered=threading.Event()
    controller.event=lambda phase,**kw:waiting.set()
    def grade(row):entered.set();return {"quality_status":"unavailable","cut":row["cut"]}
    controller._grade_locked=grade
    with (tmp_path/".cpu_judge.lock").open("a") as external:
        fcntl.flock(external.fileno(),fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(controller.grade,stage(34))
            assert waiting.wait(timeout=2)
            assert not entered.is_set()
            fcntl.flock(external.fileno(),fcntl.LOCK_UN)
            assert future.result(timeout=2)["quality_status"]=="unavailable"
            assert entered.is_set()


@pytest.mark.parametrize("interruption",[subprocess.TimeoutExpired(["fixture"],1),KeyboardInterrupt()])
def test_timeout_and_interrupt_terminate_owned_group_with_bounded_grace(monkeypatch,interruption):
    waits=[];signals=[];opened=[]
    class Process:
        pid=43210
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def wait(self,timeout):
            waits.append(timeout)
            if len(waits)==1:raise interruption
            if len(waits)==2:raise subprocess.TimeoutExpired(["fixture"],timeout)
            return -9
    def popen(*args,**kwargs):opened.append(kwargs);return Process()
    monkeypatch.setattr(study.subprocess,"Popen",popen)
    monkeypatch.setattr(study.os,"killpg",lambda pid,kind:signals.append((pid,kind)))
    with pytest.raises(type(interruption)):
        study.run_process(["fixture"],cwd=".",env={},stdout=None,stderr=None,timeout=1)
    assert opened[0]["start_new_session"] is True
    assert waits==[1,30,10]
    assert signals==[(43210,signal.SIGTERM),(43210,signal.SIGKILL)]
