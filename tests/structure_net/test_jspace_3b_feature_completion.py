"""CPU supervisor contracts: accepted-chain admission, reservations and smoke."""
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.structure_net import jspace_3b_feature_completion as runner


def setup(tmp_path,monkeypatch,*,ready=True):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    source=tmp_path/"source.py";source.write_text("# frozen watcher dependency\n")
    monkeypatch.setattr(runner,"SOURCES",(source,))
    campaign=tmp_path/"campaign";campaign.mkdir()
    runner.write(campaign/"plan.json",dict(schema=runner.CAMPAIGN_SCHEMA,fingerprint="campaign",source_sha256={str(source):runner.sha(source)}))
    if ready:runner.write(campaign/"outcome.json",dict(status="all_blocks_recovered",last_accepted_cut=0,compressed_blocks=36))
    state=tmp_path/"watcher";final=tmp_path/"final"
    runner.prepare(state,campaign,final)
    return state,campaign,final,source


def smoke():
    return dict(source_fingerprint="final",device="cuda:0",greedy=True,position_capacity=416,
        input_tokens=20,generated_token_ids=[4,5],generated_tokens=2,text="A tree grows.",
        reservation=dict(physical_uuid="GPU-0370fdb0-test",lease_id="smoke-lease",lease_bytes=11264*1024**2,
            allocator_limit_bytes=11008*1024**2))


def children(state,final,monkeypatch,*,failed=None):
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"]=="" and kwargs["env"]["OMP_NUM_THREADS"]=="1"
        if "-c" in command:
            script=command[command.index("-c")+1]
            assert "jspace_3b_feature_resident_final_gate" in script
            assert "m.numerical.checked" in script and "m.checked(" not in script
            return SimpleNamespace(returncode=0)
        module=command[command.index("-m")+1]
        phase="smoke" if module==runner.SMOKE_MODULE else command[command.index("-m")+2]
        if phase==failed:return SimpleNamespace(returncode=9)
        if phase=="prepare":
            final.mkdir();runner.write(final/"plan.json",dict(fingerprint="final",campaign_root=runner.read(state/"plan.json")["campaign"]))
        elif phase=="finalize":runner.write(final/"result.json",dict(status="completed",fingerprint="final",classification="measured_not_confirmed"))
        elif phase=="smoke":kwargs["stdout"].write(("runtime notice\n"+json.dumps(smoke())+"\n").encode())
        else:
            (final/phase).mkdir();runner.write(final/phase/"manifest.json",dict(status="completed",fingerprint="final"))
            if phase=="compact_export":
                (final/phase/"model").mkdir();runner.write(final/phase/"model/manifest.json",dict(source_fingerprint="final"))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess,"run",run)
    return calls


def test_plan_archive_and_campaign_identity_are_immutable(tmp_path,monkeypatch):
    state,campaign,_,source=setup(tmp_path,monkeypatch,ready=False)
    plan=runner.verify(state)
    assert plan["poll_seconds"]==15 and plan["stale_seconds"]==10800
    assert runner.observe(plan,100,now=10899) is None
    assert runner.observe(plan,100,now=10900)["status"]=="stalled"
    (state/"producer/source.py").write_text("changed")
    with pytest.raises(ValueError,match="producer changed"):runner.verify(state)
    with pytest.raises(ValueError):runner.prepare(tmp_path/"other",campaign,tmp_path/"otherfinal",stale_hours=2.9)


def test_full36_marker_is_required_before_even_cpu_final_preparation(tmp_path,monkeypatch):
    state,_,final,_=setup(tmp_path,monkeypatch,ready=False)
    monkeypatch.setattr(runner,"child",lambda *a:pytest.fail("final preparation started without full36"))
    with pytest.raises(ValueError,match="Full36 accepted"):
        runner.finish(state)
    assert not final.exists()


def test_six_reserved_final_phases_and_separate_smoke_before_confirmation(tmp_path,monkeypatch):
    state,_,final,_=setup(tmp_path,monkeypatch);calls=children(state,final,monkeypatch)
    result=runner.finish(state)
    assert result["status"]=="completed" and result["confirmed"] is False
    phases=[("smoke" if c[c.index("-m")+1]==runner.SMOKE_MODULE else c[c.index("-m")+2]) for c in calls if "-m" in c]
    assert phases==["prepare","compact_export","compact_validate","smoke",*list(runner.PHASES)[2:],"finalize"]
    gpu=[c for c in calls if "lease" in c];assert len(gpu)==7
    for command in gpu:
        assert command.count("--allocation")==1
        assert command[command.index("--allocation")+1]=="GPU-0370fdb0=11264"
        assert command[command.index("--minutes")+1]=="120"
    smoke_command=next(c for c in gpu if runner.SMOKE_MODULE in c)
    assert smoke_command[smoke_command.index("--max-new-tokens")+1]=="16"
    assert "--json" in smoke_command
    receipt=runner.read(state/"raw_text_smoke/manifest.json")
    assert receipt["status"]=="completed" and receipt["configuration"]==runner.SMOKE
    assert receipt["result_sha256"]==runner.sha(state/"raw_text_smoke/result.json")
    calls.clear();assert runner.finish(state)==result
    assert all("-c" in c and "lease" not in c for c in calls)


@pytest.mark.parametrize("phase",["compact_validate","smoke"])
def test_failed_validation_or_smoke_never_reaches_heldout_and_is_not_retried(tmp_path,monkeypatch,phase):
    state,_,final,_=setup(tmp_path,monkeypatch);calls=children(state,final,monkeypatch,failed=phase)
    with pytest.raises(RuntimeError,match="exited 9"):runner.finish(state)
    assert not (final/"native_eager_before").exists()
    if phase=="smoke":
        calls.clear()
        with pytest.raises(RuntimeError,match="Incomplete raw-text smoke"):
            runner.finish(state)
        assert not any("lease" in c for c in calls)


@pytest.mark.parametrize("change",["fingerprint","token_cap","reservation","log"])
def test_smoke_receipt_rejects_invalid_output_or_changed_completed_log(tmp_path,monkeypatch,change):
    if change!="log":
        value=smoke()
        if change=="fingerprint":value["source_fingerprint"]="other-model"
        elif change=="token_cap":value["generated_token_ids"]=[4]*17;value["generated_tokens"]=17
        else:value["reservation"]["allocator_limit_bytes"]+=1
        with pytest.raises(ValueError,match="smoke output"):
            runner.validate_smoke_result(value,fingerprint="final")
    else:
        state,_,final,_=setup(tmp_path,monkeypatch);children(state,final,monkeypatch);runner.finish(state)
        receipt=runner.read(state/"raw_text_smoke/manifest.json")
        Path(receipt["log"]).write_text("changed historical log")
        with pytest.raises(ValueError,match="smoke provenance"):
            runner.raw_text_smoke(state)


@pytest.mark.parametrize("kind",["quality","execution"])
def test_terminal_failure_is_durable_and_never_launches_final_check(tmp_path,monkeypatch,kind):
    state,campaign,_,_=setup(tmp_path,monkeypatch,ready=False)
    if kind=="quality":runner.write(campaign/"outcome.json",dict(status="quality_stop",failed_cut=22))
    else:
        path=campaign/"stages/block_22/train";path.mkdir(parents=True)
        runner.write(path/"failure.json",dict(status="failed",error="example"))
    monkeypatch.setattr(runner,"finish",lambda *a:pytest.fail("final launched after terminal failure"))
    result=runner.watch(state)
    assert result["status"]==("quality_stop" if kind=="quality" else "campaign_execution_failure")
    assert runner.watch(state)==result


def test_duplicate_supervisor_lock_blocks_second_watcher(tmp_path,monkeypatch):
    state,_,_,_=setup(tmp_path,monkeypatch)
    with (state/"watcher.lock").open("a+") as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):runner.watch(state)
    assert not (state/"status.json").exists()
