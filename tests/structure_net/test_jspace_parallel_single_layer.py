import ast
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from experiments.structure_net import jspace_parallel_single_layer as coordinator


def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ""); monkeypatch.delenv("IC_LEASE", raising=False)
    monkeypatch.setattr(coordinator, "LAYERS", list(range(2,8)))
    root=tmp_path/"campaign"; root.mkdir()
    for name in ("logs","stages","workers"): (root/name).mkdir()
    plan=dict(schema=coordinator.SCHEMA,layers=coordinator.LAYERS,cards=[list(c) for c in coordinator.CARDS],
        pilot_layers=[2,3,4],minutes=60,input_sha256={},source_sha256={},source="native",basis_root="basis",
        quality_prefix_root="fp32dev",scope="isolated only")
    plan["fingerprint"]=coordinator.digest(plan);coordinator.write(root/"plan.json",plan)
    return root,plan


class FakeProcesses:
    def __init__(self, *, fail=None, quality_miss=None, bad_verify=None):
        self.fail,self.quality_miss,self.bad_verify=fail,quality_miss,bad_verify
        self.calls=[];self.lock=threading.Lock();self.active_pilots=0;self.peak_pilots=0

    def __call__(self, argv, **kwargs):
        stage=argv[argv.index(coordinator.MODULE)+1];root=Path(argv[argv.index("--root")+1]);layer=int(root.name[-2:])
        with self.lock:self.calls.append((layer,stage,list(argv),dict(kwargs["env"])))
        if stage=="pilot":
            with self.lock:
                self.active_pilots+=1;self.peak_pilots=max(self.peak_pilots,self.active_pilots)
            time.sleep(.03)
            with self.lock:self.active_pilots-=1
        if self.fail==(layer,stage):return SimpleNamespace(returncode=19)
        if stage=="prepare":
            assert not root.exists();root.mkdir(parents=True)
            coordinator.write(root/"plan.json",dict(schema="jspace.independent-layer-distillation.v1",layer=layer,fingerprint=str(layer)))
            return SimpleNamespace(returncode=0)
        prefix,budget=argv[argv.index("--allocation")+1].split("=")
        directory=root/("verification" if stage=="verify" else stage);directory.mkdir()
        if stage=="pilot":
            result=dict(status="completed",systems_only=True,updates=1,
                native_control=dict(passed=True,predictions_exact=True,correct=50,count=64))
        elif stage=="train":
            passed=layer!=self.quality_miss
            result=dict(status="completed",systems_only=False,native_correct=50,minimum_correct=47,
                selected=dict(correct=48 if passed else 46,count=64),quality_pass=passed,selected_epoch=0)
        else:
            result=dict(status="completed",passed=layer!=self.bad_verify,strict_state_load=True,
                training_quality_pass=layer!=self.quality_miss)
        coordinator.write(directory/"result.json",result)
        manifest=dict(status="completed",fingerprint=str(layer),result_sha256=coordinator.sha(directory/"result.json"),
            allocation={prefix+"-full-uuid":int(budget)},lease_id=f"{layer}-{stage}")
        if stage=="verify":manifest["passed"]=result["passed"]
        coordinator.write(directory/"manifest.json",manifest)
        return SimpleNamespace(returncode=0)


def test_three_concurrent_pilots_gate_separately_leased_dynamic_queue(tmp_path,monkeypatch):
    root,_=prepared(tmp_path,monkeypatch);processes=FakeProcesses()
    with pytest.raises(FileNotFoundError):coordinator.execute(root,runner=processes)
    assert not processes.calls
    gate=coordinator.pilots(root,runner=processes)
    assert gate["passed"] and processes.peak_pilots==3
    assert all(stage in ("prepare","pilot") for _,stage,_,_ in processes.calls)
    result=coordinator.execute(root,runner=processes)
    assert result["status"]=="completed" and result["verified_layers"]==list(range(2,8))
    assert result["assembly_evaluated"] is False and result["confirmed"] is False
    for layer in range(2,8):
        assert [stage for n,stage,_,_ in processes.calls if n==layer]==["prepare","pilot","train","verify"]
    for _,stage,argv,env in processes.calls:
        assert "--locked" in argv and "--offline" in argv and argv[argv.index("-e")+1]=="ncm"
        if stage=="prepare":
            assert "inference-ctl" not in " ".join(argv) and env["CUDA_VISIBLE_DEVICES"]==""
        else:
            assert argv[:4]==["/usr/bin/python3","/usr/local/bin/inference-ctl","lease","run"]
            assert argv[argv.index("--minutes")+1]=="60" and "CUDA_VISIBLE_DEVICES" not in env
            assert argv[argv.index("--allocation")+1] in {f"{c[1]}={c[2]}" for c in coordinator.CARDS}
    count=len(processes.calls)
    with pytest.raises(FileExistsError):coordinator.execute(root,runner=processes)
    assert len(processes.calls)==count


def test_failed_concurrent_pilot_prevents_any_training(tmp_path,monkeypatch):
    root,_=prepared(tmp_path,monkeypatch);processes=FakeProcesses(fail=(3,"pilot"))
    result=coordinator.pilots(root,runner=processes)
    assert not result["passed"] and result["status"]=="failed" and processes.peak_pilots==3
    with pytest.raises(ValueError,match="Three concurrent"):coordinator.execute(root,runner=processes)
    assert not any(stage=="train" for _,stage,_,_ in processes.calls)


def test_isolated_quality_miss_is_verified_and_does_not_cancel_others(tmp_path,monkeypatch):
    root,_=prepared(tmp_path,monkeypatch);processes=FakeProcesses(quality_miss=2)
    coordinator.pilots(root,runner=processes);result=coordinator.execute(root,runner=processes)
    assert result["status"]=="completed" and result["isolated_quality_misses"]==[2]
    assert result["verified_layers"]==list(range(2,8)) and result["failed_layers"]==[]
    assert (root/"blocks/block_02/verification/result.json").is_file()


@pytest.mark.parametrize("failure",["train_exit","verify_parity"])
def test_infrastructure_failure_stops_affected_worker_only_without_retry(tmp_path,monkeypatch,failure):
    root,_=prepared(tmp_path,monkeypatch)
    processes=FakeProcesses(fail=(2,"train") if failure=="train_exit" else None,bad_verify=2 if failure=="verify_parity" else None)
    coordinator.pilots(root,runner=processes);result=coordinator.execute(root,runner=processes)
    assert result["status"]=="infrastructure_failed" and result["failed_layers"]==[2]
    assert result["verified_layers"]==list(range(3,8)) and result["unattempted_layers"]==[]
    failed=coordinator.read(root/"workers/3060.json")
    assert failed["status"]=="infrastructure_failed" and failed["failed_layer"]==2
    assert sum(layer==2 and stage=="train" for layer,stage,_,_ in processes.calls)==1
    if failure=="train_exit":assert not any(layer==2 and stage=="verify" for layer,stage,_,_ in processes.calls)


def test_tampered_pilot_receipt_blocks_full_queue(tmp_path,monkeypatch):
    root,_=prepared(tmp_path,monkeypatch);processes=FakeProcesses()
    coordinator.pilots(root,runner=processes)
    path=root/"blocks/block_03/pilot/manifest.json";value=coordinator.read(path);value["lease_id"]="different"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="receipt changed"):coordinator.execute(root,runner=processes)
    assert not (root/"run_started.json").exists()


def test_stdlib_only_cpu_guard_and_prepare_does_not_launch_gpu(tmp_path,monkeypatch):
    path=Path(coordinator.__file__);tree=ast.parse(path.read_text())
    modules={node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
    modules|={alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names}
    assert modules <= {"argparse","concurrent","datetime","hashlib","json","os","pathlib","queue","shutil","subprocess","threading"}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    with pytest.raises(ValueError,match="hide CUDA"):coordinator.prepare(tmp_path,tmp_path,tmp_path,tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","");monkeypatch.setenv("IC_LEASE","occupied")
    with pytest.raises(ValueError,match="no GPU lease"):coordinator.cpu_only()


def test_wrong_gpu_or_budget_receipt_is_not_accepted(tmp_path,monkeypatch):
    root,_=prepared(tmp_path,monkeypatch);processes=FakeProcesses();coordinator.pilots(root,runner=processes)
    path=root/"blocks/block_02/pilot/manifest.json";value=coordinator.read(path)
    value["allocation"]={"GPU-0370fdb0-full-uuid":5120};path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="reservation"):
        coordinator.check_output("pilot",root/"blocks/block_02",2,coordinator.CARDS[0])


def test_prepare_requires_completed_native_control_and_launches_nothing(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","");monkeypatch.delenv("IC_LEASE",raising=False)
    source,basis,cache=(tmp_path/name for name in ("source","basis","cache"))
    for path in (source,basis/"basis",cache/"cache"):path.mkdir(parents=True)
    for path in (source/"plan.json",basis/"plan.json",basis/"basis/manifest.json",cache/"items.json"):
        coordinator.write(path,{})
    coordinator.write(cache/"plan.json",dict(schema="jspace.native-fp32-development-cache.v1",fingerprint="native"))
    result=dict(status="completed",fingerprint="native",correct=50,count=64,native_baseline_predictions_exact=False)
    coordinator.write(cache/"cache/result.json",result)
    manifest=dict(status="completed",fingerprint="native",artifact_sha256={"cache/result.json":coordinator.sha(cache/"cache/result.json")})
    coordinator.write(cache/"cache/manifest.json",manifest)
    monkeypatch.setattr(coordinator.subprocess,"run",lambda *_a,**_k:pytest.fail("Preparation must launch nothing"))
    monkeypatch.setattr(coordinator,"source_paths",lambda:[Path(coordinator.__file__),Path(__file__).resolve()])
    root=tmp_path/"new"
    with pytest.raises(ValueError,match="native FP32"):coordinator.prepare(root,source,basis,cache)
    assert not root.exists()
    result["native_baseline_predictions_exact"]=True;(cache/"cache/result.json").write_text(json.dumps(result))
    manifest["artifact_sha256"]["cache/result.json"]=coordinator.sha(cache/"cache/result.json")
    (cache/"cache/manifest.json").write_text(json.dumps(manifest))
    prepared=coordinator.prepare(root,source,basis,cache)
    assert prepared["workers"]==3 and prepared["layers"]==list(range(2,36))
    assert coordinator.verify_state(root)["protocol"]==str(coordinator.PROTOCOL)
    assert not (root/"blocks").exists() and not (root/"pilots_started.json").exists()
    with pytest.raises(FileExistsError):coordinator.prepare(root,source,basis,cache)
