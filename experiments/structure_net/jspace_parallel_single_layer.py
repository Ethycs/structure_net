"""Bounded local scheduling of independent single-layer fits; stdlib only.

Each GPU owns one worker. Scientific misses remain recorded outcomes; an
execution failure stops that worker without retrying its failed layer.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading

ROOT = Path(__file__).resolve().parents[2]
MODULE = "experiments.structure_net.jspace_independent_layer_distill"
SCHEMA = "jspace.parallel-independent-layer-coordinator.v1"
PROTOCOL = ROOT/"docs/07 - Status Reports/jspace/2026-11-14_qwen-parallel-single-layer-recovery.md"
CARDS = (("3060", "GPU-0370fdb0", 7168), ("2060", "GPU-8cf440df", 5120), ("2060_SUPER", "GPU-7b1db3cb", 7168))
LAYERS = list(range(2,36))
PIXI = ["pixi", "run", "--locked", "--offline", "-e", "ncm", "python", "-m", MODULE]
LOCK = threading.Lock()


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def write(path,value):
    with Path(path).open("x") as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
def event(root,**values):
    row=dict(time=datetime.now(timezone.utc).isoformat(),**values)
    with LOCK:
        with (Path(root)/"events.jsonl").open("a") as stream:
            stream.write(json.dumps(row,allow_nan=False)+"\n"); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps(row,allow_nan=False),flush=True)


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or os.environ.get("IC_LEASE"):
        raise ValueError("Coordinator must hide CUDA and hold no GPU lease")


def environment(gpu=False):
    result={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONUNBUFFERED":"1",
        "OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
        "PIXI_CACHE_DIR":"/tmp/structure-net-pixi-cache","UV_CACHE_DIR":"/tmp/structure-net-uv-cache"}
    if gpu: result.pop("CUDA_VISIBLE_DEVICES",None)
    return result


def command(plan,root,layer,card,stage):
    child=Path(root)/"blocks"/f"block_{layer:02d}"
    argv=PIXI+[stage,"--root",str(child)]
    if stage=="prepare":
        return argv+["--source",plan["source"],"--basis-root",plan["basis_root"],
            "--quality-prefix-root",plan["quality_prefix_root"],"--layer",str(layer)]
    if stage not in ("pilot","train","verify"): raise ValueError("Unknown independent-layer stage")
    return ["/usr/bin/python3","/usr/local/bin/inference-ctl","lease","run","--allocation",f"{card[1]}={card[2]}",
        "--minutes","60","--reason",f"JSPACE independent block{layer:02d} {stage} {card[0]}","--"]+argv


def source_paths():
    return [Path(__file__),ROOT/"tests/structure_net/test_jspace_parallel_single_layer.py",PROTOCOL,
        ROOT/"experiments/structure_net/jspace_independent_layer_distill.py",
        ROOT/"tests/structure_net/test_jspace_independent_layer_distill.py",
        ROOT/"experiments/structure_net/jspace_native_dev_cache.py",
        ROOT/"tests/structure_net/test_jspace_native_dev_cache.py"]


def prepare(root,source,basis_root,quality_prefix_root):
    cpu_only(); root,source,basis_root,quality_prefix_root=map(lambda p:Path(p).resolve(),(root,source,basis_root,quality_prefix_root))
    if root.exists(): raise FileExistsError(root)
    native=read(quality_prefix_root/"plan.json"); manifest=read(quality_prefix_root/"cache/manifest.json")
    result=read(quality_prefix_root/"cache/result.json")
    if (native.get("schema")!="jspace.native-fp32-development-cache.v1" or manifest.get("status")!="completed"
            or manifest.get("fingerprint")!=native["fingerprint"]
            or manifest["artifact_sha256"].get("cache/result.json")!=sha(quality_prefix_root/"cache/result.json")
            or result.get("status")!="completed" or result.get("fingerprint")!=native["fingerprint"]
            or result.get("correct")!=50 or result.get("count")!=64 or result.get("native_baseline_predictions_exact") is not True):
        raise ValueError("Authenticated native FP32 development cache reproducing50/64 required")
    inputs=[source/"plan.json",basis_root/"plan.json",basis_root/"basis/manifest.json",
        quality_prefix_root/"plan.json",quality_prefix_root/"items.json",quality_prefix_root/"cache/manifest.json",quality_prefix_root/"cache/result.json"]
    plan=dict(schema=SCHEMA,source=str(source),basis_root=str(basis_root),quality_prefix_root=str(quality_prefix_root),
        layers=LAYERS,cards=[list(c) for c in CARDS],pilot_layers=[2,3,4],minutes=60,protocol=str(PROTOCOL),
        input_sha256={str(p):sha(p) for p in inputs},source_sha256={str(p):sha(p) for p in source_paths()},
        scope="Independent native-input single-layer fits only; assembly and confirmation remain separate",
        failure_policy="No retries/overwrites. Scientific quality miss continues after verification; infrastructure failure stops affected worker.")
    plan["fingerprint"]=digest(plan); root.mkdir(parents=True)
    for name in ("logs","stages","workers"): (root/name).mkdir()
    for path in source_paths():
        target=root/"producer"/path.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    write(root/"plan.json",plan)
    return dict(status="prepared",fingerprint=plan["fingerprint"],layers=LAYERS,workers=3)


def verify_state(root):
    root=Path(root); plan=read(root/"plan.json")
    if (plan["schema"]!=SCHEMA or plan["fingerprint"]!=digest({k:v for k,v in plan.items() if k!="fingerprint"})
            or plan["layers"]!=LAYERS or plan["cards"]!=[list(c) for c in CARDS]
            or plan["pilot_layers"]!=[2,3,4] or plan["minutes"]!=60): raise ValueError("Coordinator plan changed")
    for path,expected in plan["input_sha256"].items():
        if sha(path)!=expected: raise ValueError("Changed shared input: "+path)
    for path,expected in plan["source_sha256"].items():
        if sha(path)!=expected or sha(root/"producer"/Path(path).relative_to(ROOT))!=expected:
            raise ValueError("Changed coordinator or worker source")
    return plan


def check_output(stage,child,layer,card):
    plan=read(child/"plan.json")
    if plan.get("schema")!="jspace.independent-layer-distillation.v1" or plan.get("layer")!=layer:
        raise ValueError("Worker prepared the wrong independent layer")
    if stage=="prepare": return dict(plan_sha256=sha(child/"plan.json"))
    directory=child/("verification" if stage=="verify" else stage)
    manifest=read(directory/"manifest.json"); result=read(directory/"result.json")
    if (manifest.get("status")!="completed" or manifest.get("fingerprint")!=plan["fingerprint"]
            or manifest.get("result_sha256")!=sha(directory/"result.json") or result.get("status")!="completed"):
        raise ValueError("Incomplete or mismatched worker receipt")
    allocation=manifest["allocation"]
    if len(allocation)!=1 or not next(iter(allocation)).startswith(card[1]) or next(iter(allocation.values()))!=card[2]:
        raise ValueError("Worker used a different physical GPU or reservation")
    receipt=dict(manifest_sha256=sha(directory/"manifest.json"),result_sha256=sha(directory/"result.json"),
        lease_id=manifest["lease_id"],allocation=allocation)
    if stage=="pilot":
        control=result["native_control"]
        if (result.get("systems_only") is not True or result.get("updates")!=1
                or control.get("passed") is not True or control.get("predictions_exact") is not True
                or control.get("correct")!=50 or control.get("count")!=64):
            raise ValueError("Systems pilot/native substitution control failed")
    elif stage=="train":
        if (result.get("systems_only") is not False or result.get("native_correct")!=50
                or result.get("minimum_correct")!=47 or result["selected"]["count"]!=64
                or result.get("quality_pass")!=(result["selected"]["correct"]>=47)):
            raise ValueError("Isolated diagnostic gate changed")
        receipt.update(quality_pass=result["quality_pass"],selected_correct=result["selected"]["correct"],selected_epoch=result["selected_epoch"])
    elif manifest.get("passed") is not True or result.get("passed") is not True or result.get("strict_state_load") is not True:
        raise ValueError("Independent checkpoint verification failed")
    return receipt


def stage(root,plan,layer,card,name,runner,checker):
    verify_state(root); child=root/"blocks"/f"block_{layer:02d}"
    if name=="prepare" and child.exists(): raise FileExistsError(child)
    key=f"block_{layer:02d}_{name}"; argv=command(plan,root,layer,card,name)
    event(root,worker=card[0],layer=layer,stage=name,status="started",command=argv)
    with (root/"logs"/(key+".log")).open("x") as stream:
        result=runner(argv,cwd=ROOT,env=environment(name!="prepare"),stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"{key} exited{result.returncode}; inspect child log")
    receipt=checker(name,child,layer,card)
    write(root/"stages"/(key+".json"),dict(status="completed",returncode=0,**receipt))
    event(root,worker=card[0],layer=layer,stage=name,status="completed",**receipt)
    return receipt


def pilots(root,*,runner=subprocess.run,checker=check_output):
    cpu_only(); root=Path(root).resolve(); plan=verify_state(root)
    write(root/"pilots_started.json",dict(pid=os.getpid(),fingerprint=plan["fingerprint"]))
    results=[]
    try:
        # Complete CPU preparation before simultaneous GPU launch; no barrier
        # can strand other threads if an individual preparation fails.
        for layer,card in zip(plan["pilot_layers"],CARDS): stage(root,plan,layer,card,"prepare",runner,checker)
        barrier=threading.Barrier(3)
        def launch(pair):
            layer,card=pair; barrier.wait()
            try: return dict(layer=layer,worker=card[0],status="completed",receipt=stage(root,plan,layer,card,"pilot",runner,checker))
            except BaseException as error: return dict(layer=layer,worker=card[0],status="failed",error=repr(error))
        with ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(launch,zip(plan["pilot_layers"],CARDS)))
        passed=all(r["status"]=="completed" for r in results)
        value=dict(status="completed" if passed else "failed",passed=passed,fingerprint=plan["fingerprint"],
            concurrent_workers=3,results=results)
        write(root/"pilots_gate.json",value); return value
    except BaseException as error:
        write(root/"pilots_failed.json",dict(status="failed",error=repr(error),fingerprint=plan["fingerprint"])); raise


def execute(root,*,runner=subprocess.run,checker=check_output):
    cpu_only(); root=Path(root).resolve(); plan=verify_state(root); gate=read(root/"pilots_gate.json")
    if gate.get("passed") is not True or gate.get("fingerprint")!=plan["fingerprint"] or gate.get("status")!="completed":
        raise ValueError("Three concurrent, matching GPU pilots must pass before the full queue")
    for layer,card in zip(plan["pilot_layers"],CARDS):
        expected=next(r for r in gate["results"] if r["layer"]==layer and r["worker"]==card[0])
        if checker("pilot",root/"blocks"/f"block_{layer:02d}",layer,card)!=expected["receipt"]:
            raise ValueError("Completed pilot receipt changed")
    write(root/"run_started.json",dict(pid=os.getpid(),fingerprint=plan["fingerprint"]))
    remaining=queue.Queue()
    for layer in plan["layers"][3:]: remaining.put(layer)
    def worker(pair):
        seed,card=pair; completed=[]; layer=seed
        while True:
            try:
                if layer!=seed:
                    stage(root,plan,layer,card,"prepare",runner,checker); stage(root,plan,layer,card,"pilot",runner,checker)
                trained=stage(root,plan,layer,card,"train",runner,checker)
                stage(root,plan,layer,card,"verify",runner,checker)  # Required even after an isolated quality miss.
                completed.append(dict(layer=layer,**trained))
                event(root,worker=card[0],layer=layer,status="verified",quality_pass=trained["quality_pass"])
            except BaseException as error:
                result=dict(status="infrastructure_failed",worker=card[0],failed_layer=layer,error=repr(error),completed=completed)
                write(root/"workers"/(card[0]+".json"),result); event(root,**{k:v for k,v in result.items() if k!="completed"}); return result
            try: layer=remaining.get_nowait()
            except queue.Empty:
                result=dict(status="completed",worker=card[0],completed=completed)
                write(root/"workers"/(card[0]+".json"),result); return result
    with ThreadPoolExecutor(max_workers=3) as pool: workers=list(pool.map(worker,zip(plan["pilot_layers"],CARDS)))
    completed=[row for w in workers for row in w["completed"]]; failed=[w["failed_layer"] for w in workers if w["status"]!="completed"]
    value=dict(status="completed" if not failed else "infrastructure_failed",fingerprint=plan["fingerprint"],workers=workers,
        verified_layers=sorted(r["layer"] for r in completed),isolated_quality_misses=sorted(r["layer"] for r in completed if not r["quality_pass"]),
        failed_layers=failed,unattempted_layers=sorted(set(plan["layers"])-{r["layer"] for r in completed}-set(failed)),
        confirmed=False,assembly_evaluated=False,scope=plan["scope"])
    write(root/"result.json",value); return value


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("stage",choices=("prepare","pilots","run"))
    parser.add_argument("--root",type=Path,required=True)
    for key in ("source","basis-root","quality-prefix-root"): parser.add_argument("--"+key,type=Path)
    args=parser.parse_args()
    if args.stage=="prepare":
        if any(v is None for v in (args.source,args.basis_root,args.quality_prefix_root)): parser.error("prepare requires all shared input roots")
        result=prepare(args.root,args.source,args.basis_root,args.quality_prefix_root)
    else: result={"pilots":pilots,"run":execute}[args.stage](args.root)
    print(json.dumps(result,allow_nan=False),flush=True)
    if result["status"] not in ("prepared","completed"): raise SystemExit(2)
