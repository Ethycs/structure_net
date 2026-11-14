"""Durable local campaign/export/validation/closeout supervisor, without retries.

The coordinator and all postprocessing run CPU-only. The standalone parity
process starts only inside inference-ctl's RTX3060 reservation. Scientific
targets may fail; execution and provenance failures stop the sequence visibly.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
COORDINATOR = "experiments.structure_net.jspace_allblocks_v2"
ARTIFACT = "experiments.structure_net.jspace_allblocks_artifact"
FINALIZER = "experiments.neural_architecture_lab.finalize_jspace_allblocks"
SCHEMA = "jspace.allblocks-completion-supervisor.v1"
CARD = "GPU-0370fdb0=11264"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""): digest.update(block)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())


def event(root, stage, status, **values):
    row = {"time": datetime.now(timezone.utc).isoformat(), "stage": stage, "status": status, **values}
    with (Path(root) / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(row, allow_nan=False), flush=True)


def environment(gpu=False):
    value = {**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "PIXI_CACHE_DIR": "/tmp/structure-net-pixi-cache", "UV_CACHE_DIR": "/tmp/structure-net-uv-cache",
        "TRITON_CACHE_DIR": "/tmp/structure-net-triton-cache"}
    # inference-ctl assigns visibility to its leased child. This parent does no GPU work.
    if gpu: value.pop("CUDA_VISIBLE_DEVICES", None)
    return value


def commands(root, artifact, report, registry):
    ncm = ["pixi", "run", "--locked", "--offline", "-e", "ncm", "python", "-m"]
    default = ["pixi", "run", "--locked", "--offline", "python", "-m"]
    common = ["--campaign", str(root), "--output", str(artifact)]
    lease = ["/usr/bin/python3", "/usr/local/bin/inference-ctl", "lease", "run", "--allocation", CARD,
        "--minutes", "120", "--reason", "JSPACE-QWEN-ALLBLOCKS-01 standalone artifact parity", "--"]
    result = [("coordinator", ncm + [COORDINATOR, "run", "--root", str(root)], False),
              ("export", ncm + [ARTIFACT, "export", *common], False),
              ("validate", lease + ncm + [ARTIFACT, "validate", *common], True)]
    for stage in ("analyze", "report", "store", "readback", "closeout"):
        result.append((stage, default + [FINALIZER, stage, "--root", str(root), "--report", str(report),
                                        "--registry", str(registry)], False))
    return [{"stage": stage, "command": command, "gpu_lease_launcher": gpu} for stage, command, gpu in result]


def _sources(primary):
    paths = set(map(Path, primary["source_sha256"]))
    paths.update(ROOT / name for name in (
        "experiments/structure_net/jspace_allblocks_finish.py", "tests/structure_net/test_jspace_allblocks_finish.py",
        "experiments/structure_net/jspace_allblocks_artifact.py", "tests/structure_net/test_jspace_allblocks_artifact.py",
        "experiments/neural_architecture_lab/finalize_jspace_allblocks.py",
        "experiments/neural_architecture_lab/finalize_jspace_qwen_affine.py",
        "src/neural_architecture_lab/jspace_allblocks_hypothesis.py", "src/neural_architecture_lab/jspace_qwen_affine_record.py",
        "tests/neural_architecture_lab/test_jspace_allblocks_hypothesis.py"))
    return sorted(paths)


def prepare(root, state_root, artifact, report, registry):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("IC_LEASE"):
        raise ValueError("Supervisor preparation is CPU-only and must not hold a GPU lease")
    root, state_root, artifact, report, registry = map(lambda p: Path(p).resolve(), (root, state_root, artifact, report, registry))
    if any(p.exists() for p in (state_root, artifact, report, registry)):
        raise FileExistsError("New supervisor, artifact, report and registry paths required")
    if (root / "coordinator.jsonl").exists(): raise FileExistsError("Coordinator already launched; this supervisor never starts a second one")
    primary = read(root / "plan.json")
    if (primary["schema"] != "jspace.qwen-parallel.v2" or primary["systems_only"] or
            primary["fingerprint"] != digest({k: v for k, v in primary.items() if k != "fingerprint"})):
        raise ValueError("Prepared primary v2 campaign with valid fingerprint required")
    source_hashes = {str(p): sha(p) for p in _sources(primary)}
    if any(source_hashes[path] != value for path, value in primary["source_sha256"].items()):
        raise ValueError("Primary frozen source changed")
    plan = {"schema": SCHEMA, "root": str(root), "artifact": str(artifact), "report": str(report), "registry": str(registry),
        "primary_fingerprint": primary["fingerprint"], "primary_plan_sha256": sha(root / "plan.json"),
        "source_sha256": source_hashes, "commands": commands(root, artifact, report, registry),
        "policy": "No retries, no overwrites, no Git operations; completed negative scientific results still export and finalize"}
    plan["fingerprint"] = digest(plan)
    state_root.mkdir(parents=True); (state_root / "logs").mkdir(); (state_root / "stages").mkdir()
    for path in _sources(primary):
        target = state_root / "producer" / path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    write(state_root / "plan.json", plan)
    return {"status": "prepared", "fingerprint": plan["fingerprint"], "stages": len(plan["commands"])}


def verify(state_root):
    state_root = Path(state_root); plan = read(state_root / "plan.json")
    if plan["schema"] != SCHEMA or plan["fingerprint"] != digest({k: v for k, v in plan.items() if k != "fingerprint"}):
        raise ValueError("Supervisor plan changed")
    if sha(Path(plan["root"]) / "plan.json") != plan["primary_plan_sha256"]: raise ValueError("Primary plan changed")
    if plan["commands"] != commands(plan["root"], plan["artifact"], plan["report"], plan["registry"]):
        raise ValueError("Supervisor launch sequence changed")
    for name, value in plan["source_sha256"].items():
        if sha(name) != value or sha(state_root / "producer" / Path(name).relative_to(ROOT)) != value:
            raise ValueError("Frozen completion source changed: " + name)
    return plan


def _hashes(paths): return {str(path): sha(path) for path in paths}


def _same_fingerprint(value, key, expected):
    if value.get(key) != expected: raise ValueError("Output campaign fingerprint mismatch")


def check_outputs(stage, plan):
    root, artifact = Path(plan["root"]), Path(plan["artifact"]); fingerprint = plan["primary_fingerprint"]
    if stage == "coordinator":
        result = read(root / "campaign_results.json"); _same_fingerprint(result, "fingerprint", fingerprint)
        if result.get("status") != "completed" or result.get("confirmed") is not False or result != read(root / "evaluation/result.json"):
            raise ValueError("Campaign/evaluation did not complete consistently")
        required = ["evaluation"]
        for group in range(6):
            prefix = f"groups/group_{group:02d}"
            required += [prefix + "/" + phase for phase in ("capture", "basis", "recovery")]
            required += [prefix + f"/block_{layer:02d}/fit_result" for layer in range(30 - group * 6, 36 - group * 6)]
        for name in required:
            manifest = read(root / name / "manifest.json"); _same_fingerprint(manifest, "fingerprint", fingerprint)
            if manifest.get("status") != "completed": raise ValueError("Incomplete primary stage: " + name)
        evaluation = read(root / "evaluation/manifest.json")
        for name, expected in evaluation["artifacts"].items():
            if sha(root / name) != expected: raise ValueError("Changed evaluation artifact: " + name)
        return _hashes([root / "campaign_results.json", root / "evaluation/manifest.json"])
    if stage == "export":
        value = read(artifact / "manifest.json"); _same_fingerprint(value, "source_fingerprint", fingerprint)
        for name, expected in value["files"].items():
            if sha(artifact / name) != expected: raise ValueError("Exported artifact changed: " + name)
        return _hashes([artifact / "manifest.json"])
    if stage == "validate":
        value = read(artifact / "validation.json"); _same_fingerprint(value, "source_fingerprint", fingerprint)
        if value.get("status") != "completed" or value.get("passed") is not True: raise ValueError("Standalone parity failed")
        for name, expected in value["inputs"].items():
            if sha(name) != expected: raise ValueError("Standalone validation input changed")
        for name, expected in value["artifacts"].items():
            if sha(artifact / name) != expected: raise ValueError("Standalone validation output changed")
        return _hashes([artifact / "validation.json"])
    analysis = read(root / "closeout_analysis.json")
    _same_fingerprint(analysis["plan"], "fingerprint", fingerprint)
    if analysis["result"] != read(root / "campaign_results.json"): raise ValueError("Analysis changed producer evidence")
    if stage == "analyze": return _hashes([root / "closeout_analysis.json"])
    if stage == "report": return _hashes([Path(plan["report"])])
    registry = read(plan["registry"])
    if registry["record"] != analysis["record"] or registry["analysis_sha256"] != sha(root / "closeout_analysis.json"):
        raise ValueError("Registry does not match the completed analysis")
    if registry["report_sha256"] != sha(plan["report"]): raise ValueError("Registry/report mismatch")
    if stage == "store": return _hashes([Path(plan["registry"])])
    receipt = read(root / "allblocks_registry_readback.json")
    if receipt != registry["storage"] or receipt.get("readback_verified") is not True: raise ValueError("Independent registry readback mismatch")
    if stage == "readback": return _hashes([root / "allblocks_registry_readback.json"])
    final = read(root / "allblocks_finalization.json")
    if final.get("status") != "completed" or final.get("confirmed") is not False or final["record"] != analysis["record"] or final["registry_readback"] != receipt:
        raise ValueError("Final closeout provenance mismatch")
    return _hashes([root / "allblocks_finalization.json"])


def execute(state_root, plan, *, runner=subprocess.run, checker=check_outputs):
    """Execute one immutable sequence; injectable process/check functions serve CPU tests."""
    state_root = Path(state_root); current = "startup"
    write(state_root / "run_started.json", {"pid": os.getpid(), "fingerprint": plan["fingerprint"],
                                           "time": datetime.now(timezone.utc).isoformat()})
    try:
        for item in plan["commands"]:
            current = item["stage"]; verify(state_root)
            event(state_root, current, "started", command=item["command"])
            with (state_root / "logs" / (current + ".log")).open("x") as stream:
                result = runner(item["command"], cwd=ROOT, env=environment(item["gpu_lease_launcher"]),
                                stdout=stream, stderr=subprocess.STDOUT)
            status = {"stage": current, "returncode": result.returncode, "status": "failed" if result.returncode else "completed"}
            if not result.returncode:
                try: status["output_sha256"] = checker(current, plan)
                except Exception as error:
                    status.update(status="failed", error=repr(error))
            write(state_root / "stages" / (current + ".json"), status)
            event(state_root, current, status["status"], returncode=result.returncode, error=status.get("error"))
            if status["status"] != "completed": raise RuntimeError(f"{current} failed; see supervisor logs and stage receipt")
        value = {"status": "completed", "fingerprint": plan["fingerprint"], "primary_fingerprint": plan["primary_fingerprint"],
                 "stages": len(plan["commands"]), "scientific_claims": "Use the final report; completion is not a successful-quality claim"}
        write(state_root / "completed.json", value); return value
    except BaseException as error:
        write(state_root / "failed.json", {"status": "failed", "stage": current, "error": repr(error), "fingerprint": plan["fingerprint"]})
        event(state_root, current, "supervisor_failed", error=repr(error)); raise


def run(state_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("IC_LEASE"):
        raise ValueError("Supervisor is CPU-only and must not hold a GPU lease")
    plan = verify(state_root)
    if (Path(plan["root"]) / "coordinator.jsonl").exists(): raise FileExistsError("Coordinator already launched")
    return execute(state_root, plan)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--state-root", required=True)
    for key in ("root", "artifact", "report", "registry"): parser.add_argument("--" + key)
    args = parser.parse_args()
    if args.stage == "prepare":
        if any(getattr(args, name) is None for name in ("root", "artifact", "report", "registry")):
            parser.error("prepare requires --root --artifact --report --registry")
        result = prepare(args.root, args.state_root, args.artifact, args.report, args.registry)
    else: result = run(args.state_root)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__": main()
