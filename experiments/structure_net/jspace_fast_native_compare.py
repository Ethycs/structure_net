"""Matched isolated native-vs-recovered-fast final-block CUDA-graph timings."""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import statistics
import time

import numpy as np
import torch

from . import jspace_compact_performance as perf
from .jspace_bench_data import sha, write
from .jspace_full_lease import wait_ready

ROOT = perf.ROOT
DEFAULT_ROOT = ROOT / "data/experiments/jspace_compact_performance/20261114_native_compare_v1"
STUDY_ID = "JSPACE-FAST-NATIVE-COMPARE-01"
ARM, CANDIDATE = "r1536_h64_gb", "combined"
CONFIG = {"arm": ARM, "candidate": CANDIDATE, "paired_repeats": 5,
          "modes": ["prefill", "decode"], "execution": "cuda_graph",
          "scope": "eight fixed cached development inputs; isolated block35; no quality selection"}


def sources():
    return sorted(set(perf.sources() + [Path(__file__), ROOT / "tests/structure_net/test_jspace_fast_native_compare.py"]))


def prepare(root, parent=perf.PARENT):
    perf.base.cpu_only()
    if root.exists():
        raise FileExistsError(root)
    prior = perf.base.verify(parent, inputs=True)
    for stage in ("train_" + ARM, "reload"):
        perf.base.checked(parent, stage)
    names = ["plan.json", "tasks.json", "fit.pt", "relation.pt", ARM + ".pt",
             "training_" + ARM + ".json", "train_" + ARM + "_manifest.json", "reload_manifest.json"]
    inputs = {**prior["input_sha256"], **{str(parent / name): sha(parent / name) for name in names}}
    items_path = Path(prior["cache_root"]) / "items.json"
    selected = perf.selected_cases(json.loads(items_path.read_text())["items"])
    inputs[str(items_path)] = sha(items_path)
    plan = {"study_id": STUDY_ID, "config": CONFIG, "helper_config": perf.CONFIG,
            "candidate_options": perf.CANDIDATES[CANDIDATE], "case_identities": [c["identity"] for c in selected],
            "parent_root": str(parent), "parent_fingerprint": prior["fingerprint"],
            "model": prior["model"], "cache_root": prior["cache_root"], "label_ids": prior["label_ids"],
            "created_at": datetime.now(timezone.utc).isoformat(), "input_sha256": inputs,
            "source_sha256": {str(p): sha(p) for p in sources()}}
    plan["fingerprint"] = perf.base.digest(plan)
    root.mkdir(parents=True)
    for path in sources():
        target = root / "producer" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    write(root / "plan.json", plan)
    return {"root": str(root), "fingerprint": plan["fingerprint"], "cases": len(selected)}


def verify(root, inputs=False, model=False):
    plan = json.loads((root / "plan.json").read_text())
    if (plan["study_id"] != STUDY_ID or plan["config"] != CONFIG or plan["helper_config"] != perf.CONFIG
            or plan["candidate_options"] != perf.CANDIDATES[CANDIDATE]
            or plan["fingerprint"] != perf.base.digest({k: v for k, v in plan.items() if k != "fingerprint"})):
        raise ValueError("Changed native-comparison protocol")
    for name, expected in plan["source_sha256"].items():
        if sha(name) != expected or sha(root / "producer" / Path(name).relative_to(ROOT)) != expected:
            raise ValueError("Changed native-comparison producer: " + name)
    if inputs:
        for name, expected in plan["input_sha256"].items():
            if sha(name) != expected:
                raise ValueError("Changed native-comparison input: " + name)
    if model:
        for name, expected in plan["model"]["files"].items():
            if sha(Path(plan["model"]["path"]) / name) != expected:
                raise ValueError("Changed pinned native model")
    return plan


class Stage(perf.Stage):
    def __init__(self, root):
        self.plan = verify(root, inputs=True, model=True)
        self.lease = wait_ready([0]); self.guard = wait_ready
        self.root, self.name, self.path = root, "native_compare", root / "native_compare"
        self.path.mkdir(exist_ok=False); self.started = time.perf_counter(); self.counts = Counter()
        torch.set_num_threads(4); torch.manual_seed(perf.CONFIG["seed"]); torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        self.parent = Path(self.plan["parent_root"])
        self.model, self.tokenizer = perf.base.load_small_qwen(self.plan["model"]["path"], device="cuda:0", dtype=torch.bfloat16)
        self.native = copy.deepcopy(self.model.model.layers[-1]).cpu()
        self.native_norm = {k: v.detach().cpu().clone() for k, v in self.model.model.norm.state_dict().items()}
        cache = Path(self.plan["cache_root"])
        self.items = json.loads((cache / "items.json").read_text())["items"]
        self.input = np.load(cache / "boundary_35.npy", mmap_mode="r")
        self.fit = torch.load(self.parent / "fit.pt", map_location="cpu", weights_only=True)
        self.relation = torch.load(self.parent / "relation.pt", map_location="cpu", weights_only=True)
        write(self.path / "lease.json", {k: self.lease[k] for k in ("id", "holder", "allocation", "state", "ready", "end")})

    def finish(self, result):
        self.guard([0]); verify(self.root)
        peak = torch.cuda.max_memory_reserved()
        if peak > sum(self.lease["allocation"].values()) * 1024**2:
            raise RuntimeError("Reservation exceeded")
        write(self.path / "result.json", result)
        manifest = {"status": "completed", "study_id": STUDY_ID, "fingerprint": self.plan["fingerprint"],
                    "elapsed_seconds": time.perf_counter() - self.started, "peak_reserved_bytes": peak,
                    "lease_id": self.lease["id"], "gpu": torch.cuda.get_device_name(0),
                    "artifact_sha256": {str(p.relative_to(self.root)): sha(p) for p in self.path.rglob("*") if p.is_file()}}
        write(self.root / "native_compare_manifest.json", manifest)
        return {**manifest, "summary": result["summary"]}


def pair_summary(rows):
    result = {}
    for metric in ("wall_seconds", "cuda_event_seconds"):
        values = {variant: statistics.median(row[metric] for row in rows if row["variant"] == variant)
                  for variant in ("reference", "candidate")}
        if min(values.values()) <= 0:
            raise ValueError("Positive measured timings required")
        result[metric] = {"native": values["reference"], "fast": values["candidate"],
                          "speedup": values["reference"] / values["candidate"]}
    return result


def aggregate(pairs):
    result = {}
    for mode in CONFIG["modes"]:
        selected = [pair for pair in pairs if pair["mode"] == mode]
        result[mode] = {metric: {"native_seconds_sum": sum(p["summary"][metric]["native"] for p in selected),
                                "fast_seconds_sum": sum(p["summary"][metric]["fast"] for p in selected)}
                        for metric in ("wall_seconds", "cuda_event_seconds")}
        for value in result[mode].values():
            value["speedup"] = value["native_seconds_sum"] / value["fast_seconds_sum"]
    return result


@torch.inference_mode()
def run(root):
    stage = Stage(root)
    cases = perf.selected_cases(stage.items)
    if [c["identity"] for c in cases] != stage.plan["case_identities"]:
        raise ValueError("Frozen comparison inputs changed")
    native = stage.model.model.layers[-1]  # Remains resident while the compact block is installed.
    compact = stage.install_arm(ARM)
    candidate = perf.candidate_factory(compact, CANDIDATE)
    stage.model.model.layers[-1] = candidate; del compact
    pairs = []
    for number, case in enumerate(cases):
        x = stage.x(case); perf.ensure_fusion(candidate, CANDIDATE, x)
        for mode in CONFIG["modes"]:
            calls = {"reference": perf.block_call(stage, native, x, mode),
                     "candidate": perf.block_call(stage, candidate, x, mode)}
            graphs, setup = {}, {}
            for key, call in calls.items():
                graphs[key], setup[key] = perf.warm_and_graph(call)
            error = perf.numerical_error(calls["reference"](), calls["candidate"]())
            rows = []
            for repeat, order in perf.paired_orders(CONFIG["paired_repeats"]):
                for key in order:
                    rows.append({"repeat": repeat, "order": list(order), "variant": key,
                                 **perf.timed(graphs[key].replay)})
            pair = {"identity": case["identity"], "tokens": x.shape[1], "mode": mode,
                    "native_graph": setup["reference"], "fast_graph": setup["candidate"],
                    "native_vs_fast_error": error, "summary": pair_summary(rows), "timings": rows}
            pairs.append(pair); write(stage.path / f"pair_{number:02d}_{mode}.json", pair)
            del calls, graphs, call; gc.collect(); torch.cuda.empty_cache(); stage.guard([0])
        stage.event("comparison", completed=number + 1, total=len(cases))
    return stage.finish({"reference": "original native block35", "candidate": ARM + "/" + CANDIDATE,
        "execution": "both CUDA graphs; identical cached inputs; own eager/graph parity checked",
        "scope": CONFIG["scope"], "quality_selection": False, "final_norm_in_timing": False,
        "summary_method": "ratio of sums of per-input median graph latencies; host and device reported separately",
        "summary": aggregate(pairs), "pairs": pairs})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run")); parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--parent", type=Path, default=perf.PARENT); args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.parent) if args.command == "prepare" else run(args.root), allow_nan=False), flush=True)
