"""Reproduce released-checkpoint XSum scoring through unchanged author scripts."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net.ncm_released_xsum_control import DATA, NAMES, read_pickle

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-released-xsum-benchmark-plan.md")
EXTRAS = ROOT / "released_benchmark_dependencies_v1"
PILOT = ROOT / "20261022_released_xsum_benchmark_pilot_v1"
FULL = ROOT / "20261022_released_xsum_benchmark_full_v1"
CONTROL = ROOT / "20261022_released_xsum_control_v2"
GPT2 = Path("/home/rabbit/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e")
TARGETS = {"rouge1": 35.0, "rouge2": 12.8, "rougeL": 28.0}
MARGIN = .5


def compatible(scores):
    if set(scores) != set(TARGETS) or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("invalid ROUGE endpoints")
    return all(abs(scores[k]-v) <= MARGIN for k, v in TARGETS.items())


def validate_outputs(output, references):
    n = len(references)
    if output["idxs"] != list(range(n)) or len(output["hyps"]) != n or output["refs"] != references:
        raise ValueError("missing, reordered, duplicate or mismatched benchmark rows")
    if not all(isinstance(h, str) and h.strip() for h in output["hyps"]):
        raise ValueError("empty or malformed hypothesis")


def parse_author_metrics(log):
    lines = log.splitlines()
    header = next(i for i, line in enumerate(lines) if line.split() == ["R-1", "R-2", "R-L", "genPPL", "words"])
    values = [float(v) for v in lines[header+1].split()]
    if len(values) != 5 or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("nonfinite/malformed author metrics")
    return dict(zip([*TARGETS, "gen_ppl", "mean_words"], values))


def score_rows(output):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(list(TARGETS), use_stemmer=True)
    rows = [{"index": i, **{k: v.fmeasure*100 for k, v in scorer.score(ref, hyp).items()}}
            for i, ref, hyp in zip(output["idxs"], output["refs"], output["hyps"])]
    return rows, {k: sum(r[k] for r in rows)/len(rows) for k in TARGETS}


def run(pilot):
    import torch
    from experiments.structure_net.ncm_llama import DEPENDENCIES, INITIAL, VENDOR
    from experiments.structure_net.ncm_paper_controls import producer
    root, n = (PILOT, 32) if pilot else (FULL, 11334)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("select physical GPU0 before startup")
    if not torch.cuda.is_available() or "3060" not in torch.cuda.get_device_name():
        raise ValueError("planned CUDA device unavailable")
    sys.path.insert(0, str(EXTRAS.resolve()))
    import nltk
    import transformers
    import rouge_score
    import absl
    import numpy
    inputs = [DATA/name for name in (*NAMES, "result.json")]
    inputs += [INITIAL/name for name in ("config.yaml", "tokenizer.pt", "generator.pt")]
    inputs += sorted(p for p in GPT2.iterdir() if p.is_file())
    inputs += [CONTROL / "outputs.json", CONTROL / "result.json"]
    receipt = q.read(DATA / "result.json")
    for name in NAMES:
        if file_hash(DATA/name) != receipt["files"][name]["sha256"]:
            raise ValueError("released data changed")
    if not q.read(CONTROL / "result.json")["author_local_exact"]:
        raise ValueError("prior released-domain parity required")
    p = producer(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "full_released_checkpoint_benchmark",
        inputs, {"cases": n, "batch": 32, "shards": 1, "precision": "FP32", "training": False,
                 "targets": TARGETS, "compatibility_margin_points": MARGIN})
    sources = [Path(__file__), PLAN, Path("experiments/structure_net/ncm_released_xsum_control.py")]
    sources += sorted(EXTRAS.rglob("*.py")) + sorted(DEPENDENCIES.rglob("*.py"))
    # Preserve the isolated scorer sources as well as hashes. Bytecode is excluded.
    for i, path in enumerate(sources):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"benchmark_{i:04d}_{path.name}").write_bytes(path.read_bytes())
    p["hypothesis_id"] = "ncm-released-xsum-benchmark-v1"
    p["runtime"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                    "numpy": numpy.__version__, "nltk": nltk.__version__, "rouge_score": "0.1.2",
                    "absl": absl.__version__, "python": sys.version,
                    "rouge_module": rouge_score.__file__, "gpu": torch.cuda.get_device_name()}
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    try:
        if not pilot:
            previous, result = q.read(PILOT / "producer.json"), q.read(PILOT / "result.json")
            if result["status"] != "passed":
                raise ValueError("matching batch32 lifecycle pilot required")
            if any(previous[k] != p[k] for k in ("source_hashes", "input_hashes", "runtime")):
                raise ValueError("pilot/full producer drift")
            verify(previous)
        environment = {**os.environ, "NCM_ROOT": str(VENDOR.resolve()),
            "PYTHONPATH": os.pathsep.join(map(str, [EXTRAS.resolve(), DEPENDENCIES.resolve(),
                                                       Path("src").resolve(), Path.cwd()]))}
        commands = {
            "generate": [sys.executable, str(VENDOR / "scripts/eval_shard.py"), "--config", str(INITIAL / "config.yaml"),
                "--checkpoint-dir", str(INITIAL), "--data-dir", str(DATA), "--n", str(n), "--bs", "32",
                "--out", str(root / "author.pickle")],
            "merge": [sys.executable, str(VENDOR / "scripts/eval_merge.py"), "--config", str(INITIAL / "config.yaml"),
                "--glob", str(root / "author.pickle")]}
        write_json(root / "commands.json", commands)
        times = {}
        references = read_pickle(DATA / NAMES[1])[:n]
        for stage, command in commands.items():
            verify(p)
            start = time.monotonic()
            print(json.dumps({"stage": stage, "cases": n, "root": str(root)}), flush=True)
            with (root / (stage+".log")).open("w") as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            times[stage] = time.monotonic()-start
            if stage == "generate":
                output = read_pickle(root / "author.pickle")
                validate_outputs(output, references)
                previous_texts = [r["author_text"] for r in q.read(CONTROL / "outputs.json")]
                if [h.strip() for h in output["hyps"][:32]] != previous_texts:
                    raise ValueError("batch32 output differs from prior batch4 control")
                if not pilot and output["hyps"][:32] != read_pickle(PILOT / "author.pickle")["hyps"]:
                    raise ValueError("pilot/full output drift")
        author = parse_author_metrics((root / "merge.log").read_text())
        rows, scores = score_rows(output)
        if any(abs(scores[k]-author[k]) > .0051 for k in TARGETS):
            raise ValueError("independent/author ROUGE discrepancy")
        write_json(root / "rouge_per_example.json", rows)
        verify(p)
        passed = compatible(scores) if not pilot else None
        result = {"status": "passed" if pilot else "completed", "hypothesis_id": p["hypothesis_id"],
            "classification": "systems_only" if pilot else ("checkpoint_scores_compatible" if passed else "checkpoint_scores_mismatch"),
            "cases": n, "rouge": scores, "author_printed_metrics": author,
            "paper_targets": TARGETS, "differences_points": {k: scores[k]-TARGETS[k] for k in TARGETS},
            "checkpoint_compatibility_passed": passed, "compatibility_margin_points": MARGIN,
            "original_scripts_unchanged": True, "prior32_exact": True, "stage_seconds": times,
            "task_accuracy": None, "training_replicated": False, "trained": False, "confirmed": False,
            "final_test_opened": False, "speed_tested": False, "producer_fingerprint": p["fingerprint"],
            "deviations": ["one GPU shard rather than config two", "release nine levels versus paper eight NFE"],
            "output_hashes": {str(root/name): file_hash(root/name) for name in
                              ("author.pickle", "generate.log", "merge.log", "commands.json", "rouge_per_example.json")}}
        write_json(root / "result.json", result)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        write_json(root / "failure.json", {"status": "failed", "exception": repr(exc),
            "producer_fingerprint": p["fingerprint"], "training_replicated": False, "task_accuracy": None})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "full"])
    run(parser.parse_args().command == "pilot")
