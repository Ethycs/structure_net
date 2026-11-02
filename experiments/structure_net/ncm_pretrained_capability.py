"""Run untouched author NCM inference and independently score Sudoku."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import runpy
import sys
import time
import urllib.request

ROOT = Path("data/experiments/ncm_pretrained_capability/20261030_native_v1")
VENDOR = Path("data/experiments/ncm_llama/vendor/ncm-20261021")
DEPS = Path("data/experiments/ncm_llama/dependencies")
LM1B = Path("data/experiments/ncm_paper_regime/20261024_lm1b_native_release_v1/files/lm1b")
BERT = Path("data/experiments/ncm_paper_regime/20261024_long_context_release_inventory_v1/bert-tokenizer")
PLAN = Path("docs/07 - Status Reports/ncm/2026-10-30_ncm-pretrained-capability-plan.md")
REVISIONS = {"models": "d342ce8aaea58830eef3b8ad473664673e41f692",
             "datasets": "6580ba41a4b0a976dde876fbe84d25327617b0e6"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f"dataset requires forbidden class {module}.{name}")


def read_pickle(path):
    return PrimitiveUnpickler(io.BytesIO(Path(path).read_bytes())).load()


def board(ids, *, prompt=False):
    ids = list(ids)
    if prompt:
        if len(ids) != 91 or ids[0] != 11 or ids[-1] != 11:
            return None
        ids = ids[1:-1]
    if len(ids) != 89:
        return None
    cells = []
    for i, value in enumerate(ids):
        if type(value) is not int:
            return None
        if i % 10 == 9:
            if value != 10:
                return None
        elif (0 if prompt else 1) <= value <= 9:
            cells.append(value)
        else:
            return None
    return cells


def score_board(raw, puzzle, reference):
    p, r, g = board(puzzle, prompt=True), board(reference), board(raw)
    if p is None or r is None:
        raise ValueError("malformed official puzzle/reference")
    valid = False
    clues = False
    if g is not None:
        groups = [g[i:i + 9] for i in range(0, 81, 9)]
        groups += [g[i::9] for i in range(9)]
        groups += [[g[(br + dr) * 9 + bc + dc] for dr in range(3) for dc in range(3)]
                   for br in (0, 3, 6) for bc in (0, 3, 6)]
        valid = all(set(group) == set(range(1, 10)) for group in groups)
        clues = all(a == 0 or a == b for a, b in zip(p, g))
    return {"format_valid": g is not None, "grid_valid": valid,
            "clues_preserved": clues, "solved": valid and clues,
            "exact": list(raw) == list(reference),
            "correct_cells": sum(a == b for a, b in zip(g, r)) if g else 0,
            "total_cells": 81}


def download():
    out = ROOT / "release"
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for kind, repo in (("models", "ncm-checkpoints"), ("datasets", "ncm-datasets")):
        rev = REVISIONS[kind]
        api = f"https://huggingface.co/api/{kind}/hazyresearch/{repo}/tree/{rev}/sudoku?recursive=true"
        inventory = json.load(urllib.request.urlopen(api, timeout=30))
        write(out / f"{kind}_inventory.json", {"url": api, "entries": inventory})
        for meta in inventory:
            name = meta["path"]
            if meta["type"] != "file" or not any(name.startswith(f"sudoku/{x}/") for x in ("easy", "hard")):
                continue
            base = "https://huggingface.co/" + ("datasets/" if kind == "datasets" else "")
            url = f"{base}hazyresearch/{repo}/resolve/{rev}/{name}"
            path = out / kind / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as response, path.open("xb") as f:
                while part := response.read(8 * 1024 * 1024):
                    f.write(part)
            digest = sha(path)
            assert path.stat().st_size == meta["size"], name
            if "lfs" in meta:
                assert digest == meta["lfs"]["oid"], name
            else:
                blob = b"blob " + str(meta["size"]).encode() + b"\0" + path.read_bytes()
                assert hashlib.sha1(blob).hexdigest() == meta["oid"], name
            records.append({"path": str(path), "sha256": digest, "bytes": meta["size"], "url": url})
            print(json.dumps(records[-1]), flush=True)
    write(out / "downloads.json", records)


def run(task, stage):
    import torch
    from unittest.mock import patch
    from transformers import AutoTokenizer
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("select physical GPU0 before Python")
    assert torch.cuda.is_available() and "3060" in torch.cuda.get_device_name(0)
    assert torch.cuda.mem_get_info()[0] > 4_000_000_000
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    for path in (DEPS, VENDOR):
        sys.path.insert(0, str(path.resolve()))
    from models.next_concept import NextConceptTransformer
    out = ROOT / f"{task}_{stage}"
    out.mkdir(parents=True, exist_ok=False)
    if task == "lm1b":
        checkpoint = LM1B / "small"
        files = [checkpoint / "config.yaml", checkpoint / "generator.pt", LM1B / "tokenizer.pt"]
        expected = {"generator.pt": "b0789d5a2aad9114d40976d26bd1af09d8cea0b0bec7d385b6a44739f77256fe",
                    "tokenizer.pt": "d88aa17b66d730f2cb1fb31c5dbe63888a10da47ed8bea29dcb1f89423c3059c"}
        for path in files[1:]:
            assert sha(path) == expected[path.name]
        n, bs = (8 if stage == "pilot" else 16), 8
        more = ["--nshards", "5", "--shard", "0"]  # Exactly seed 42 from five published seeds.
        data = None
    else:
        checkpoint = ROOT / "release/models/sudoku" / task
        data = ROOT / "release/datasets/sudoku" / task
        files = [checkpoint / x for x in ("config.yaml", "generator.pt", "tokenizer.pt")]
        files += [data / "puzzles.pickle", data / "solutions.pickle"]
        expected = {r["path"]: r["sha256"] for r in json.loads((ROOT / "release/downloads.json").read_text())}
        for path in files:
            assert sha(path) == expected[str(path)]
        n, bs = (2 if stage == "pilot" else 2000), (2 if stage == "pilot" else 32)
        more = ["--data-dir", str(data)]
        puzzles, refs = read_pickle(data / "puzzles.pickle"), read_pickle(data / "solutions.pickle")
        assert len(puzzles) == len(refs) == 2000
        assert all(score_board(r, p, r)["solved"] for p, r in zip(puzzles, refs))
    config = checkpoint / "config.yaml"
    vendor_config = VENDOR / "config/eval" / ("lm1b-small.yaml" if task == "lm1b" else f"sudoku-{task}.yaml")
    assert config.read_bytes() == vendor_config.read_bytes()
    sources = [Path(__file__), PLAN, *sorted(VENDOR.rglob("*.py"))]
    bindings = {str(p): sha(p) for p in [*files, *sources]}
    write(out / "producer.json", {"task": task, "stage": stage, "n": n, "batch": bs,
          "hashes": bindings, "revision": REVISIONS, "torch": torch.__version__,
          "gpu": torch.cuda.get_device_name(0), "physical_gpu": 0, "precision": "FP32", "TF32": False,
          "training": False, "application_final_test": False})
    source_dir = out / "producer_source"
    source_dir.mkdir()
    for i, path in enumerate(sources):
        (source_dir / f"{i}_{path.name}").write_bytes(path.read_bytes())
    orig_load, orig_tokenizer = torch.load, AutoTokenizer.from_pretrained
    orig_generate = NextConceptTransformer.generate
    codes = []

    def safe_load(*args, **kwargs):
        kwargs["weights_only"] = True
        return orig_load(*args, **kwargs)

    def local_tokenizer(model, *args, **kwargs):
        if str(model) == "bert-base-uncased":
            model, kwargs["local_files_only"] = BERT, True
        return orig_tokenizer(model, *args, **kwargs)

    def capture_generate(self, *args, **kwargs):
        result = orig_generate(self, *args, **kwargs)
        codes.extend(result.detach().cpu().tolist())
        return result

    argv = ["eval_shard.py", "--config", str(config), "--checkpoint-dir", str(checkpoint),
            "--n", str(n), "--bs", str(bs), "--out", str(out / "author_output.pickle"), *more]
    write(out / "invocation.json", {"argv": argv, "wrapper_changes": ["weights_only loading",
          "same local BERT tokenizer", "capture generated code IDs"]})
    start = time.monotonic()
    with patch.object(sys, "argv", argv), patch.object(torch, "load", safe_load), \
            patch.object(AutoTokenizer, "from_pretrained", local_tokenizer), \
            patch.object(NextConceptTransformer, "generate", capture_generate):
        scope = runpy.run_path(str(VENDOR / "scripts/eval_shard.py"), run_name="__main__")
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    rows = []
    for i, (text, raw, code) in enumerate(zip(scope["hyps"], scope["RAW"], codes, strict=True)):
        row = {"index": i, "text": text, "raw_ids": raw, "codes": code}
        if data:
            row.update(puzzle=puzzles[i], reference=refs[i],
                       scores=score_board(raw, puzzles[i], refs[i]))
        rows.append(row)
    assert len(rows) == n
    write(out / "outputs.json", rows)
    result = {"task": task, "n": n, "elapsed_seconds_including_model_load": elapsed,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "speed_comparison": False,
              "generator_parameters": sum(p.numel() for p in scope["net"].parameters()),
              "codec_parameters": sum(p.numel() for p in scope["tk"].vqvae.parameters())}
    if data:
        result["counts"] = {k: sum(r["scores"][k] for r in rows) for k in rows[0]["scores"]}
    else:
        result["nonempty"] = sum(bool(r["text"].strip()) for r in rows)
        result["accuracy"] = None
    if stage == "full":
        pilot = json.loads((ROOT / f"{task}_pilot/outputs.json").read_text())
        result["pilot_comparison"] = {"n": len(pilot), "exact_raw_ids": sum(
            p["raw_ids"] == f["raw_ids"] for p, f in zip(pilot, rows))}
    assert all(sha(Path(p)) == digest for p, digest in bindings.items())
    write(out / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["download", "pilot", "full"])
    ap.add_argument("--task", choices=["easy", "hard", "lm1b"])
    args = ap.parse_args()
    if args.stage == "download":
        download()
    else:
        if args.task is None:
            ap.error("--task is required for inference")
        run(args.task, args.stage)
