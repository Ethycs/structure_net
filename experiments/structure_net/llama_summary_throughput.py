"""Reproducible bounded-summary fixtures and isolated native-runtime cells."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from experiments.structure_net.prepare_llama_cotenancy import MODEL_PATH, sha256, write_json

HYPOTHESIS_ID = "llama32-3b-summary-throughput-v1"
SCHEMA_VERSION = "nal.llama-summary-throughput.v1"
ROOT = Path("data/experiments/llama_summary_throughput")
VENDOR = Path("data/experiments/llama_tensorized_cotenancy/llama.cpp")
BINARY = ROOT / "20261013_tools/summary_runner"
PLAN = Path("docs/07 - Status Reports/2026-10-13_llama32-summary-throughput-plan.md")
CPP = Path(__file__).with_suffix(".cpp")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def make_fragment(seed, index, minimum_tokens, tokenizer, hierarchy=False):
    rng = random.Random(seed * 100 + index)
    owner = rng.choice(["Maya Chen", "Omar Reed", "Priya Shah", "Leo Grant", "Nora Bell", "Iris Park"])
    old_owner = rng.choice(["Alex Stone", "Robin Cole", "Sam Holt"])
    amount = rng.randrange(12, 89) * 100
    old_amount = amount + 1700
    day = rng.randrange(11, 25)
    date = f"November {day}"
    old_date = f"November {day - 4}"
    project = f"Project {rng.choice(['Cedar', 'Maple', 'Birch', 'Elm', 'Aspen'])} {index + 1}"
    final = f"{project}: {owner} owns the launch, the approved budget is ${amount:,}, and the final deadline is {date}."
    source = [f"User: We need a final handoff summary for {project}.",
              f"Assistant: The draft proposed {old_owner} as owner, a ${old_amount:,} budget, and a {old_date} deadline.",
              "User: Those are provisional proposals. Do not treat them as approved decisions."]
    filler = [
        "Assistant: The team discussed the document layout and where to put the background notes.",
        "User: Keep the handoff focused on the approved owner, budget and deadline. Leave routine discussion out.",
        "Assistant: The meeting also covered headings, agenda order, note formatting and how to archive drafts.",
        "User: Background discussion does not change the approved project decisions.",
    ]
    if hierarchy:
        source = [f"Earlier summary: {project} initially had a provisional proposal for {old_owner}, ${old_amount:,}, and {old_date}."]
    while len(tokenizer.encode("\n".join(source), add_special_tokens=False)) < minimum_tokens - 100:
        source.append(rng.choice(filler))
    source += ["User: Final decision. " + final,
               f"Assistant: Confirmed. The earlier proposal is superseded; use {owner}, ${amount:,}, and {date}."]
    text = "\n".join(source)
    return {"id": f"F{index+1:02d}", "source": text, "required": {"owner": owner, "budget": str(amount), "deadline": date},
            "superseded": {"owner": old_owner, "budget": str(old_amount), "deadline": old_date},
            "reference_fact": final, "source_tokens": len(tokenizer.encode(text, add_special_tokens=False))}


def make_fixture(tokenizer, seed, count, target, kind, split):
    hierarchy = kind == "hierarchy"
    fragments = [make_fragment(seed, i, max(192, (target - 180) // count), tokenizer, hierarchy) for i in range(count)]
    system = ("Write faithful concise summaries of final decisions. Preserve the approved owner, budget and deadline. "
              "Ignore superseded proposals and routine discussion. Do not invent facts. "
              "For each source fragment output its ID followed by one short paragraph, at most 128 tokens. "
              "Keep fragments separate. Output only these summaries.")
    def render():
        body = "\n\n".join(f"[{f['id']}]\n{f['source']}" for f in fragments)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": body}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    ids = render()
    # Grow with whole irrelevant sentences, retaining natural token boundaries.
    filler = "Assistant: The background discussion covered formatting the routine meeting notes.\n"
    while len(ids) + 20 <= target:
        f = min(fragments, key=lambda x: x["source_tokens"])
        new = filler + f["source"]
        n = len(tokenizer.encode(new, add_special_tokens=False))
        if n > 2048:
            break
        f["source"], f["source_tokens"] = new, n
        ids = render()
    if len(ids) > 6300 or any(f["source_tokens"] > 2048 for f in fragments):
        raise ValueError("fixture exceeds raw workload contract")
    return {"id": f"{split}_{kind}_{seed}_{count}_{target}", "split": split, "kind": kind, "seed": seed,
            "fragments": fragments, "prompt_ids": ids, "prompt_tokens": len(ids), "cap": 512 if hierarchy else 768,
            "stop_eos": True, "target_prompt_tokens": target, "synthetic_proxy": True}


def prepare(path, stage):
    from transformers import AutoTokenizer
    if path.exists():
        raise ValueError("fixtures are immutable")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if stage == "pilot":
        specs = [(103, 1, 512, "raw"), (107, 8, 6280, "raw")]
    else:
        # No outcome-based fixture filtering. Held-out seeds differ from development.
        seeds = (701, 709, 719, 727) if stage == "development" else (1103, 1109, 1117, 1123, 1129)
        specs = [(s, n, t, k) for s in seeds for n, t, k in
                 ((1, 512, "raw"), (1, 2000, "raw"), (8, 6280, "raw"), (1, 1024, "hierarchy"))]
    examples = [make_fixture(tokenizer, *spec, stage) for spec in specs]
    value = {"schema": SCHEMA_VERSION, "stage": stage, "context": 8192, "ubatch": 512,
             "drafts": [0, 4, 8, 16], "repeats": 5, "examples": examples,
             "quality_boundary": "synthetic final-decision summaries; not production application accuracy"}
    write_json(path, value)
    print(json.dumps({"fixtures": str(path), "sha256": sha256(path), "lengths": [e["prompt_tokens"] for e in examples]}))


def gpu_state():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,memory.used,utilization.gpu",
                                    "--format=csv,noheader"], text=True)


def prepare_model(output):
    """Convert the pinned original with a preserved CPU-only producer."""
    if output.exists():
        raise ValueError("choose a new conversion root")
    original = Path("data/models/Llama-3.2-3B-cotenancy/manifest.json")
    manifest = json.loads(original.read_text())
    for name, expected in manifest["files"].items():
        if sha256(MODEL_PATH / name) != expected["sha256"]:
            raise ValueError("original model fingerprint changed: " + name)
    output.mkdir(parents=True)
    dependencies = ROOT / "20261013_conversion_deps"
    sources = [Path(__file__), PLAN, VENDOR / "convert_hf_to_gguf.py"]
    sources += sorted((VENDOR / "conversion").rglob("*.py"))
    sources += sorted((VENDOR / "gguf-py/gguf").rglob("*.py"))
    sources += sorted((dependencies / "sentencepiece").glob("*.so"))
    source_hashes = {str(p): sha256(p) for p in sources}
    snapshot = output / "producer_source"
    snapshot.mkdir()
    for i, path in enumerate(sources):
        shutil.copyfile(path, snapshot / f"{i:03d}_{path.name}")
    target = output / "Llama-3.2-3B-Instruct-F16.gguf"
    cmd = [sys.executable, str(VENDOR / "convert_hf_to_gguf.py"), str(MODEL_PATH),
           "--outfile", str(target), "--outtype", "f16"]
    started = time.time()
    with (output / "conversion.log").open("w") as log:
        process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, "OMP_NUM_THREADS": "2", "PYTHONPATH": str(dependencies.resolve())})
    result = {"schema": SCHEMA_VERSION, "status": "completed" if process.returncode == 0 else "failed",
              "returncode": process.returncode, "input_manifest": manifest, "source_hashes": source_hashes,
              "command": cmd, "cpu_only_preparation": True, "elapsed_seconds": time.time() - started}
    if process.returncode == 0:
        result.update(model=str(target), model_sha256=sha256(target), bytes=target.stat().st_size)
    write_json(output / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_manifest", "source_hashes")}), flush=True)
    if process.returncode:
        raise RuntimeError("conversion failed; see preserved log")


def run(args):
    if args.output.exists():
        raise ValueError("choose a new run root")
    paths = [Path(__file__), CPP, PLAN, args.fixtures, BINARY,
             Path("experiments/structure_net/prepare_llama_cotenancy.py"),
             Path("experiments/structure_net/prepare_llama_summary_prompt_v2.py"),
             Path("experiments/structure_net/prepare_llama_summary_prompt_v3.py"),
             Path("docs/07 - Status Reports/2026-10-13_llama32-summary-quality-amendment.md"),
             Path("docs/07 - Status Reports/2026-10-13_llama32-summary-cardinality-amendment.md")]
    paths += [VENDOR / "build/bin" / name for name in ("libllama.so.0.6.0", "libggml.so.0.26.0", "libggml-base.so.0.26.0",
                                                        "libggml-cpu.so.0.26.0", "libggml-cuda.so.0.26.0")]
    config = {"hypothesis_id": HYPOTHESIS_ID, "schema": SCHEMA_VERSION,
              "model": str(args.model), "model_sha256": sha256(args.model), "stage": args.stage,
              "source_hashes": {str(p): sha256(p) for p in paths},
              "vendor_commit": subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip(),
              "gpu_before": gpu_state(), "command": sys.argv}
    config["fingerprint"] = digest(config)
    snapshot = args.output / "producer_source"
    snapshot.mkdir(parents=True)
    for i, path in enumerate(paths):
        shutil.copyfile(path, snapshot / f"{i:02d}_{path.name}")
    write_json(args.output / "producer.json", config)
    with (args.output / "runtime.log").open("w") as log:
        cmd = [str(BINARY.resolve()), str(args.model.resolve()), str(args.fixtures.resolve()),
               str((args.output / "raw.json").resolve()), "pilot" if args.stage == "pilot" else "full"]
        started = time.time()
        process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    result = {"schema": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
              "status": "completed" if process.returncode == 0 else "failed", "returncode": process.returncode,
              "elapsed_seconds": time.time() - started, "producer_fingerprint": config["fingerprint"],
              "gpu_after": gpu_state()}
    if (args.output / "raw.json").exists():
        result["raw_sha256"] = sha256(args.output / "raw.json")
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)
    if process.returncode:
        raise RuntimeError("native run failed; see preserved runtime.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("fixtures")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--stage", choices=("pilot", "development", "heldout"), required=True)
    convert = sub.add_parser("prepare-model")
    convert.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--fixtures", type=Path, required=True)
    execute.add_argument("--model", type=Path, required=True)
    execute.add_argument("--stage", choices=("pilot", "development", "heldout"), required=True)
    args = parser.parse_args()
    if args.command == "fixtures":
        prepare(args.output, args.stage)
    elif args.command == "prepare-model":
        prepare_model(args.output)
    else:
        # Deliberately before importing any CUDA library in this producer.
        os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0", LD_LIBRARY_PATH="",
                          OMP_NUM_THREADS="2")
        run(args)


if __name__ == "__main__":
    main()
