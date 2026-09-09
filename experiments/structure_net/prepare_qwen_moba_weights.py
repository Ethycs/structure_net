#!/usr/bin/env python3
"""Fetch only the pinned public Qwen3-8B shard needed by the six-layer study."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
HASHES = {
    "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "model.safetensors.index.json": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "model-00001-of-00005.safetensors": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        path = output / name
        if not path.exists():
            url = f"https://huggingface.co/Qwen/Qwen3-8B/resolve/{REVISION}/{name}"
            temporary = path.with_suffix(path.suffix + ".download")
            with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as destination:
                while chunk := source.read(8 * 1024 * 1024):
                    destination.write(chunk)
            if sha256(temporary) != expected:
                raise RuntimeError(f"download hash mismatch: {name}")
            temporary.replace(path)
        if sha256(path) != expected:
            raise RuntimeError(f"existing file hash mismatch: {name}")
        print(f"verified {name}", flush=True)
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    prefixes = tuple(f"model.layers.{i}." for i in range(6))
    keys = sorted(key for key in index if key == "model.embed_tokens.weight" or key.startswith(prefixes))
    plan = {"model": "Qwen/Qwen3-8B", "revision": REVISION, "layers": 6,
            "keys": keys, "shards": sorted({index[key] for key in keys})}
    if len(keys) != 67 or plan["shards"] != ["model-00001-of-00005.safetensors"]:
        raise RuntimeError("pinned prefix manifest mismatch")
    plan_path = output / "prefix_download_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("existing prefix plan changed")
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    manifest = {**HASHES, "prefix_download_plan.json": sha256(plan_path)}
    manifest_path = output / "download_hashes.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("existing download manifest changed")
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/models/Qwen3-8B-prefix6"))
    args = parser.parse_args()
    prepare(args.output)
