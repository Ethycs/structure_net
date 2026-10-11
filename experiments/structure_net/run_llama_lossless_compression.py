"""CPU-only, exhaustive storage experiment on frozen lossless components."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

import numpy as np
import torch
import zstandard

from structure_net.components.layers import lossless_tensor_archive as codec

HYPOTHESIS_ID = "llama32-3b-lossless-component-compression-v1"
SCHEMA = "nal.llama-lossless-compression.v1"
ROOT = Path("data/experiments/llama_tensorized_cotenancy")
ORIGINAL = ROOT / "20261010_control_v1/model.pt"
FACTORS = ROOT / "20261010_campaign_v2/svd2_r215_conversion/checkpoint/model.pt"
PARENT = ROOT / "20261010_lossless_v1/campaign_results.json"
PLAN = Path("docs/07 - Status Reports/2026-10-10_llama32-lossless-compression-plan.md")
EXPECTED = {str(ORIGINAL): "f01cc0653ec6c9bd7026b8676864612b91b6c79d2dc4df42599cbe4cf99eb9a9",
            str(FACTORS): "71e621f4021a9ee89a282576a65512857b61f509ec3605eed54d365919c92d6d",
            str(PARENT): "4be53fe363bdd889919260583776e89a56f614e606a6a0581efdf2741800331a"}
CONFIG = {"layouts": list(codec.LAYOUTS), "level": 3, "chunk_bytes": 1024 * 1024,
          "screen_elements": 65536, "screen_layers": [0, 14, 27], "device": "cpu",
          "threads": 1, "selection": "smallest_total_screen_payload_then_layout_order"}


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def inputs():
    for path, expected in EXPECTED.items():
        if codec.digest_file(path) != expected:
            raise ValueError(f"parent input hash mismatch: {path}")
    parent = read(PARENT)
    cell = next(c for c in parent["cells"] if c["name"] == "weights")
    if codec.digest_file(cell["result"]) != cell["sha256"]:
        raise ValueError("parent weight audit hash mismatch")
    weights = read(cell["result"])["weights"]
    if len(weights) != 197 or len({r["name"] for r in weights}) != 197:
        raise ValueError("incomplete residual set")
    rows = [{"group": "original", "id": "original", "path": str(ORIGINAL), "sha256": EXPECTED[str(ORIGINAL)]},
            {"group": "factors", "id": "factors", "path": str(FACTORS), "sha256": EXPECTED[str(FACTORS)]}]
    for i, row in enumerate(weights):
        if codec.digest_file(row["residual_path"]) != row["residual_sha256"]:
            raise ValueError("parent residual hash mismatch")
        rows.append({"group": "residuals", "id": f"residual_{i:03d}", "name": row["name"],
                     "path": row["residual_path"], "sha256": row["residual_sha256"]})
    return rows


def selected(name):
    match = re.search(r"layers\.(\d+)\.", name)
    return "embed_tokens" in name or (match is not None and int(match[1]) in CONFIG["screen_layers"])


def screen_slices(value, row):
    for name, tensor in codec.flatten(value)[1]:
        all_values = "/vectors/" in name or name.endswith(("/indices", "/repairs"))
        if not all_values and not selected(row.get("name", name)):
            continue
        flat = tensor.reshape(-1)
        count = min(CONFIG["screen_elements"], flat.numel())
        starts = [0] if all_values else sorted({0, (flat.numel() - count) // 2, flat.numel() - count})
        for offset in starts:
            sample = flat if all_values else flat[offset:offset + count]
            yield name, sample


def screen(rows, output):
    totals = {g: {layout: 0 for layout in codec.LAYOUTS} for g in ("original", "factors", "residuals")}
    samples, occupancy = [], {}
    started = time.perf_counter()
    for row in rows:
        value = torch.load(row["path"], map_location="cpu", weights_only=False, mmap=True)
        for name, tensor in screen_slices(value, row):
            raw = codec.tensor_bytes(tensor).tobytes()
            width = tensor.element_size()
            result = {"input": row["id"], "name": name, "raw_bytes": len(raw), "sizes": {}}
            for layout in codec.LAYOUTS:
                encoded, compressed = codec.encode_chunk(raw, width, layout, CONFIG["level"])
                if codec.decode_chunk(encoded, width, tensor.numel(), layout, compressed) != raw:
                    raise ValueError("screen roundtrip failure")
                totals[row["group"]][layout] += len(encoded)
                result["sizes"][layout] = len(encoded)
            samples.append(result)
            key = row["group"] + "/" + str(tensor.dtype)
            entry = occupancy.setdefault(key, {"count": 0, "one_counts_little_endian": [0] * (8 * width)})
            ones = np.unpackbits(np.frombuffer(raw, np.uint8).reshape(-1, width), axis=1, bitorder="little").sum(axis=0)
            entry["count"] += tensor.numel()
            entry["one_counts_little_endian"] = [a + int(b) for a, b in zip(entry["one_counts_little_endian"], ones)]
        del value
    winners = {group: min(codec.LAYOUTS, key=lambda k: sizes[k]) for group, sizes in totals.items()}
    result = {"schema": SCHEMA, "stage": "exploratory_screen", "totals": totals, "selected": winners,
              "samples": samples, "bit_occupancy": occupancy, "seconds": time.perf_counter() - started,
              "all_screen_roundtrips_equal": True}
    codec.write_json(output / "screen.json", result)
    print(json.dumps({"screen_selected": winners, "totals": totals}), flush=True)
    return result


def family(name):
    if name.endswith("/indices") or name.endswith("/repairs"):
        return "rounding_repairs"
    if "embed_tokens" in name:
        return "embedding"
    if "self_attn" in name:
        return "attention"
    if "mlp" in name:
        return "mlp"
    if "/vectors/" in name or "norm" in name:
        return "norms"
    return "other"


def summarize(records):
    if len(records) != 199 or {r["id"] for r in records} != {"original", "factors"} | {f"residual_{i:03d}" for i in range(197)}:
        raise ValueError("full experiment requires all 199 archives")
    groups = {}
    for group in ("original", "factors", "residuals"):
        rows = [r for r in records if r["group"] == group]
        groups[group] = {key: sum(r[key] for r in rows) for key in
                         ("raw_bytes", "archive_bytes", "tensor_count", "encode_seconds", "decode_verify_seconds")}
        groups[group]["fraction_of_raw"] = groups[group]["archive_bytes"] / groups[group]["raw_bytes"]
    dense = groups["original"]
    combined = groups["factors"]["archive_bytes"] + groups["residuals"]["archive_bytes"]
    exact = all(r["bitwise_equal"] for r in records)
    classification = ("lossless_roundtrip_failed" if not exact else "lossless_storage_reduction"
                      if min(dense["archive_bytes"], combined) < dense["raw_bytes"] else "lossless_no_storage_reduction")
    return {"groups": groups, "compressed_factors_plus_residuals_bytes": combined,
            "combined_vs_compressed_original": combined / dense["archive_bytes"],
            "all_tensors_bitwise": exact, "classification": classification}


def run(output, shakedown=False):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(codec.__file__), PLAN]
    source_hashes = {str(p): codec.digest_file(p) for p in sources}
    frozen = output / "producer_source"
    frozen.mkdir()
    for i, path in enumerate(sources):
        shutil.copy2(path, frozen / f"{i:02d}_{path.name}")
    if shutil.disk_usage(output).free < 30 * 1024 ** 3:
        raise RuntimeError("less than 30 GiB free disk")
    rows = inputs()
    producer = {"schema": SCHEMA, "hypothesis_id": HYPOTHESIS_ID, "profile": "EXPLORATORY",
                "configuration": CONFIG, "inputs": rows, "sources": source_hashes,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "shakedown": shakedown, "environment": {"python": platform.python_version(),
                "numpy": np.__version__, "torch": torch.__version__, "zstandard": zstandard.__version__,
                "machine": platform.machine(), "processor": platform.processor()}}
    producer["fingerprint"] = fingerprint(producer)
    codec.write_json(output / "producer.json", producer)
    if shakedown:
        source = torch.load(ORIGINAL, map_location="cpu", weights_only=False, mmap=True)
        name = "model.layers.0.self_attn.q_proj.weight"
        value = {"config": source["config"], "matrix": source["matrices"][name]["cores"][0][:33].contiguous(),
                 "norm": next(iter(source["vectors"].values()))}
        results = [codec.save_archive(value, output / layout, layout) for layout in codec.LAYOUTS]
        for row in results:
            restored = codec.load_archive(row["directory"])
            codec.verify_archive(row["directory"], restored)
        codec.write_json(output / "result.json", {"status": "systems_lifecycle_only_not_quality_evidence", "results": results})
        return
    screened = screen(rows, output)
    records = []
    started = time.perf_counter()
    for row in rows:
        value = torch.load(row["path"], map_location="cpu", weights_only=False, mmap=True)
        result = codec.save_archive(value, output / row["id"], screened["selected"][row["group"]],
                                    CONFIG["chunk_bytes"], CONFIG["level"])
        result.update({k: v for k, v in row.items() if k != "path"})
        manifest = read(Path(result["directory"]) / "manifest.json")
        breakdown = {}
        for entry in manifest["tensors"]:
            key = family(row.get("name", "") + "/" + entry["name"])
            category = breakdown.setdefault(key, {"raw_bytes": 0, "payload_bytes": 0})
            category["raw_bytes"] += entry["numel"] * entry["width"]
            category["payload_bytes"] += sum(c["bytes"] for c in entry["chunks"])
        result["families"] = breakdown
        records.append(result)
        codec.write_json(output / row["id"] / "result.json", result)
        print(json.dumps({"completed": len(records), "requested": len(rows), "id": row["id"],
                          "raw_bytes": result["raw_bytes"], "archive_bytes": result["archive_bytes"],
                          "bitwise_equal": result["bitwise_equal"]}), flush=True)
        del value
        gc.collect()
    if any(codec.digest_file(p) != h for p, h in source_hashes.items()):
        raise ValueError("producer changed during run")
    aggregate = {"schema": SCHEMA, "hypothesis_id": HYPOTHESIS_ID, "profile": "EXPLORATORY", "status": "completed",
                 "producer_fingerprint": producer["fingerprint"], "producer_sha256": codec.digest_file(output / "producer.json"),
                 "screen_sha256": codec.digest_file(output / "screen.json"), "selected_layouts": screened["selected"],
                 "requested": len(rows), "completed": len(records), "failed": 0, "reused": 0,
                 "records": records, "seconds": time.perf_counter() - started, **summarize(records),
                 "application_accuracy_tested": False, "inference_speed_tested": False,
                 "limitations": ["One frozen Llama checkpoint; exploratory codec selection.",
                    "Archived bytes include payload and decoder metadata, not experiment logs or source copies.",
                    "CPU encode/decode costs are descriptive and not isolated benchmarks.",
                    "Decoding restores dense storage and compute; compressed GPU execution is untested.",
                    "Exact weights preserve original accuracy; parent validation failure remains unchanged."]}
    codec.write_json(output / "campaign_results.json", aggregate)
    print(json.dumps({"classification": aggregate["classification"], "groups": aggregate["groups"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument("--decode", type=Path, help="decode this archive to --output checkpoint instead of running study")
    args = parser.parse_args()
    if args.decode:
        if args.output.exists():
            raise FileExistsError(args.output)
        torch.save(codec.load_archive(args.decode), args.output)
        return
    try:
        run(args.output, args.shakedown)
    except Exception as error:
        if args.output.exists() and not (args.output / "campaign_results.json").exists() and not (args.output / "failure.json").exists():
            codec.write_json(args.output / "failure.json", {"status": "failed", "error": str(error), "type": type(error).__name__})
        raise


if __name__ == "__main__":
    main()
