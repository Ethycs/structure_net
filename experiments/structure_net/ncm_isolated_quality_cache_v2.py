"""Correct embedding-shaped validation for immutable native source KV."""
import argparse
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_isolated_quality_data as data
from experiments.structure_net.ncm_native_query_distillation import cpu_prefix, pack_prefix
from experiments.structure_net.ncm_native_readout import frozen_hash
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify

AMENDMENT = Path("docs/07 - Status Reports/2026-10-23_ncm-quality-cache-validation-amendment.md")
PILOT = ROOT / "20261023_isolated_quality_cache_pilot_v2"
CACHE = ROOT / "20261023_isolated_quality_cache_v2"


@torch.no_grad()
def roundtrip_logits(backbone, memory, restored, targets, eos):
    device = memory.source_mask.device
    previous = targets[:-1][None].to(device).masked_fill(targets[:-1][None].to(device).eq(-100), eos)
    query = backbone.predecessor_embeddings(memory, previous)
    actual = backbone.query_logits(memory, query)
    reloaded = pack_prefix([restored["prefix"]], device)
    restored_previous = restored["targets"][:-1][None].to(device)
    restored_previous = restored_previous.masked_fill(restored_previous.eq(-100), eos)
    restored_query = backbone.predecessor_embeddings(reloaded, restored_previous)
    reload = backbone.query_logits(reloaded, restored_query)
    if not torch.equal(actual, reload) or not torch.isfinite(actual).all():
        raise ValueError("source cache save/reload logits differ")
    return {"query_shape": list(query.shape), "logits_shape": list(actual.shape), "exact": True}


def producing(root, stage, inputs, config):
    p = data.producing(root, stage, inputs + [Path(__file__), AMENDMENT],
        {**config, "cache_validation_version": 2})
    for path in (Path(__file__), AMENDMENT):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("cache_v2_" + path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = data.run.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


@torch.no_grad()
def main(pilot=False):
    data.run.require_gpu(); data.terminal(data.DEST)
    rows = data.run.q.read(data.DEST / "cases.json")
    if file_hash(data.DEST / "cases.json") != data.run.q.read(data.DEST / "result.json")["cases_sha256"]:
        raise ValueError("training metadata changed")
    if pilot:
        ids = data.run.q.read(data.DEST / "selection.json")["pilot_ids"]
        selected = [next(r for r in rows if r["id"] == identifier) for identifier in ids]
    else:
        data.terminal(PILOT)
        if data.run.q.read(PILOT / "result.json")["status"] != "systems_cache_identity_passed_not_quality_evidence":
            raise ValueError("longest-frame corrected cache pilot required")
        selected = rows
    root = PILOT if pilot else CACHE
    inputs = [data.DEST / n for n in ("producer.json", "result.json", "cases.json", "selection.json", "worker_terminal_receipt.json")]
    inputs += [data.run.MODEL_PATH / n for n in ("config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")]
    if not pilot:
        inputs += [PILOT / n for n in ("producer.json", "result.json", "manifest.json", "worker_terminal_receipt.json")]
    p = producing(root, "longest_source_cache_systems_pilot" if pilot else "mixed_length_source_KV_cache",
        inputs, {"pilot": pilot, "selected": [r["id"] for r in selected], "batch": 1, "physical_gpu": 0})
    (root / "cases").mkdir()
    net, tokenizer = data.run.model_and_tokenizer()
    before = frozen_hash(net.backbone)
    manifest = {"schema": "nal.isolated-quality-source-cache.v2", "cases": {}, "ids": [r["id"] for r in selected],
        "producer_fingerprint": p["fingerprint"], "native_eos": tokenizer.eos_token_id}
    probes = []
    for i, row in enumerate(selected):
        if row["old_cache_entry"] is not None:
            entry = row["old_cache_entry"]
            if file_hash(entry["path"]) != entry["sha256"]:
                raise ValueError("old source cache changed")
            manifest["cases"][row["id"]] = {**entry, "reused_old_cache": True}
        else:
            memory = net.backbone.source_cache(**data.old_data.source_inputs([row], tokenizer.eos_token_id, "cuda"))
            targets = data.old_data.target_tensor([row], "cpu")[0]
            example = {"id": row["id"], "prefix": cpu_prefix(memory, 0), "targets": targets,
                "kind": "summary", "window_eos_injected": False, "source_frame": row["source_frame"],
                "producer_fingerprint": p["fingerprint"]}
            path = root / "cases" / (row["id"] + ".pt")
            torch.save(example, path)
            restored = torch.load(path, map_location="cpu", weights_only=True)
            probe = roundtrip_logits(net.backbone, memory, restored, targets, tokenizer.eos_token_id)
            probes.append({"id": row["id"], **probe})
            manifest["cases"][row["id"]] = {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size,
                "source_frame": row["source_frame"], "source_sha256": row["source_sha256"], "reused_old_cache": False}
            del memory
        write_json(root / "manifest.json", manifest)
        if pilot or (i + 1) % 16 == 0 or i + 1 == len(selected):
            print(json.dumps({"cached": i + 1, "total": len(selected)}), flush=True)
    if frozen_hash(net.backbone) != before:
        raise ValueError("original source backbone mutated")
    verify(p)
    write_json(root / "reload_probes.json", probes)
    result = {"status": "systems_cache_identity_passed_not_quality_evidence" if pilot else "mixed_length_source_caches_prepared",
        "cases": len(selected), "reused_old": sum(e["reused_old_cache"] for e in manifest["cases"].values()),
        "source_only": True, "cache_reload_logits_exact": True, "original_backbone_unchanged": True,
        "max_source_frame": max(r["source_frame"] for r in selected), "manifest_sha256": file_hash(root / "manifest.json"),
        "task_accuracy": None, "NCM_success_established": False, "full_training_admitted": False,
        "producer_fingerprint": p["fingerprint"]}
    write_json(root / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--pilot", action="store_true")
    main(parser.parse_args().pilot)
