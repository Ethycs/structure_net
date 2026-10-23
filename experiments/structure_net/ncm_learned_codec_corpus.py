"""Unpaired text for a learned codec; no atomic-to-whole-summary relabeling."""
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_isolated_quality_data as quality
from experiments.structure_net.ncm_llama import GPT2
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, PREP, DATA as ORIGINAL_DATA, freeze, verify

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-learned-codec-corpus-plan.md")
DEST = ROOT / "20261023_learned_codec_corpus_v1"


@dataclass(frozen=True)
class Config:
    block: int = 48
    summary_canvas: int = 192
    eos: int = 50256
    pad: int = 50257


def pack(ids, width, pad, capacity=None):
    """Preserve an already EOS-terminated stream, with padding only at its tail."""
    if not ids or width < 1 or (capacity is not None and (capacity % width or len(ids) > capacity)):
        raise ValueError("complete nonempty token stream must fit declared blocks")
    total = capacity if capacity is not None else ((len(ids) + width - 1) // width) * width
    return torch.tensor(ids + [pad] * (total - len(ids)), dtype=torch.long).reshape(-1, width)


def prepare():
    from dataclasses import asdict
    import transformers
    from transformers import AutoTokenizer
    if DEST.exists():
        raise ValueError("codec corpus artifacts are append-only")
    config = Config()
    quality.terminal(quality.DEST)
    old_path = quality.old_data.DEST / "cases.json"
    old = quality.run.q.read(old_path)
    summaries = quality.run.q.read(quality.DEST / "cases.json")
    pool = quality.run.q.read(quality.POOL_META)
    development = quality.run.q.read(PREP / "development_sources.json")
    heldout = [*development, *old["language_diagnostic"], *pool["diagnostic"], *pool["sealed_final"]]
    excluded = {k: {r[k] for r in [*heldout, *summaries]} for k in ("source_id", "group_id", "source_sha256")}
    allowed = {r["group_id"] for r in old["language_train"]} - excluded["group_id"]
    manifest = quality.run.q.read(ORIGINAL_DATA / "manifest.json")
    shared = ORIGINAL_DATA / "native_train.jsonl"
    if (file_hash(shared) != manifest["files"]["native_train.jsonl"]["sha256"]
            or manifest["source_session_overlap"] or manifest["eligible_input_overlap"]):
        raise ValueError("original train partition proof differs")
    tokenizer = AutoTokenizer.from_pretrained(GPT2, local_files_only=True)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.model_max_length = 1_000_000
    if tokenizer.eos_token_id != config.eos or tokenizer.pad_token_id != config.pad:
        raise ValueError("released tokenizer identities differ")
    target_hashes = {text_hash(r["target_text"]) for r in summaries}
    texts = {}
    for line in shared.open():
        row = json.loads(line)
        if row["split"] != "train":
            raise ValueError("heldout row in original train store")
        if row["group_id"] not in allowed:
            continue
        if any(row[k] in excluded[k] for k in excluded):
            raise ValueError("fit text overlaps codec/other heldout source identities")
        for field in ("source", "target"):
            value = row[field]; key = text_hash(value)
            if key in target_hashes:
                continue
            entry = texts.setdefault(key, {"text": value, "origins": []})
            entry["origins"].append({"pair_id": row["id"], "source_id": row["source_id"],
                "group_id": row["group_id"], "source_sha256": row["source_sha256"], "field": field})
    arrays, records, rejected, cursor = [], [], [], 0
    for key, entry in sorted(texts.items()):
        ids = tokenizer.encode(entry["text"], add_special_tokens=False)
        if not ids or set(ids) & {config.eos, config.pad}:
            rejected.append({"sha256": key, "reason": "empty_or_embedded_control_id"}); continue
        if tokenizer.decode(ids, skip_special_tokens=False) != entry["text"]:
            raise ValueError("unpaired text does not round trip")
        ids += [config.eos]
        blocks = pack(ids, config.block, config.pad)
        records.append({"text_sha256": key, "origins": entry["origins"], "tokens_with_EOS": len(ids),
            "window_start": cursor, "windows": len(blocks)})
        arrays.append(blocks)
        cursor += len(blocks)
    complete = []
    for row in summaries:
        ids = tokenizer.encode(row["target_text"], add_special_tokens=False)
        if set(ids) & {config.eos, config.pad} or tokenizer.decode(ids) != row["target_text"]:
            raise ValueError("complete summary tokenizer/control identity differs")
        ids += [config.eos]
        complete.append({"id": row["id"], "source_id": row["source_id"], "group_id": row["group_id"],
            "source_sha256": row["source_sha256"], "text": row["target_text"], "target_ids": ids,
            "packed_ids": pack(ids, config.block, config.pad, config.summary_canvas).tolist()})
    if len(complete) != 252 or not arrays:
        raise ValueError("complete source-disjoint fit and 252-summary diagnostic required")
    files = [Path(__file__), PLAN, old_path, quality.DEST / "cases.json", quality.DEST / "result.json",
        quality.DEST / "producer.json", quality.POOL_META, PREP / "development_sources.json",
        ORIGINAL_DATA / "manifest.json", shared]
    files += [p for p in GPT2.iterdir() if p.is_file() and (p.suffix == ".json" or p.name == "merges.txt")]
    producer = freeze(DEST, "unpaired_codec_corpus_only_not_summary_training", files,
        {**asdict(config), "allowed_groups": len(allowed), "atomic_targets_are_unpaired_text_only": True,
         "parent_loaded": False, "task_accuracy": None, "codec_weights_trained": False,
         "torch": torch.__version__, "transformers": transformers.__version__})
    producer["hypothesis_id"] = "ncm-broad-learned-codec-reconstruction-v1"
    for path in (Path(__file__), PLAN):
        producer["source_hashes"][str(path)] = file_hash(path)
        (DEST / "producer_source" / ("learned_codec_" + path.name)).write_bytes(path.read_bytes())
    producer["fingerprint"] = quality.run.q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    write_json(DEST / "producer.json", producer)
    tensors = torch.cat(arrays)
    torch.save(tensors, DEST / "fit_windows.pt")
    if not torch.equal(tensors, torch.load(DEST / "fit_windows.pt", weights_only=True)):
        raise ValueError("codec corpus tensor reload differs")
    write_json(DEST / "fit_origins.json", records)
    write_json(DEST / "summary_diagnostic.json", complete)
    write_json(DEST / "rejected_unpaired_texts.json", rejected)
    verify(producer)
    result = {"status": "prepared_unpaired_codec_text_not_quality_evidence", "fit_texts": len(records),
        "fit_windows": len(tensors), "fit_valid_tokens": sum(r["tokens_with_EOS"] for r in records),
        "allowed_groups": len(allowed), "summary_diagnostic_cases": len(complete),
        "summary_target_max": max(len(r["target_ids"]) for r in complete),
        "source_disjoint": True, "summary_truncated": False, "codec_weights_trained": False,
        "task_accuracy": None, "final_test_opened": False, "producer_fingerprint": producer["fingerprint"],
        "artifacts": {p.name: file_hash(p) for p in DEST.iterdir() if p.is_file()}}
    write_json(DEST / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    prepare()
