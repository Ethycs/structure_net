"""Verify grading passages against the publicly released LongMemEval-S file.

The only network operation reads public file metadata from Hugging Face. It
does not transmit source passages, generated summaries, or local metadata.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

PUBLIC_API = "https://huggingface.co/api/datasets/xiaowu0162/longmemeval-cleaned/tree/main"
PUBLIC_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
PUBLIC_FILE = Path("data/experiments/ncm_memory/20261022_shared_v1/source/data/longmemeval_s_cleaned.json")


def verify_cases(samples, cases):
    turns = defaultdict(dict)
    for sample in samples:
        for sid, session in zip(sample["haystack_session_ids"], sample["haystack_sessions"]):
            for turn in session:
                turns[str(sid)][(turn["role"], text_hash(turn["content"]))] = turn["content"]
    matched = {}
    for case in cases:
        parts = []
        for span in case["spans"]:
            try:
                turn = turns[span["source_id"]][(span["role"], span["turn_text_sha256"])]
            except KeyError as error:
                raise ValueError("source turn absent from public dataset") from error
            if not 0 <= span["start_char"] < span["end_char"] <= len(turn):
                raise ValueError("invalid public source span")
            part = turn[span["start_char"]:span["end_char"]]
            if text_hash(part) != span["span_text_sha256"]:
                raise ValueError("public span hash mismatch")
            parts.append(part)
        source = "\n".join(parts)
        if source != case["source"]:
            raise ValueError("passage differs from public source spans")
        if case["id"] in matched:
            raise ValueError("duplicate evaluation case")
        matched[case["id"]] = text_hash(source)
    return matched


def main(args):
    destination = args.root / "public_data_readback.json"
    if destination.exists():
        raise ValueError("fresh provenance receipt required")
    digest = file_hash(args.public_dataset)
    if digest != PUBLIC_SHA256:
        raise ValueError("local public dataset changed")
    import requests
    response = requests.get(PUBLIC_API, timeout=30)
    response.raise_for_status()
    metadata = next(x for x in response.json() if x["path"] == "longmemeval_s_cleaned.json")
    if metadata["lfs"]["oid"] != digest:
        raise ValueError("published file SHA256 differs")
    selection = args.root / "selection.json"
    cases = json.loads(selection.read_text())["test"]
    matched = verify_cases(json.loads(args.public_dataset.read_text()), cases)
    if len(matched) != 64:
        raise ValueError("fixed 64-case cohort required")
    write_json(destination, {
        "status": "verified_exact_public_benchmark_sources", "public_api": PUBLIC_API,
        "public_file_metadata": metadata, "local_file": str(args.public_dataset),
        "local_sha256": digest, "matched_exact_public_source_spans": len(matched),
        "requested_cases": 64, "matched_ids": list(matched), "source_hashes": matched,
        "unmatched_ids": [], "selection_sha256": file_hash(selection),
        "producer_sha256": file_hash(__file__),
        "payload_scope": "Only exact public source spans and generated summaries; "
        "no credentials, training targets, internal paths, or private corpus."})
    print(json.dumps({"status": "verified_exact_public_benchmark_sources", "cases": len(matched)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--public-dataset", type=Path, default=PUBLIC_FILE)
    main(parser.parse_args())
