"""Restore shared memory stores and make conversation-disjoint summary pairs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import sqlite3
import subprocess
import tarfile


LAKEFS_COMMIT = "82d974caec273e113dd864f9da352c003540e9225fbcce3766a81c103e005dab"
GIT_COMMIT = "fc25ea0d3cccc589346ddc13747218b8dc7a1221"


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def restore(bundles, output):
    output.mkdir(parents=True, exist_ok=False)
    catalog = json.loads((bundles / "catalog.json").read_text())
    receipts = []
    for item in catalog["archives"]:
        archive = bundles / item["archive"]
        if file_hash(archive) != item["archive_sha256"]:
            raise ValueError("archive hash mismatch: " + archive.name)
        manifest = json.loads(Path(str(archive) + ".json").read_text())
        expected = {m["path"]: m for m in manifest["members"]}
        seen = set()
        process = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
                for member in tar:
                    name = PurePosixPath(member.name)
                    if not member.isfile() or name.is_absolute() or ".." in name.parts:
                        raise ValueError("unexpected archive member")
                    record = expected.get(member.name)
                    if record is None or member.name in seen or member.size != record["bytes"]:
                        raise ValueError("archive member differs from manifest")
                    dest = output.joinpath(*name.parts)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    h = hashlib.sha256()
                    with tar.extractfile(member) as source, dest.open("xb") as target:
                        for block in iter(lambda: source.read(8 << 20), b""):
                            target.write(block)
                            h.update(block)
                    if h.hexdigest() != record["sha256"]:
                        raise ValueError("restored member hash mismatch")
                    seen.add(member.name)
            if process.wait() != 0 or seen != set(expected):
                raise ValueError("incomplete archive restore")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        receipts.append({"archive": archive.name, "sha256": item["archive_sha256"],
                         "verified_members": len(seen)})
        print(json.dumps(receipts[-1]), flush=True)
    write_json(output / "restore_receipt.json", {
        "status": "verified", "lakefs_commit": LAKEFS_COMMIT,
        "catalog_sha256": file_hash(bundles / "catalog.json"), "archives": receipts,
        "files": sum(r["verified_members"] for r in receipts)})


def hydrate(section, turns):
    parts = []
    for span in section["spans"]:
        turn = turns[span["turn_id"]]
        text = turn["text"]
        if (turn["source_id"] != span["source_id"] or turn["role"] != span["role"]
                or text_hash(text) != span["turn_text_sha256"]
                or not 0 <= span["start_char"] < span["end_char"] <= len(text)):
            raise ValueError("source span provenance mismatch")
        part = text[span["start_char"]:span["end_char"]]
        if text_hash(part) != span["span_text_sha256"]:
            raise ValueError("source span hash mismatch")
        parts.append(part)
    return "\n".join(parts)


def split_rows(rows, test_fraction=.1, seed=20261022):
    """Keep entire source sessions and exact duplicate inputs in one partition."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    first = {}
    # Generic short bypass text ("Thanks!") must not connect all conversations.
    # Bypass rows follow their session split and never enter model training.
    for row in rows:
        find(row["source_id"])
        if row["input_tokens"] >= 192:
            key = row["source_sha256"]
            if key in first:
                union(row["source_id"], first[key])
            else:
                first[key] = row["source_id"]
    groups = sorted({find(r["source_id"]) for r in rows},
                    key=lambda x: text_hash(f"{seed}:{x}"))
    if len(groups) < 2:
        raise ValueError("at least two source groups required")
    count = max(1, min(len(groups)-1, round(len(groups)*test_fraction)))
    test = set(groups[:count])
    result = []
    for row in rows:
        row = dict(row)
        row["group_id"] = find(row["source_id"])
        row["split"] = "test" if row["group_id"] in test else "train"
        result.append(row)
    return result


def prepare(bundles, restored, output):
    from transformers import AutoTokenizer
    import tiktoken
    from experiments.structure_net.ncm_llama import GPT2

    output.mkdir(parents=True, exist_ok=False)
    gpt = AutoTokenizer.from_pretrained(GPT2, local_files_only=True)
    gpt.model_max_length = 1_000_000
    app = tiktoken.get_encoding("cl100k_base")
    catalog = json.loads((bundles / "catalog.json").read_text())
    rows, seen, counts, stores = [], set(), Counter(), []
    for i, question in enumerate(catalog["questions"]):
        store = restored / question["store"]
        with sqlite3.connect((store / "memory.db").resolve().as_uri()+"?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            turns = {r["turn_id"]: dict(r) for r in db.execute(
                "SELECT turn_id, role, text, source_id FROM turns ORDER BY ordinal")}
        with sqlite3.connect((store / "native-spine-live-v1.sqlite").resolve().as_uri()+"?mode=ro", uri=True) as db:
            sections = [(kind, json.loads(payload)) for kind, payload in db.execute(
                "SELECT kind,payload FROM sections ORDER BY kind,sid")]
        stores.append({"store": question["store"], "question_id": question["question_id"],
                       "turns": len(turns), "sections": len(sections)})
        for kind, section in sections:
            counts["sections_"+kind] += 1
            if kind != "atomic":
                continue
            source = hydrate(section, turns)
            target = section["summary"]
            if not source.strip() or not target.strip():
                raise ValueError("empty source or saved summary")
            sh, th = text_hash(source), text_hash(target)
            key = (section["source_id"], sh, th)
            if key in seen:
                counts["duplicate_pairs"] += 1
                continue
            seen.add(key)
            source_ids = gpt.encode(source, add_special_tokens=False)
            target_ids = gpt.encode(target, add_special_tokens=False)
            input_tokens = len(app.encode(source, disallowed_special=()))
            target_tokens = len(app.encode(target, disallowed_special=()))
            reasons = []
            if input_tokens < 192:
                reasons.append("application_bypass")
            if len(source_ids) > 512:
                reasons.append("source_over_512_gpt2")
            if len(target_ids) > 47:
                reasons.append("target_over_47_gpt2")
            if target_tokens > 128:
                reasons.append("target_over_128_application_tokens")
            for reason in reasons:
                counts[reason] += 1
            rows.append({"id": text_hash("|".join(key)), "source_id": section["source_id"],
                         "source": source, "target": target, "source_sha256": sh,
                         "target_sha256": th, "input_tokens": input_tokens,
                         "target_tokens": target_tokens, "source_gpt2_tokens": len(source_ids),
                         "target_gpt2_tokens": len(target_ids), "native_eligible": not reasons,
                         "exclusions": reasons, "store": question["store"],
                         "section_id": section["section_id"], "spans": section["spans"],
                         "summarizer_identity": section["summarizer_identity"]})
        if (i+1) % 25 == 0:
            print(json.dumps({"stores": i+1, "unique_pairs": len(rows)}), flush=True)
    rows = split_rows(rows)
    files, splits = {}, {}
    for split in ("train", "test"):
        chosen = [r for r in rows if r["split"] == split]
        native = [r for r in chosen if r["native_eligible"]]
        for name, records in [(split, chosen), ("native_"+split, native)]:
            p = output / (name+".jsonl")
            with p.open("w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n")
            files[p.name] = {"sha256": file_hash(p), "rows": len(records)}
        splits[split] = {"rows": len(chosen), "native_rows": len(native),
                         "source_sessions": len({r["source_id"] for r in chosen}),
                         "groups": len({r["group_id"] for r in chosen})}
    eligible = {s: {r["source_sha256"] for r in rows if r["split"] == s and r["native_eligible"]}
                for s in ("train", "test")}
    sessions = {s: {r["source_id"] for r in rows if r["split"] == s} for s in ("train", "test")}
    if eligible["train"] & eligible["test"] or sessions["train"] & sessions["test"]:
        raise ValueError("train/test source overlap")
    result = {"schema": "nal.ncm-memory-data.v1", "status": "prepared",
              "lakefs_commit": LAKEFS_COMMIT, "git_commit": GIT_COMMIT,
              "catalog_sha256": file_hash(bundles / "catalog.json"),
              "restore_receipt_sha256": file_hash(restored / "restore_receipt.json"),
              "producer_sha256": file_hash(__file__), "seed": 20261022,
              "test_group_fraction": .1, "target_origin": "unchanged saved atomic summaries",
              "tokenizer": {"application": "cl100k_base", "native": str(GPT2)},
              "stores": stores, "counts": dict(counts), "splits": splits, "files": files,
              "eligible_input_overlap": 0, "source_session_overlap": 0,
              "scope": "Atomic source-to-summary pairs; hierarchy and parent sections retained in restored stores.",
              "benchmark_note": "LongMemEval is now training data; this partition is not the untouched benchmark."}
    write_json(output / "manifest.json", result)
    print(json.dumps({"status": "prepared", "splits": splits, "counts": dict(counts)}), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["restore", "prepare"])
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--restored", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "restore":
        restore(args.bundles, args.output)
    else:
        prepare(args.bundles, args.restored, args.output)


if __name__ == "__main__":
    main()
