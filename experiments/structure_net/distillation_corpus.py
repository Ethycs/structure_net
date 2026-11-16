"""Metadata-only local corpus inventory and pointer-only response-file indexing.

Inventory reads only the named historical receipts, never corpus examples or
sealed evaluations. ``prepare`` indexes an explicitly supplied tokenized JSONL;
it does not admit a new scientific study or copy the underlying text payload.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = Path("data/experiments/ncm_paper_regime")
RECEIPTS = {
    "union": ("20261028_combined_prepared_pair_inventory_v1/result.json", "a10aacce5d41a80ecc9fa1119b3623d527a43d8e01e99d2de9467517356f35cd"),
    "public": ("20261026_public_training_catalog_v1/result.json", "4baebd5d9b5ad15387ef246bf73f3472d60d90a90da44ba0b607e965ad71c5b2"),
    "public_audit": ("20261026_public_training_catalog_audit_v1/result.json", "63045e80261a03a7bdf75aec961475c21fdc6d4debeb6c801ab6df3423396aa5"),
    "recovered": ("20261028_recovered_paraphrase_catalog_v1/result.json", "ef2ba5661a46498510b417f3aef56d36ab6530f51190837dce46e849efce5c5c"),
    "recovered_admission": ("20261028_recovered_paraphrase_training_prepare_v1/result.json", "1c979cf4769b8363d146687a809b10ae77a47b83b98eaed35067cdb189d988e7"),
    "anchors": ("20261027_expanded_teacher_catalog_prepare_v3/result.json", "8263ce24ded8d84740975e4e0d3e0c94643949cd44624be8a5c0dd796156ec15"),
    "census": ("20261027_available_pair_total_v3/result.json", "443b7011c80f42a250d4c6d729284f05f87be8a46a82676772d38495ce613b64"),
}


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(repository=ROOT):
    repository = Path(repository)
    records, evidence = {}, {}
    for name, (relative, expected) in RECEIPTS.items():
        path = repository/BASE/relative
        if path.stat().st_size > 2*1024**2: raise ValueError("Inventory only reads bounded metadata")
        actual = sha(path)
        if actual != expected: raise ValueError("Historical receipt changed: "+str(path))
        records[name] = json.loads(path.read_text())
        evidence[name] = dict(path=str(path), sha256=actual)
    union, census = records["union"], records["census"]
    if union["training_admitted"] or census["interpretation"]["training_admitted"]:
        raise ValueError("Unexpected change to historical inventory-only admission")
    if not records["public_audit"]["training_data_admitted"] or not records["recovered_admission"]["data_admitted"]:
        raise ValueError("Historical individual catalog admission missing")
    definitions = [
        ("public_xsum", "public", "20261026_public_training_catalog_v1/training.jsonl", 162353),
        ("recovered_conversation", "recovered", "20261028_recovered_paraphrase_catalog_v1/pairs.jsonl", 43894),
        ("reviewed_anchors", "anchors", "20261027_expanded_teacher_catalog_prepare_v3/cases.json", 2384),
        ("prepared_union_index", "union", "20261028_combined_prepared_pair_inventory_v1/pairs_index.jsonl", 208631),
    ]
    payloads = []
    for name, receipt, relative, count in definitions:
        path = repository/BASE/relative
        recorded = records[receipt]["artifacts"].get(str(BASE/relative))
        if not recorded: raise ValueError("Missing original payload binding")
        payloads.append(dict(name=name, path=str(path), historical_rows=count,
            actual_file_bytes=path.stat().st_size, recorded_sha256=recorded,
            payload_rehashed_now=False, new_study_admitted=False))
    export = census["candidate_export"]
    candidate = repository/export["path"]
    if candidate.stat().st_size != export["bytes"]: raise ValueError("Candidate payload byte count changed")
    payloads.append(dict(name="shared_archive_candidates", path=str(candidate), historical_rows=export["pairs"],
        actual_file_bytes=candidate.stat().st_size, recorded_sha256=export["sha256"],
        payload_rehashed_now=False, new_study_admitted=False))
    ordinary = [x for x in census["by_input_kind"] if x["kind"] in {"raw", "children"}]
    return dict(schema="nal.distillation-corpus-inventory.v1", status="metadata_inventory_only", evidence=evidence,
        payloads=payloads, prepared_union_pairs=union["distinct_content_pairs"],
        prepared_union_distinct_source_texts=union["distinct_source_texts"],
        prepared_union_near_duplicate_split_audit_completed=union["union_near_duplicate_split_audit_completed"],
        prepared_domain_counts=union["input_rows"], cross_domain_exact_source_overlap=union["cross_domain_exact_source_overlap"],
        archive_ordinary_pairs=census["ordinary_source_summary_pairs"],
        archive_length_fit_candidates=census["ordinary_application_length_fit_pairs"],
        archive_candidate_cl100k_input_tokens=sum(x["input_tokens_across_fit_pairs"] for x in ordinary),
        archive_candidate_cl100k_target_tokens=sum(x["target_tokens_across_fit_pairs"] for x in ordinary),
        xsum_source_cl100k_tokens=records["public"]["counts"]["source_cl100k_tokens"],
        xsum_target_old_native_lexical_tokens=records["public"]["counts"]["native_lexical_tokens"],
        recovered_conditioning=records["recovered"]["coverage"],
        qwen_token_counts_measured=False, qwen_416_fit_count=None,
        new_corpus_admitted=False, payload_examples_opened=False, sealed_evaluations_opened=False,
        warning="Counts overlap: prepared conversations/anchors are within the archive supply. Do not sum populations. Old codec tokens/features are not Qwen teacher targets.")


def prepare(records_path, output, *, max_length=416, max_records=10000):
    """Validate a bounded explicit response JSONL and emit offsets, not copied text.

    Exact duplicate token pairs in one split retain all origin pointers but only
    the first row is active. Cross-split group/prompt duplicates are rejected.
    This index is preparation evidence; a study must still admit its corpus.
    """
    from experiments.neural_architecture_lab import distillation_recovery as cli
    data = cli.module("distillation_data")
    records_path, output = Path(records_path).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    identity = data._identity(records_path); guard = data.SplitGuard()
    index, duplicate_of, counts = [], {}, Counter()
    with records_path.open("rb") as stream:
        while True:
            offset = stream.tell(); line = stream.readline()
            if not line: break
            if len(index) >= max_records: raise ValueError("Explicit small-record admission limit exceeded; no partial/truncated corpus")
            row = json.loads(line)
            if row.get("split") not in {"train", "development"}: raise ValueError("Test rows are not permitted in this training index")
            record = data.ResponseRecord(**row); guard.add(record)
            if len(record.input_ids) > max_length: raise ValueError("Oversize response rejected; no silent truncation")
            key = data.digest(record.input_ids); original = duplicate_of.get(key)
            entry = dict(example_id=record.example_id, group_id=record.group_id, split=record.split,
                offset_bytes=offset, length_bytes=len(line), record_sha256=hashlib.sha256(line).hexdigest(),
                token_record_sha256=data.record_fingerprint(record), duplicate_of=original,
                input_tokens=len(record.input_ids), response_tokens=len(record.response_ids),
                supervised_tokens=len(record.prediction_positions))
            index.append(entry); counts["input_records"] += 1
            if original is None:
                duplicate_of[key] = record.example_id; counts["active_records"] += 1
                counts[record.split] += 1
                for field in ("input_tokens", "response_tokens", "supervised_tokens"): counts[field] += entry[field]
            else: counts["exact_duplicate_records"] += 1
    if not index: raise ValueError("Empty response corpus")
    payload_sha = sha(records_path)
    if data._identity(records_path) != identity: raise ValueError("Source response file changed during preparation")
    output.mkdir(parents=True)
    with (output/"index.jsonl").open("x") as stream:
        for row in index: stream.write(json.dumps(row, sort_keys=True)+"\n")
    manifest = dict(schema="nal.response-corpus-index.v1", status="validated_not_study_admitted",
        source_path=str(records_path), source_sha256=payload_sha, source_identity=identity,
        index_sha256=sha(output/"index.jsonl"), tokenization=guard.tokenization, counts=dict(counts),
        max_length=max_length, truncation="reject", duplicate_policy="first exact-token pair active; all original pointers retained",
        near_duplicate_policy="not evaluated; study admission must declare additional checks", test_opened=False,
        new_study_admitted=False)
    data._json(output/"manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    sub.add_parser("inventory")
    item = sub.add_parser("prepare"); item.add_argument("--records", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True); item.add_argument("--max-length", type=int, default=416)
    item.add_argument("--max-records", type=int, default=10000)
    args = parser.parse_args()
    result = inventory() if args.phase == "inventory" else prepare(args.records, args.output,
        max_length=args.max_length, max_records=args.max_records)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
