"""Public-span-only wider source pools; no labels, model execution or test reads."""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, PREP, DATA, freeze, verify
from experiments.structure_net.ncm_frozen_body_public_data import PUBLIC_FILE, PUBLIC_SHA256
from experiments.structure_net.ncm_expanded_pairs import public_index, matches_public

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-wide-source-pool-plan.md")
AUDIT = ROOT / "20261023_application_source_availability_v1"
POOL = ROOT / "20261023_application_source_pool_v1"
BINS = ((512, 767), (768, 1023), (1024, 1535), (1536, 2048))
IDENTITY = ("id", "source_id", "group_id", "source_sha256")
FORBIDDEN_FILES = [PREP / "train_sources.json", PREP / "development_sources.json",
    ROOT / "20261022_expanded_data_v2/train_sources.json", ROOT / "20261022_expanded_data_v2/pilot_sources.json",
    Path("data/experiments/ncm_recipe_update/20261022_quality_v2/selected.json")]


def source_bin(tokens):
    return next((i for i, (low, high) in enumerate(BINS) if low <= tokens <= high), None)


def clean_source(row):
    return {key: row[key] for key in (*IDENTITY, "source", "spans")}


def forbidden():
    result = {key: set() for key in IDENTITY}
    for path in FORBIDDEN_FILES:
        value = q.read(path)
        if isinstance(value, dict):
            value = [r for split in ("train", "test") for r in value.get(split, [])]
        for row in value:
            for key in IDENTITY:
                result[key].add(row[key])
    return result


def producing(root, stage, inputs, config):
    files = [Path(__file__), PLAN, Path("experiments/structure_net/ncm_expanded_pairs.py")]
    p = freeze(root, stage, inputs + files, {**config, "source_bins_cl100k": BINS,
        "model_execution": False, "old_targets_exported": False, "original_test_files_read": False,
        "selection_namespace": "ncm-application-range-source-v1:", "original_partition": "train"})
    for path in files:
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("wide_sources_" + path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def audit():
    manifest = q.read(DATA / "manifest.json")
    source = DATA / "train.jsonl"
    if file_hash(source) != manifest["files"][source.name]["sha256"] or file_hash(PUBLIC_FILE) != PUBLIC_SHA256:
        raise ValueError("original train or public release identity differs")
    blocked = forbidden()
    p = producing(AUDIT, "CPU_public_wide_source_availability_not_model_quality",
        [source, DATA / "manifest.json", PUBLIC_FILE, *FORBIDDEN_FILES], {})
    index = public_index(q.read(PUBLIC_FILE))
    eligible, counts = [], Counter()
    with source.open() as handle:
        for line_number, line in enumerate(handle):
            row = json.loads(line)
            counts["original_train_rows"] += 1
            if row["split"] != "train":
                raise ValueError("original training partition changed")
            bucket = source_bin(row["input_tokens"])
            if bucket is None:
                counts["outside_requested_length_bins"] += 1
                continue
            if any(row[key] in blocked[key] for key in IDENTITY):
                counts["known_source_group_excluded"] += 1
                continue
            if not matches_public(row, index):
                counts["not_exact_public_spans"] += 1
                continue
            eligible.append({**{key: row[key] for key in IDENTITY}, "line_number": line_number,
                             "source_cl100k_tokens": row["input_tokens"], "bin": bucket})
    if counts["original_train_rows"] != manifest["files"][source.name]["rows"]:
        raise ValueError("original train row count changed")
    verify(p)
    write_json(AUDIT / "eligible_index.json", eligible)
    result = {"status": "availability_audited", "counts": dict(counts), "eligible_rows": len(eligible),
        "unique_groups": len({r["group_id"] for r in eligible}),
        "rows_by_bin": [sum(r["bin"] == i for r in eligible) for i in range(len(BINS))],
        "groups_by_bin": [len({r["group_id"] for r in eligible if r["bin"] == i}) for i in range(len(BINS))],
        "index_sha256": file_hash(AUDIT / "eligible_index.json"), "producer_fingerprint": p["fingerprint"],
        "task_accuracy": None, "NCM_success_established": False, "final_test_opened": False, "speed_tested": False}
    write_json(AUDIT / "result.json", result)
    print(json.dumps(result), flush=True)


def select(index, quotas):
    if set(quotas) != {"train", "diagnostic", "sealed_final"} or any(type(n) is not int or n < 1 for n in quotas.values()):
        raise ValueError("positive explicit per-bin quotas for all three splits required")
    used = {key: set() for key in IDENTITY}
    chosen = {name: [] for name in quotas}
    for bucket in reversed(range(len(BINS))):
        ordered = sorted((r for r in index if r["bin"] == bucket),
                         key=lambda r: text_hash("ncm-application-range-source-v1:" + r["id"]))
        for name in ("sealed_final", "diagnostic", "train"):
            count = 0
            for row in ordered:
                if any(row[key] in used[key] for key in IDENTITY):
                    continue
                chosen[name].append(row)
                for key in IDENTITY:
                    used[key].add(row[key])
                count += 1
                if count == quotas[name]:
                    break
            if count != quotas[name]:
                raise ValueError(f"insufficient disjoint groups for bin{bucket}/{name}: requested{quotas[name]},found{count}")
    return chosen


def prepare(quotas):
    from experiments.structure_net.ncm_native_label_quality import dependencies
    dependencies()
    import tiktoken
    from transformers import AutoTokenizer
    from experiments.structure_net.ncm_frozen_body import MODEL_PATH
    from experiments.structure_net.ncm_native_parent_control import SYSTEM
    verify(q.read(AUDIT / "producer.json"))
    if q.read(AUDIT / "worker_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("availability audit must be observed terminal")
    if file_hash(AUDIT / "eligible_index.json") != q.read(AUDIT / "result.json")["index_sha256"]:
        raise ValueError("availability index changed")
    p = producing(POOL, "CPU_unlabeled_source_selection_no_training_admission",
        [AUDIT / n for n in ("producer.json", "result.json", "eligible_index.json", "worker_terminal_receipt.json")]
        + [DATA / "train.jsonl", PUBLIC_FILE, *FORBIDDEN_FILES,
           Path("experiments/structure_net/ncm_native_parent_control.py"), MODEL_PATH / "tokenizer.json", MODEL_PATH / "tokenizer_config.json"],
        {"per_bin_quotas": quotas, "native_system_prompt": SYSTEM, "no_source_truncation": True})
    try:
        chosen = select(q.read(AUDIT / "eligible_index.json"), quotas)
    except ValueError as error:
        write_json(POOL / "result.json", {"status": "insufficient_disjoint_availability", "error": str(error),
            "requested_per_bin": quotas, "producer_fingerprint": p["fingerprint"], "NCM_success_established": False})
        raise
    needed = {row["line_number"]: row for rows in chosen.values() for row in rows}
    encoding = tiktoken.get_encoding("cl100k_base")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    public = public_index(q.read(PUBLIC_FILE))
    recovered = {}
    with (DATA / "train.jsonl").open() as handle:
        for line_number, line in enumerate(handle):
            if line_number not in needed:
                continue
            row = json.loads(line); entry = needed[line_number]
            tokens = len(encoding.encode(row["source"], disallowed_special=()))
            if (any(row[key] != entry[key] for key in IDENTITY) or tokens != entry["source_cl100k_tokens"]
                    or text_hash(row["source"]) != row["source_sha256"] or not matches_public(row, public)):
                raise ValueError("selected complete public source or actual token count changed")
            prompt = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "<passage>\n" + row["source"] + "\n</passage>"}], add_generation_prompt=True)
            recovered[line_number] = {**clean_source(row), "source_cl100k_tokens": tokens, "bin": entry["bin"],
                                      "native_prompt_tokens": len(prompt)}
    if len(recovered) != len(needed):
        raise ValueError("selected source missing")
    exported = {name: [{**recovered[r["line_number"]], "pool_split": name} for r in rows] for name, rows in chosen.items()}
    hashes = {}
    for name, rows in exported.items():
        path = POOL / (name + "_sources.json")
        write_json(path, rows); hashes[str(path)] = file_hash(path)
    verify(p)
    result = {"status": "unlabeled_public_pool_prepared", "counts": {k: len(v) for k, v in exported.items()},
        "per_bin_quotas": quotas, "native_prompt_min": min(v["native_prompt_tokens"] for v in recovered.values()),
        "native_prompt_max": max(v["native_prompt_tokens"] for v in recovered.values()),
        "source_cl100k_min": min(v["source_cl100k_tokens"] for v in recovered.values()),
        "source_cl100k_max": max(v["source_cl100k_tokens"] for v in recovered.values()), "files": hashes,
        "unique_groups": len({v["group_id"] for v in recovered.values()}), "old_targets_exported": False,
        "new_model_generation": False, "task_accuracy": None, "NCM_success_established": False,
        "original_final_test_opened": False, "new_sealed_sources_selected_without_outputs": True,
        "training_admitted": False, "speed_tested": False, "producer_fingerprint": p["fingerprint"]}
    write_json(POOL / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["audit", "prepare"])
    parser.add_argument("--train-per-bin", type=int)
    parser.add_argument("--diagnostic-per-bin", type=int)
    parser.add_argument("--final-per-bin", type=int)
    args = parser.parse_args()
    if args.command == "audit":
        audit()
    else:
        prepare({"train": args.train_per_bin, "diagnostic": args.diagnostic_per_bin, "sealed_final": args.final_per_bin})
