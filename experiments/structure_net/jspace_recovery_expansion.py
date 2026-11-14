"""Freeze additional training examples without opening confirmation content.

This CPU-only contingency consumes pinned training Parquets. Confirmation
exclusions are extracted from identity-field lines only: prompts, choices,
answers and tokens from that population are never parsed or selected here.
No teacher/model inference or checkpoint training is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil

from .jspace_bench_data import DATASETS, normalize

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT/"data/experiments/jspace_recovery_expansion/20261114_v1"
JOINT_ROOT = ROOT/"data/experiments/jspace_joint_recovery/20261114_v1"
CONFIRMATION_ROOT = ROOT/"data/experiments/jspace_accuracy_recovery_data/20261114_v1"
DATA_ROOT = ROOT/"data/benchmarks/jspace_four_tasks_20261113"
CONFIG = dict(seed=582462, additional_training=4096, max_tokens=384, tokenizer_batch=64)
SYSTEM = "Answer the multiple-choice question. Reply with only the letter of the correct option."


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def confirmation_identities_only(path, *, expected_count=512):
    """Read identity-field lines from the frozen pretty-printed artifact.

    No JSON decoder is used and no other field value is inspected. A prompt
    containing the word 'identity' cannot match an anchored JSON field line.
    File identity is separately verified against the frozen manifest hash.
    """
    pattern = re.compile(r'^\s*"identity"\s*:\s*"([0-9a-f]{64})"\s*,?\s*$')
    identities = []
    with Path(path).open() as stream:
        for line in stream:
            match = pattern.fullmatch(line.rstrip("\n"))
            if match:
                identities.append(match.group(1))
    if len(identities) != expected_count or len(set(identities)) != expected_count:
        raise ValueError("Confirmation identity count/uniqueness mismatch; do not fall back to content parsing")
    return set(identities)


def balanced_quotas(available, total):
    """Deterministic max-min allocation, with family order breaking one-row ties."""
    if (type(total) is not int or total < 1 or not available
            or any(type(n) is not int or n < 0 for n in available.values())
            or sum(available.values()) < total):
        raise ValueError("Insufficient eligible training rows or invalid allocation")
    quotas = dict.fromkeys(available, 0)
    remaining = total
    while remaining:
        for family, count in available.items():
            if quotas[family] < count and remaining:
                quotas[family] += 1
                remaining -= 1
    return quotas


def _prompt(question, choices):
    return question+"\n\n"+"\n".join(f"{chr(65+i)}. {value}" for i,value in enumerate(choices))


def scan_training_family(family, rows, excluded, tokenizer, *, seed, max_tokens=384,
                         keep_limit=4096, batch_size=64):
    """Count every eligible row; retain only a bounded deterministic prefix."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    seen, pending, retained = set(excluded), [], []
    discovered = set()
    counts = Counter(raw_rows=len(rows))
    eligible_tokens = 0

    def consume():
        nonlocal eligible_tokens
        if not pending:
            return
        conversations = [[{"role":"system", "content":SYSTEM},
                          {"role":"user", "content":case["prompt"]}] for case in pending]
        encoded = tokenizer.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True)
        if len(encoded) != len(pending):
            raise ValueError("Tokenizer did not preserve batch rows")
        for case, tokens in zip(pending, encoded):
            if not 3 <= len(tokens) <= max_tokens:
                counts["outside_token_limit"] += 1
                continue
            counts["eligible_total"] += 1
            eligible_tokens += len(tokens)
            if len(retained) < keep_limit:
                retained.append(dict(case, input_ids=list(tokens)))
        pending.clear()

    for index in order:
        try:
            question, choices, answer = normalize(family, rows[index])
        except ValueError:
            counts["unsupported"] += 1
            continue
        identity = hashlib.sha256(json.dumps([question, choices]).encode()).hexdigest()
        if identity in seen:
            counts["excluded_or_duplicate"] += 1
            continue
        seen.add(identity)
        discovered.add(identity)
        counts["unseen_unique_before_length_filter"] += 1
        pending.append(dict(identity=identity, family=family, source_split="train", source_row=index,
                            prompt=_prompt(question, choices), choices=choices, answer=answer))
        if len(pending) == batch_size:
            consume()
    consume()
    report = dict(counts)
    report.update(eligible_total=counts["eligible_total"], eligible_total_tokens=eligible_tokens,
                  retained_candidate_count=len(retained), candidate_order_seed=seed)
    return retained, report, discovered


def assemble_items(original_items, additions, label_ids):
    """Keep original rows/order verbatim; append new training/logit rows."""
    old = list(original_items)
    if Counter(c["split"] for c in old) != {"train":512, "development":64}:
        raise ValueError("Original512 training and64 development cases must remain intact")
    identities = {c["identity"] for c in old}
    if len(identities) != len(old):
        raise ValueError("Original identities are duplicated")
    offset = max(c.get("stop", 0) for c in old)
    new = []
    for case in additions:
        if case["identity"] in identities or case["source_split"] != "train":
            raise ValueError("New training rows overlap or came from a non-training source")
        identities.add(case["identity"])
        tokens = case["input_ids"]
        positions = [0, (len(tokens)-2)//2, len(tokens)-2, len(tokens)-1]
        if len(tokens) < 3 or not 0 <= case["answer"] < len(case["choices"]) <= len(label_ids):
            raise ValueError("Invalid training tokens or choice targets")
        targets = [tokens[p+1] for p in positions[:-1]]+[label_ids[case["answer"]]]
        new.append(dict(case, split="train", offset=offset, stop=offset+len(tokens),
                        loss_positions=positions, targets=targets))
        offset += len(tokens)
    return old+new


def freeze(root, *, joint_root=JOINT_ROOT, confirmation_root=CONFIRMATION_ROOT):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Expansion freezing requires CUDA_VISIBLE_DEVICES=''")
    root, joint_root, confirmation_root = map(lambda p:Path(p).resolve(), (root,joint_root,confirmation_root))
    if root.exists():
        raise FileExistsError(root)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    joint = read(joint_root/"plan.json")
    if joint["fingerprint"] != digest({k:v for k,v in joint.items() if k != "fingerprint"}):
        raise ValueError("Changed joint source plan")
    original = read(joint_root/"items.json")
    if digest(original) != joint["items_digest"]:
        raise ValueError("Changed original training/development rows")
    parent_path = Path(joint["source"])/"plan.json"
    parent = read(parent_path)
    if (parent["fingerprint"] != joint["source_fingerprint"] or
            parent["fingerprint"] != digest({k:v for k,v in parent.items() if k != "fingerprint"})):
        raise ValueError("Changed original native model source")
    confirmation_manifest = read(confirmation_root/"manifest.json")
    if confirmation_manifest["status"] != "frozen_unscored":
        raise ValueError("Confirmation population must remain frozen and unscored")
    input_sha = {str(p):sha(p) for p in (joint_root/"plan.json", joint_root/"items.json", parent_path,
                 confirmation_root/"manifest.json", DATA_ROOT/"manifest.json")}
    for name in ("plan.json", "tasks.json", "exclusions.json"):
        path = confirmation_root/name
        observed = sha(path)
        if confirmation_manifest["artifacts"][name] != observed:
            raise ValueError("Changed frozen confirmation identity source")
        input_sha[str(path)] = observed
    confirmation_ids = confirmation_identities_only(confirmation_root/"tasks.json")
    prior_ids = set(read(confirmation_root/"exclusions.json")["identities"])
    original_ids = {c["identity"] for c in original["items"]}
    if original_ids & confirmation_ids:
        raise ValueError("Original fitting population overlaps untouched confirmation")
    excluded = prior_ids | confirmation_ids | original_ids
    model = Path(parent["model"]["path"])
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"):
        expected = parent["model"]["files"][name]
        if sha(model/name) != expected:
            raise ValueError("Changed pinned tokenizer/config")
        input_sha[str(model/name)] = expected
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    manifest = read(DATA_ROOT/"manifest.json")
    pools, inventories = {}, {}
    for index,family in enumerate(DATASETS):
        entry = manifest[family]["files"]["train"]
        path = DATA_ROOT/entry["local"]
        if sha(path) != entry["sha256"]:
            raise ValueError("Changed pinned training dataset")
        input_sha[str(path)] = entry["sha256"]
        rows = pq.read_table(path).to_pylist()
        pool, inventory, discovered = scan_training_family(family, rows, excluded, tokenizer,
            seed=CONFIG["seed"]+index, max_tokens=CONFIG["max_tokens"],
            keep_limit=CONFIG["additional_training"], batch_size=CONFIG["tokenizer_batch"])
        pools[family], inventories[family] = pool, inventory
        excluded.update(discovered)
        print(json.dumps(dict(stage="training_inventory", family=family, **inventory)), flush=True)
        del rows
    quotas = balanced_quotas({f:d["eligible_total"] for f,d in inventories.items()}, CONFIG["additional_training"])
    additions = [case for family in DATASETS for case in pools[family][:quotas[family]]]
    items = assemble_items(original["items"], additions, joint["label_ids"])
    added = items[len(original["items"]):]
    if {c["identity"] for c in additions} & (prior_ids|confirmation_ids|original_ids):
        raise ValueError("Expansion leakage")
    tasks = dict(splits={s:[c for c in items if c["split"] == s] for s in ("train","development")},
                 scope="Training expansion only; original development unchanged; no confirmation content included")
    token_counts = {s:sum(len(c["input_ids"]) for c in cases) for s,cases in tasks["splits"].items()}
    new_tokens = sum(len(c["input_ids"]) for c in added)
    source_paths = [Path(__file__).resolve(), ROOT/"experiments/structure_net/jspace_bench_data.py",
                    ROOT/"tests/structure_net/test_jspace_recovery_expansion.py"]
    plan = dict(schema="jspace.recovery-expansion.v1", config=CONFIG,
        created_at=datetime.now(timezone.utc).isoformat(), joint_root=str(joint_root),
        joint_fingerprint=joint["fingerprint"], original_model_fingerprint=parent["fingerprint"],
        confirmation_fingerprint=confirmation_manifest["fingerprint"],
        original_items_digest=joint["items_digest"], items_digest=digest(items), tasks_digest=digest(tasks),
        inventory=inventories, quotas=quotas, counts={s:len(v) for s,v in tasks["splits"].items()},
        token_counts=token_counts, new_training_tokens=new_tokens,
        source_sha256={str(p):sha(p) for p in source_paths}, input_sha256=input_sha,
        original_teacher_path=joint["teacher_path"], original_teacher_rows=len(original["items"]),
        original_teacher_sha256=joint["input_sha256"][joint["teacher_path"]],
        new_teacher_rows=len(added), label_ids=joint["label_ids"],
        teacher_logits=dict(vocabulary=151936, positions=4, storage="native BF16 bits in int16",
            new_payload_bytes=len(added)*4*151936*2, combined_payload_bytes=len(items)*4*151936*2),
        confirmation_policy="Only identity-field lines extracted; no prompts, answers, choices or input IDs parsed",
        checkpoint_policy="No checkpoint selected here; downstream warm-start must freeze a completed development-selected checkpoint and reset optimizer")
    plan["fingerprint"] = digest(plan)
    root.mkdir(parents=True)
    for path in source_paths:
        target = root/"producer"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write(root/"plan.json", plan)
    write(root/"items.json", dict(items=items, tokens=sum(token_counts.values()), original_item_count=len(original["items"])))
    write(root/"tasks.json", tasks)
    write(root/"new_training.json", added)
    write(root/"confirmation_identities.json", dict(identities=sorted(confirmation_ids), source_fingerprint=confirmation_manifest["fingerprint"]))
    write(root/"manifest.json", dict(status="frozen_untrained", fingerprint=plan["fingerprint"],
        artifacts={name:sha(root/name) for name in ("plan.json","items.json","tasks.json","new_training.json","confirmation_identities.json")}))
    return verify(root)


def verify(root):
    root = Path(root)
    plan, manifest = read(root/"plan.json"), read(root/"manifest.json")
    if (plan["fingerprint"] != digest({k:v for k,v in plan.items() if k != "fingerprint"})
            or manifest["fingerprint"] != plan["fingerprint"] or manifest["status"] != "frozen_untrained"):
        raise ValueError("Changed expansion protocol")
    for name,expected in manifest["artifacts"].items():
        if sha(root/name) != expected:
            raise ValueError("Changed expansion artifact")
    for name,expected in plan["source_sha256"].items():
        snapshot = root/"producer"/Path(name).relative_to(ROOT)
        if sha(name) != expected or sha(snapshot) != expected:
            raise ValueError("Changed expansion producer")
    for name,expected in plan["input_sha256"].items():
        if sha(name) != expected:
            raise ValueError("Changed pinned expansion input")
    items, tasks, added = read(root/"items.json")["items"], read(root/"tasks.json"), read(root/"new_training.json")
    ids = {c["identity"] for c in items}
    excluded = set(read(root/"confirmation_identities.json")["identities"])
    original = read(Path(plan["joint_root"])/"items.json")
    if (digest(original) != plan["original_items_digest"] or items[:plan["original_teacher_rows"]] != original["items"]
            or digest(items) != plan["items_digest"] or digest(tasks) != plan["tasks_digest"]
            or len(ids) != len(items) or ids & excluded or len(added) != 4096
            or Counter(c["split"] for c in items) != {"train":4608,"development":64}
            or added != items[plan["original_teacher_rows"]:]
            or any(c["source_split"] != "train" or not 3 <= len(c["input_ids"]) <= 384 for c in added)):
        raise ValueError("Population, preserved development, or confirmation-exclusion mismatch")
    return dict(status="frozen_untrained", root=str(root), fingerprint=plan["fingerprint"],
                counts=plan["counts"], additional_by_family=plan["quotas"], token_counts=plan["token_counts"],
                teacher_logits=plan["teacher_logits"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze","verify"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--joint-root", type=Path, default=JOINT_ROOT)
    parser.add_argument("--confirmation-root", type=Path, default=CONFIRMATION_ROOT)
    args = parser.parse_args()
    result = freeze(args.root, joint_root=args.joint_root, confirmation_root=args.confirmation_root) if args.stage == "freeze" else verify(args.root)
    print(json.dumps(result, allow_nan=False))
