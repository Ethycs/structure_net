"""Freeze untouched confirmation data for accuracy recovery; never score models."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import shutil

from .jspace_bench_data import DATASETS, normalize

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "data/experiments/jspace_accuracy_recovery_data/20261114_v1"
DATA_ROOT = ROOT / "data/benchmarks/jspace_four_tasks_20261113"
PARENT = ROOT / "data/experiments/jspace_quadratic_recipe/20261114_v1"
CONFIG = dict(seed=582461, confirmation_per_family=128, max_tokens=384)
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


def all_identities(value):
    """Include historical exclusion lists, not just the newest split rows."""
    result = set()
    if isinstance(value, dict):
        if isinstance(value.get("identity"), str):
            result.add(value["identity"])
        for key, child in value.items():
            if key.endswith("identities") and isinstance(child, list):
                result.update(x for x in child if isinstance(x, str))
            result.update(all_identities(child))
    elif isinstance(value, list):
        for child in value:
            result.update(all_identities(child))
    return result


def identity(question, choices):
    return hashlib.sha256(json.dumps([question, choices]).encode()).hexdigest()


def candidate_rows(family, rows, excluded):
    seen = set(excluded)
    result, skipped = [], Counter()
    for index, row in enumerate(rows):
        try:
            question, choices, answer = normalize(family, row)
        except ValueError:
            skipped["unsupported"] += 1
            continue
        key = identity(question, choices)
        if key in seen:
            skipped["excluded_or_duplicate"] += 1
            continue
        seen.add(key)
        result.append(dict(identity=key, family=family, source_row=index,
                           question=question, choices=choices, answer=answer))
    return result, dict(skipped)


def select_confirmation(family, rows, excluded, tokenizer, *, count=128, max_tokens=384, seed=582461):
    candidates, skipped = candidate_rows(family, rows, excluded)
    random.Random(seed).shuffle(candidates)
    selected = []
    for case in candidates:
        prompt = case["question"] + "\n\n" + "\n".join(
            f"{chr(65+i)}. {choice}" for i, choice in enumerate(case["choices"]))
        tokens = tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True)
        if len(tokens) > max_tokens:
            skipped["long_before_selection"] = skipped.get("long_before_selection", 0) + 1
            continue
        selected.append({k: v for k, v in case.items() if k != "question"} |
                        dict(prompt=prompt, input_ids=tokens))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} eligible untouched cases for {family}; need {count}")
    return selected, dict(unseen_unique_before_length_filter=len(candidates), skipped=skipped)


def freeze(root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Data freezing requires hidden CUDA")
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError(root)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    parent = read(PARENT / "plan.json")
    if parent["fingerprint"] != digest({k: v for k, v in parent.items() if k != "fingerprint"}):
        raise ValueError("Changed parent scientific plan")
    prior_paths = sorted((ROOT / "data/experiments").glob("jspace*/**/tasks.json"))
    excluded = set().union(*(all_identities(read(path)) for path in prior_paths))
    if not excluded or not prior_paths:
        raise ValueError("Historical task inventory is empty")
    input_sha = {str(path): sha(path) for path in [*prior_paths, PARENT / "plan.json", DATA_ROOT / "manifest.json"]}
    model = Path(parent["model"]["path"])
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"):
        expected = parent["model"]["files"][name]
        if sha(model / name) != expected:
            raise ValueError("Changed tokenizer/config: " + name)
        input_sha[str(model / name)] = expected
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    manifest = read(DATA_ROOT / "manifest.json")
    selected, inventories = [], {}
    for family_index, family in enumerate(DATASETS):
        entry = manifest[family]
        inventories[family] = {}
        for split, info in entry["files"].items():
            path = DATA_ROOT / info["local"]
            if sha(path) != info["sha256"]:
                raise ValueError("Changed pinned dataset: " + str(path))
            input_sha[str(path)] = info["sha256"]
            rows = pq.read_table(path).to_pylist()
            candidates, skipped = candidate_rows(family, rows, excluded)
            inventories[family][split] = dict(raw_rows=len(rows),
                unseen_unique_before_length_filter=len(candidates), skipped=skipped)
            if split == entry["evaluation_source_split"]:
                cases, details = select_confirmation(family, rows, excluded | {c["identity"] for c in selected},
                    tokenizer, count=CONFIG["confirmation_per_family"], max_tokens=CONFIG["max_tokens"],
                    seed=CONFIG["seed"]+family_index)
                selected += [dict(case, source_split=split) for case in cases]
                inventories[family][split]["confirmation_selection"] = details
    tasks = dict(splits={"confirmation": selected}, status="frozen_unscored",
        scope="Untouched confirmation only; no checkpoint selection, fitting, or tuning may use these cases")
    exclusions = dict(identities=sorted(excluded), source_task_files=[str(p) for p in prior_paths])
    source_paths = [Path(__file__), ROOT / "experiments/structure_net/jspace_bench_data.py",
                    ROOT / "tests/structure_net/test_jspace_recovery_data.py"]
    plan = dict(schema="jspace.accuracy-recovery-data.v1", config=CONFIG,
        created_at=datetime.now(timezone.utc).isoformat(), parent_fingerprint=parent["fingerprint"],
        input_sha256=input_sha, source_sha256={str(path): sha(path) for path in source_paths},
        excluded_identity_count=len(excluded), exclusions_digest=digest(exclusions),
        task_digest=digest(tasks), inventory=inventories,
        tokenizer_model=parent["model"], runtime_packages={name: version(name) for name in ("transformers", "tokenizers", "pyarrow")},
        policy="Select and hash before recovery; reveal scoring only after development-selected candidate is frozen",
        inventory_scope="Unused source-row counts precede384-token eligibility; only selected confirmation eligibility is established")
    plan["fingerprint"] = digest(plan)
    root.mkdir(parents=True)
    for path in source_paths:
        target = root / "producer" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write(root / "plan.json", plan); write(root / "tasks.json", tasks); write(root / "exclusions.json", exclusions)
    write(root / "manifest.json", dict(status="frozen_unscored", fingerprint=plan["fingerprint"],
        artifacts={name: sha(root / name) for name in ("plan.json", "tasks.json", "exclusions.json")}))
    return verify(root)


def verify(root):
    root = Path(root)
    plan, tasks, exclusions, manifest = (read(root / name) for name in
        ("plan.json", "tasks.json", "exclusions.json", "manifest.json"))
    if plan["fingerprint"] != digest({k: v for k, v in plan.items() if k != "fingerprint"}):
        raise ValueError("Changed confirmation plan")
    if manifest["fingerprint"] != plan["fingerprint"] or manifest["status"] != "frozen_unscored":
        raise ValueError("Changed confirmation status")
    if digest(tasks) != plan["task_digest"] or digest(exclusions) != plan["exclusions_digest"]:
        raise ValueError("Changed confirmation identities")
    for name, expected in manifest["artifacts"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed confirmation artifact")
    for name, expected in plan["input_sha256"].items():
        if sha(name) != expected:
            raise ValueError("Changed pinned source data")
    for name, expected in plan["source_sha256"].items():
        if sha(name) != expected or sha(root / "producer" / Path(name).relative_to(ROOT)) != expected:
            raise ValueError("Changed data-freezer source")
    cases = tasks["splits"]["confirmation"]
    identities = {case["identity"] for case in cases}
    counts = Counter(case["family"] for case in cases)
    if (len(identities) != 512 or len(cases) != 512 or identities & set(exclusions["identities"]) or
            counts != {family: 128 for family in DATASETS} or
            any(not 0 < len(case["input_ids"]) <= 384 for case in cases)):
        raise ValueError("Confirmation leakage, duplicates, population or length mismatch")
    return dict(status="frozen_unscored", root=str(root), fingerprint=plan["fingerprint"],
                count=len(cases), excluded_identity_count=len(exclusions["identities"]), families=dict(counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "verify"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](args.root), allow_nan=False))
