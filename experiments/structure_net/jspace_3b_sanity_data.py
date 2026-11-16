"""Freeze a new balanced sanity set while leaving final confirmation sealed."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import shutil

from .jspace_bench_data import DATASETS, normalize
from .jspace_recovery_data import SYSTEM, digest, identity, read, sha, write

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT/"data/benchmarks/jspace_four_tasks_20261113"
CONFIRMATION = ROOT/"data/experiments/jspace_accuracy_recovery_data/20261114_v1"
EXPANSION = ROOT/"data/experiments/jspace_recovery_expansion/20261114_v1"
CONFIG = dict(seed=582463, per_family=32, max_tokens=384)
PROMPT_COLUMNS = dict(arc=["question", "choices"], boolq=["passage", "question"],
    hellaswag=["ctx", "endings"], winogrande=["sentence", "option1", "option2"])
LABEL_COLUMNS = dict(arc="answerKey", boolq="answer", hellaswag="label", winogrande="answer")
HASH_PATTERN = re.compile(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")


def identity_candidates(path):
    """Conservatively extract hashes without decoding any example or label."""
    found = set()
    with Path(path).open("rb") as stream:
        for line in stream:
            found.update(value.decode("ascii") for value in HASH_PATTERN.findall(line))
    return found


def reserved_identities(confirmation=CONFIRMATION, expansion=EXPANSION):
    """Consume the existing identity-only manifest, never confirmation tasks."""
    confirmation, expansion = Path(confirmation).resolve(), Path(expansion).resolve()
    cp, cm = read(confirmation/"plan.json"), read(confirmation/"manifest.json")
    ep, em = read(expansion/"plan.json"), read(expansion/"manifest.json")
    for plan in (cp, ep):
        if plan["fingerprint"] != digest({k:v for k,v in plan.items() if k != "fingerprint"}):
            raise ValueError("Changed frozen identity-source plan")
    identity_path = expansion/"confirmation_identities.json"
    if (cm["status"] != "frozen_unscored" or cm["fingerprint"] != cp["fingerprint"]
            or em["status"] != "frozen_untrained" or em["fingerprint"] != ep["fingerprint"]
            or ep["confirmation_fingerprint"] != cp["fingerprint"]
            or sha(identity_path) != em["artifacts"][identity_path.name]):
        raise ValueError("Unauthenticated confirmation identity-only manifest")
    receipt = read(identity_path)
    identifiers = receipt["identities"]
    if (receipt["source_fingerprint"] != cp["fingerprint"] or len(identifiers) != 512
            or len(set(identifiers)) != 512 or any(not re.fullmatch("[0-9a-f]{64}", x) for x in identifiers)):
        raise ValueError("Exactly 512 reserved identities required")
    # Hashing bytes is permitted; no reserved task fields are decoded here.
    pinned = {}
    for name in ("plan.json", "tasks.json", "exclusions.json"):
        path = confirmation/name
        if sha(path) != cm["artifacts"][name]:
            raise ValueError("Reserved confirmation artifact changed")
        pinned[str(path)] = cm["artifacts"][name]
    for path in (confirmation/"manifest.json", expansion/"plan.json", expansion/"manifest.json", identity_path):
        pinned[str(path)] = sha(path)
    return set(identifiers), cp, pinned


def prompt_parts(family, row):
    """Reuse exact benchmark formatting without reading a source answer."""
    dummy = dict(row)
    if family == "arc":
        dummy["answerKey"] = row["choices"]["label"][0]
    else:
        dummy[LABEL_COLUMNS[family]] = False if family == "boolq" else "1" if family == "winogrande" else "0"
    question, choices, _ = normalize(family, dummy)
    return question, choices


def select_unlabeled(family, rows, excluded, tokenizer, *, seed, count=32, max_tokens=384):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    selected, seen = [], set(excluded)
    stats = Counter(raw_rows=len(rows))
    for index in order:
        try:
            question, choices = prompt_parts(family, rows[index])
        except (ValueError, IndexError):
            stats["unsupported"] += 1
            continue
        key = identity(question, choices)
        if key in seen:
            stats["excluded_or_duplicate"] += 1
            continue
        seen.add(key)
        prompt = question+"\n\n"+"\n".join(f"{chr(65+i)}. {c}" for i,c in enumerate(choices))
        tokens = tokenizer.apply_chat_template([
            dict(role="system", content=SYSTEM), dict(role="user", content=prompt)],
            tokenize=True, add_generation_prompt=True)
        if not 3 <= len(tokens) <= max_tokens:
            stats["outside_token_limit"] += 1
            continue
        selected.append(dict(identity=key, family=family, source_row=index, prompt=prompt,
            choices=choices, input_ids=list(tokens), split="sanity"))
        if len(selected) == count:
            return selected, dict(stats)
    raise ValueError(f"Only {len(selected)} unseen eligible sanity cases for {family}; need {count}")


def validate_cases(cases, exclusions, *, per_family=32, max_tokens=384):
    keys = {c["identity"] for c in cases}
    if (len(keys) != len(cases) or keys & set(exclusions)
            or Counter(c["family"] for c in cases) != {f:per_family for f in DATASETS}
            or any(c["split"] != "sanity" or not 3 <= len(c["input_ids"]) <= max_tokens
                   or not 0 <= c["answer"] < len(c["choices"]) for c in cases)):
        raise ValueError("Sanity balance, disjointness, label or token bound failed")


def freeze(root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Sanity data preparation requires CUDA hidden")
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError(root)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    reserved, confirmation, inputs = reserved_identities()
    sealed_tasks = (CONFIRMATION/"tasks.json").resolve()
    historical = set()
    for pattern in ("jspace*/**/tasks.json", "jspace*/**/items.json", "jspace*/**/new_training.json", "jspace*/**/exclusions.json"):
        historical.update(p.resolve() for p in (ROOT/"data/experiments").glob(pattern))
    historical.discard(sealed_tasks)
    if not historical:
        raise ValueError("Historical identity inventory is empty")
    excluded = set(reserved)
    inventory = []
    for path in sorted(historical):
        found = identity_candidates(path)
        excluded.update(found)
        inputs[str(path)] = sha(path)
        inventory.append(dict(path=str(path), sha256=inputs[str(path)], identity_candidates=len(found)))
    model = confirmation["tokenizer_model"]
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"):
        path = Path(model["path"])/name
        if sha(path) != model["files"][name]:
            raise ValueError("Pinned tokenizer changed")
        inputs[str(path)] = model["files"][name]
    tokenizer = AutoTokenizer.from_pretrained(model["path"], local_files_only=True, trust_remote_code=False)
    manifest = read(DATA_ROOT/"manifest.json")
    inputs[str(DATA_ROOT/"manifest.json")] = sha(DATA_ROOT/"manifest.json")
    cases, details = [], {}
    for i, family in enumerate(DATASETS):
        entry = manifest[family]
        split = entry["evaluation_source_split"]
        file = entry["files"][split]
        path = DATA_ROOT/file["local"]
        if sha(path) != file["sha256"]:
            raise ValueError("Pinned evaluation-source Parquet changed")
        inputs[str(path)] = file["sha256"]
        rows = pq.read_table(path, columns=PROMPT_COLUMNS[family]).to_pylist()
        selected, details[family] = select_unlabeled(family, rows, excluded | {c["identity"] for c in cases},
            tokenizer, seed=CONFIG["seed"]+i, count=CONFIG["per_family"], max_tokens=CONFIG["max_tokens"])
        # Materialize answer values only for selected, already-disjoint rows.
        labels = pq.read_table(path, columns=[LABEL_COLUMNS[family]]).take([c["source_row"] for c in selected]).to_pylist()
        for case, label in zip(selected, labels):
            _, choices, answer = normalize(family, {**rows[case["source_row"]], **label})
            if choices != case["choices"]:
                raise ValueError("Selected prompt/label source mismatch")
            cases.append(dict(case, answer=answer, source_split=split))
    validate_cases(cases, excluded)
    label_ids = [tokenizer.encode(chr(65+i), add_special_tokens=False) for i in range(4)]
    if any(len(ids) != 1 for ids in label_ids):
        raise ValueError("Single-token choice labels required")
    tasks = dict(status="frozen_unscored", splits={"sanity":cases}, scope="New sanity benchmark, not final confirmation; no fitting")
    exclusions = dict(identities=sorted(excluded), reserved_identities=sorted(reserved), historical_files=inventory,
        method="Conservative 64-hex identity candidates from historical task/item/exclusion bytes; reserved IDs from authenticated identity-only manifest")
    sources = [Path(__file__).resolve(), ROOT/"experiments/structure_net/jspace_recovery_data.py",
        ROOT/"experiments/structure_net/jspace_bench_data.py", ROOT/"tests/structure_net/test_jspace_3b_sanity_data.py"]
    plan = dict(schema="jspace.3b-sanity-data.v1", config=CONFIG, created_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=inputs, source_sha256={str(p):sha(p) for p in sources}, model=model,
        confirmation_fingerprint=confirmation["fingerprint"], reserved_count=len(reserved),
        historical_file_count=len(historical), exclusion_count=len(excluded), exclusions_digest=digest(exclusions),
        tasks_digest=digest(tasks), label_ids=[ids[0] for ids in label_ids], selection_inventory=details,
        token_counts={f:sum(len(c["input_ids"]) for c in cases if c["family"] == f) for f in DATASETS},
        confirmation_content_decoded=False, policy="Sanity only; frozen before native/compact scoring, no final-confirmation access")
    plan["fingerprint"] = digest(plan)
    root.mkdir(parents=True)
    for source in sources:
        destination = root/"producer"/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write(root/"plan.json", plan)
    write(root/"tasks.json", tasks)
    write(root/"exclusions.json", exclusions)
    write(root/"manifest.json", dict(status="frozen_unscored", fingerprint=plan["fingerprint"],
        artifacts={name:sha(root/name) for name in ("plan.json", "tasks.json", "exclusions.json")}))
    return verify(root)


def verify(root):
    root = Path(root)
    plan, manifest = read(root/"plan.json"), read(root/"manifest.json")
    if (plan["fingerprint"] != digest({k:v for k,v in plan.items() if k != "fingerprint"})
            or manifest["fingerprint"] != plan["fingerprint"] or manifest["status"] != "frozen_unscored"
            or plan["config"] != CONFIG):
        raise ValueError("Changed sanity data plan")
    for name, expected in manifest["artifacts"].items():
        if sha(root/name) != expected:
            raise ValueError("Changed sanity artifact")
    for name, expected in plan["input_sha256"].items():
        if sha(name) != expected:
            raise ValueError("Changed pinned sanity input")
    for name, expected in plan["source_sha256"].items():
        if sha(name) != expected or sha(root/"producer"/Path(name).relative_to(ROOT)) != expected:
            raise ValueError("Changed sanity producer")
    tasks, exclusions = read(root/"tasks.json"), read(root/"exclusions.json")
    if digest(tasks) != plan["tasks_digest"] or digest(exclusions) != plan["exclusions_digest"]:
        raise ValueError("Changed sanity identities")
    validate_cases(tasks["splits"]["sanity"], exclusions["identities"])
    return dict(status="frozen_unscored", root=str(root.resolve()), fingerprint=plan["fingerprint"], count=128,
        families={f:32 for f in DATASETS}, token_counts=plan["token_counts"], exclusion_count=plan["exclusion_count"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT/"data/experiments/jspace_3b_sanity_data/20261115_v1")
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](args.root), allow_nan=False))
