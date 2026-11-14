"""Pinned public multiple-choice calibration and held-out task splits."""
import hashlib
import json
from pathlib import Path
import random
import requests


DATASETS = {
    "arc": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "boolq": ("google/boolq", "data", "validation"),
    "hellaswag": ("Rowan/hellaswag", "data", "validation"),
    "winogrande": ("allenai/winogrande", "winogrande_debiased", "validation"),
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(text)


def fetch(root):
    """Download public parquet files; each dataset revision is pinned before use."""
    root = Path(root)
    if (root / "manifest.json").exists():
        manifest = json.loads((root / "manifest.json").read_text())
        for entry in manifest.values():
            for f in entry["files"].values():
                if sha(root / f["local"]) != f["sha256"]:
                    raise ValueError("Dataset cache digest mismatch")
        return manifest
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for family, (repo, folder, test_split) in DATASETS.items():
        response = requests.get("https://huggingface.co/api/datasets/" + repo, timeout=30)
        response.raise_for_status()
        revision = response.json()["sha"]
        files = {}
        for split in ("train", test_split):
            remote = f"{folder}/{split}-00000-of-00001.parquet"
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{remote}"
            local = root / f"{family}_{split}.parquet"
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            if local.exists():
                if local.read_bytes() != response.content:
                    raise ValueError("Partial download conflicts with pinned data")
            else:
                with local.open("xb") as stream:
                    stream.write(response.content)
            files[split] = {"url": url, "local": local.name, "sha256": sha(local)}
        manifest[family] = {"repository": repo, "revision": revision, "files": files,
                            "evaluation_source_split": test_split}
    write(root / "manifest.json", manifest)
    return manifest


def normalize(family, row):
    if family == "arc":
        question = row["question"]
        choices = list(row["choices"]["text"])
        answer = list(row["choices"]["label"]).index(row["answerKey"])
    elif family == "boolq":
        question = row["passage"] + "\n\nQuestion: " + row["question"]
        choices, answer = ["No", "Yes"], int(row["answer"])
    elif family == "hellaswag":
        question = "Choose the most plausible continuation.\n" + row["ctx"]
        choices, answer = list(row["endings"]), int(row["label"])
    elif family == "winogrande":
        question = "Fill the blank in this sentence.\n" + row["sentence"]
        choices, answer = [row["option1"], row["option2"]], int(row["answer"]) - 1
    else:
        raise ValueError(family)
    if not 2 <= len(choices) <= 4 or not 0 <= answer < len(choices):
        raise ValueError("Unsupported or unlabeled multiple-choice item")
    return question, choices, answer


def make_tasks(root, tokenizer, config):
    import pyarrow.parquet as pq
    manifest = fetch(root)
    result = {"train": [], "development": [], "evaluation": []}
    exclusions = {}
    seen = set()
    for fi, (family, entry) in enumerate(manifest.items()):
        pools = {}
        for si, source_split in enumerate(("train", entry["evaluation_source_split"])):
            rows = pq.read_table(Path(root) / entry["files"][source_split]["local"]).to_pylist()
            indices = list(range(len(rows)))
            random.Random(config["seed"] + 100 * fi + si).shuffle(indices)
            kept, skipped = [], {"long": 0, "duplicate": 0}
            needed = config["train_per_task"] + config["dev_per_task"] if si == 0 else config["eval_per_task"]
            for index in indices:
                question, choices, answer = normalize(family, rows[index])
                identity = hashlib.sha256(json.dumps([question, choices]).encode()).hexdigest()
                if identity in seen:
                    skipped["duplicate"] += 1
                    continue
                user = question + "\n\n" + "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices))
                messages = [{"role": "system", "content": "Answer the multiple-choice question. Reply with only the letter of the correct option."},
                            {"role": "user", "content": user}]
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                if len(ids) > config["max_tokens"]:
                    skipped["long"] += 1
                    continue
                seen.add(identity)
                kept.append({"family": family, "source_split": source_split, "source_row": index,
                             "identity": identity, "prompt": user, "input_ids": ids,
                             "choices": choices, "answer": answer})
                if len(kept) == needed:
                    break
            if len(kept) != needed:
                raise ValueError("Not enough eligible benchmark items")
            pools[source_split] = kept
            exclusions[family + "/" + source_split] = skipped
        cut = config["train_per_task"]
        result["train"] += pools["train"][:cut]
        result["development"] += pools["train"][cut:]
        result["evaluation"] += pools[entry["evaluation_source_split"]]
    return {"splits": result, "datasets": manifest, "exclusions_before_count": exclusions,
            "scope": "length-filtered single-letter multiple-choice accuracy; no free-generation quality claim"}
