"""Fresh country-attribute lookup worlds and disjoint generic lens calibration."""
import copy
import hashlib
import json
from pathlib import Path
import random

from .jspace_relation_tasks import COUNTRIES, NAME_POOLS
from .jspace_relation_capture import _prompt
from .jspace_causal_components import query_start
from .jspace_relation_study import ROOT, digest, file_hash


ANSWERS = {
    "capital": dict(zip(COUNTRIES, ("Beijing", "Paris", "Berlin", "Moscow"))),
    "nationality": dict(zip(COUNTRIES, ("Chinese", "French", "German", "Russian"))),
}
HELDOUT = (("China", "nationality"), ("Russia", "capital"))
COUNTS = {"train": 384, "development": 24, "evaluation": 128, "shakedown": 4}
SEED = 541


def render(mapping, order, query, attribute, case_id, swap_to):
    country = mapping[query]
    question = (f"What is the {attribute} of the country assigned to {query}? "
                "Reply with one word only.")
    return {"case_id": case_id, "prompt": "Use these assignments.\n"+json.dumps({n: mapping[n] for n in order})+"\n"+question,
            "question": question, "mapping": mapping, "query": query, "attribute": attribute,
            "intermediate": country, "answer": ANSWERS[attribute][country], "swap_to": swap_to,
            "swap_answer": ANSWERS[attribute][swap_to], "composition_heldout": (country, attribute) in HELDOUT}


def make_tasks(tokenizer):
    seen, splits = set(), {}
    for split, count in COUNTS.items():
        rng = random.Random(SEED+{"train": 0, "development": 1009, "evaluation": 2017, "shakedown": 1000003}[split])
        cells = [(c, a) for c in COUNTRIES for a in ANSWERS
                 if split not in ("train", "development") or (c, a) not in HELDOUT]
        rows, attempts = [], 0
        while len(rows) < count and attempts < count*500:
            country, attribute = cells[len(rows) % len(cells)]
            attempts += 1
            names = rng.sample(list(NAME_POOLS[split]), 4)
            countries = rng.sample(list(COUNTRIES), 4)
            mapping = dict(zip(names, countries))
            world = digest(mapping)
            if world in seen:
                continue
            query = names[countries.index(country)]
            alternatives = [c for c in COUNTRIES if c != country
                            and (split not in ("train", "development") or (c, attribute) not in HELDOUT)]
            target = rng.choice(alternatives)
            other = names[countries.index(target)]
            cid = f"{split}-{len(rows):05d}-{world[:10]}-{attribute}"
            source = render(mapping, names, query, attribute, cid, target)
            donor = render(mapping, names, other, attribute, cid+":donor", country)
            prompts = [_prompt(tokenizer, row) for row in (source, donor)]
            ids = [tokenizer(p)["input_ids"] for p in prompts]
            starts = [query_start(tokenizer, p, row["question"]) for p, row in zip(prompts, (source, donor))]
            if len(ids[0]) != len(ids[1]) or starts[0] != starts[1] or ids[0][:starts[0]] != ids[1][:starts[1]]:
                continue
            if len(ids[0]) > 192 or starts[0] >= len(ids[0])-1:
                raise ValueError("Prompt outside frozen length/position bounds")
            seen.add(world)
            rows.append({"source": source, "donor": donor, "world_id": world,
                         "question_start": starts[0], "sequence_length": len(ids[0])})
        if len(rows) != count:
            raise ValueError(f"Could not construct all {split} pairs")
        rng.shuffle(rows)
        splits[split] = rows
    payload = {"seed": SEED, "counts": COUNTS, "countries": list(COUNTRIES), "answers": ANSWERS,
               "heldout_country_attribute_cells": [list(c) for c in HELDOUT], "pairs": splits,
               "independent_unit": "named_assignment_world; source and donor remain paired",
               "scope": "new named worlds; country-attribute query holdout is for fitted probes/transport, not pretrained knowledge"}
    validate_tasks(payload)
    return payload


def validate_tasks(value):
    seen = set()
    for split, pairs in value["pairs"].items():
        if len(pairs) != COUNTS[split]:
            raise ValueError("Split count changed")
        for pair in pairs:
            if pair["world_id"] in seen:
                raise ValueError("World overlaps splits or examples")
            seen.add(pair["world_id"])
            for side in ("source", "donor"):
                row = pair[side]
                if row["mapping"][row["query"]] != row["intermediate"] or ANSWERS[row["attribute"]][row["intermediate"]] != row["answer"]:
                    raise ValueError("Task labels inconsistent")
                if split in ("train", "development") and row["composition_heldout"]:
                    raise ValueError("Composition holdout leaked into fit/development")


def calibration():
    import pyarrow as pa
    import pyarrow.ipc as ipc
    parent_path = ROOT/".pixi/jspace-calibration.json"
    parent = json.loads(parent_path.read_text())
    path = Path(parent["source_path"])
    if file_hash(path) != parent["source_sha256"]:
        raise ValueError("Cached WikiText changed")
    with pa.memory_map(str(path), "r") as source:
        table = ipc.open_stream(source).read_all()
    texts = table.column("text").to_pylist()
    # This is a new, fixed calibration sample, independent of task worlds.
    eligible = [(i, t.strip()) for i, t in enumerate(texts) if len(t.strip().split()) >= 80]
    eligible.sort(key=lambda row: hashlib.sha256(f"{SEED}:{row[0]}".encode()).hexdigest())
    selected, hashes = [], set()
    for index, text in eligible:
        h = hashlib.sha256(text.encode()).hexdigest()
        if h not in hashes:
            selected.append({"row_id": index, "text": text, "sha256": h})
            hashes.add(h)
        if len(selected) == 64:
            break
    return {"source": "Salesforce/wikitext wikitext-2-raw-v1 train", "source_path": str(path),
            "source_sha256": parent["source_sha256"], "selection_seed": SEED,
            "selection": "64 unique >=80-word rows ordered by SHA256(seed:row_id), truncated to 128 model tokens",
            "rows": selected}
