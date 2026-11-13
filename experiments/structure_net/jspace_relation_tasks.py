"""Fresh closed-world lookup tasks for J-space relation measurements.

Every answer is obtained from two explicit, randomized bijections:
person -> country -> colour. The labels audit task reasoning; they must not be
substituted for continuous J-lens scores or imposed as Boolean relations on
those scores. This generator makes no claim about neural polynomial closure.

World identity includes the actual names and both mapping tables, but excludes
the query and presentation. A world is emitted only once, so rewording it does
not manufacture additional independent observations. Anonymous permutation
shapes can recur (there are only 24 * 24 such shapes); their overlap is recorded.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from itertools import combinations
import json
import math
import random
import re
from typing import Any, Mapping


SCHEMA_VERSION = "jspace.relation_tasks.v1"
STUDY_ID = "JSPACE-RELATION-PROPAGATION-01"
COUNTRIES = ("China", "France", "Germany", "Russia")
COLOURS = ("red", "blue", "green", "yellow")
NAME_POOLS = {
    "train": ("Alice", "Ben", "Clara", "David", "Elena", "Felix", "Grace", "Henry", "Iris", "Jack", "Kira", "Leo"),
    "development": ("Maya", "Noah", "Olive", "Peter", "Quinn", "Ruby", "Simon", "Tara", "Uma", "Victor", "Wendy", "Xavier"),
    "evaluation": ("Ada", "Bruno", "Celia", "Dylan", "Esme", "Flora", "Gavin", "Hazel", "Ingrid", "Jonah", "Lena", "Micah"),
    "shakedown": ("Nia", "Owen", "Paula", "Reid", "Sana", "Theo", "Una", "Vera", "Will", "Yara", "Zane", "Amir"),
}
TEMPLATE_FAMILIES = {
    "train": ("train_sentences", "train_tables", "train_arrows", "train_compact"),
    "development": ("development_records", "development_checklist"),
    "evaluation": ("evaluation_json", "evaluation_reverse_roster"),
    "shakedown": ("shakedown_lookup",),
}


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rng(*parts: Any) -> random.Random:
    return random.Random(int(_digest(parts), 16))


def world_signature(person_to_country: Mapping[str, str], country_to_colour: Mapping[str, str]) -> str:
    """Hash actual mapping tables; query and wording intentionally do not enter."""
    return _digest({"person_to_country": dict(person_to_country), "country_to_colour": dict(country_to_colour)})


def _abstract_mapping_id(person_to_country: Mapping[str, str], country_to_colour: Mapping[str, str]) -> str:
    return _digest({
        "countries_for_alphabetically_sorted_people": [person_to_country[name] for name in sorted(person_to_country)],
        "colours_for_canonical_countries": [country_to_colour[country] for country in COUNTRIES],
    })


def _render(template: str, people: list[tuple[str, str]], colours: list[tuple[str, str]], query: str) -> tuple[str, str]:
    """Render only the two tables and a query; no answer/intermediate argument."""
    question = f"What colour does {query} receive? Reply with only the colour."
    if template == "train_sentences":
        body = "Use only these fictional assignments. Each person receives the colour of their assigned country.\n"
        body += "\n".join(f"{name} is assigned to {country}." for name, country in people)
        body += "\n" + "\n".join(f"{country} is assigned {colour}." for country, colour in colours)
    elif template == "train_tables":
        body = "Follow the two lookup tables: person to country, then country to colour.\nPerson | Country\n"
        body += "\n".join(f"{name} | {country}" for name, country in people)
        body += "\nCountry | Colour\n" + "\n".join(f"{country} | {colour}" for country, colour in colours)
    elif template == "train_arrows":
        body = "The arrows form person -> country -> colour chains. Use only these links.\nPeople:\n"
        body += "\n".join(f"{name} -> {country}" for name, country in people)
        body += "\nCountry colours:\n" + "\n".join(f"{country} -> {colour}" for country, colour in colours)
    elif template == "train_compact":
        body = "Each person's country determines their colour in this puzzle.\nPerson assignments: "
        body += "; ".join(f"{name}={country}" for name, country in people)
        body += ".\nCountry colours: " + "; ".join(f"{country}={colour}" for country, colour in colours) + "."
    elif template == "development_records":
        body = "Find the person's country record, then look up that country's colour record.\n"
        body += "\n".join(f"(person: {name}; country: {country})" for name, country in people)
        body += "\n" + "\n".join(f"(country: {country}; colour: {colour})" for country, colour in colours)
        question = f"Which colour follows from the two records for {query}? Return only the colour."
    elif template == "development_checklist":
        body = "A person's result is found by following their assigned country to its colour.\nAssignments:\n"
        body += "\n".join(f"- Assign {country} to {name}." for name, country in people)
        body += "\nColours:\n" + "\n".join(f"- Assign {colour} to {country}." for country, colour in colours)
        question = f"Give only the resulting colour for {query}."
    elif template == "evaluation_json":
        body = "Use these fictional dictionaries: look up the person, then look up the resulting country's colour.\n"
        body += json.dumps({"person_to_country": dict(people), "country_to_colour": dict(colours)}, ensure_ascii=False)
        question = f"Following both dictionaries, what colour belongs to {query}? Answer with only the colour."
    elif template == "evaluation_reverse_roster":
        body = "The first list names the person assigned to each country. The second gives each country's colour.\nCountry : assigned person\n"
        body += "\n".join(f"{country} : {name}" for name, country in people)
        body += "\nCountry : colour\n" + "\n".join(f"{country} : {colour}" for country, colour in colours)
        question = f"Look up {query} and report only their country's assigned colour."
    elif template == "shakedown_lookup":
        body = "Lookup check. Match a person to a country and then match that country to a colour.\nFirst lookup:\n"
        body += "\n".join(f"{name}: {country}" for name, country in people)
        body += "\nSecond lookup:\n" + "\n".join(f"{country}: {colour}" for country, colour in colours)
        question = f"For {query}, return only the colour from the second lookup."
    else:
        raise ValueError(f"Unknown template family: {template}")
    return body + "\n\n" + question, question


def _balanced_cells(seed: int, split: str, block: int) -> list[tuple[str, str]]:
    """All 16 country/answer pairs once; every four-cell prefix is balanced."""
    rng = _rng(seed, split, "country_answer_block", block)
    countries, colours, shifts = list(COUNTRIES), list(COLOURS), list(range(4))
    rng.shuffle(countries)
    rng.shuffle(colours)
    rng.shuffle(shifts)
    cells = []
    for shift in shifts:
        indices = list(range(4))
        rng.shuffle(indices)
        cells.extend((countries[(i + shift) % 4], colours[i]) for i in indices)
    return cells


def _make_split(split: str, count: int, seed: int, seen_worlds: set[str]) -> list[dict]:
    rows = []
    families = TEMPLATE_FAMILIES[split]
    for index in range(count):
        template = families[(index // 16) % len(families)]
        intermediate, answer = _balanced_cells(seed, split, index // 16)[index % 16]
        for attempt in range(1000):
            rng = _rng(seed, split, "world", index, attempt)
            names = rng.sample(NAME_POOLS[split], 4)
            assignments = list(COUNTRIES)
            rng.shuffle(assignments)
            person_map = dict(zip(names, assignments))
            other_countries = [c for c in COUNTRIES if c != intermediate]
            other_colours = [c for c in COLOURS if c != answer]
            rng.shuffle(other_colours)
            colour_map = {intermediate: answer, **dict(zip(other_countries, other_colours))}
            world_id = world_signature(person_map, colour_map)
            if world_id not in seen_worlds:
                break
        else:
            raise ValueError(f"Could not sample a fresh world for {split}/{index}")
        seen_worlds.add(world_id)
        query = next(name for name, country in person_map.items() if country == intermediate)
        swap_to = rng.choice(other_countries)
        people, colours = list(person_map.items()), list(colour_map.items())
        # Presentation order is randomized independently of the query's target.
        rng.shuffle(people)
        rng.shuffle(colours)
        prompt, question = _render(template, people, colours, query)
        rows.append({
            "case_id": f"{split}-{index:05d}-{world_id[:12]}",
            "split": split, "template_id": template, "world_id": world_id,
            "abstract_mapping_id": _abstract_mapping_id(person_map, colour_map),
            "prompt": prompt, "question": question, "query_person": query,
            "intermediate": intermediate, "answer": answer,
            "swap_to": swap_to, "swap_answer": colour_map[swap_to],
            "person_to_country": person_map, "country_to_colour": colour_map,
            "person_display_order": [name for name, _ in people],
            "country_display_order": [country for country, _ in colours],
            "generation_seed": seed, "quality_data": split != "shakedown",
        })
    # Avoid returning blocks of templates or a visible target-label schedule.
    return sorted(rows, key=lambda row: _digest([seed, "case_order", row["case_id"]]))


def _overlaps(sets: Mapping[str, set[str]]) -> dict[str, int]:
    return {f"{first}:{second}": len(sets[first] & sets[second])
            for first, second in combinations(sets, 2)}


def build_dataset(
    seed: int = 7, train_count: int = 512, development_count: int = 128,
    evaluation_count: int = 128, shakedown_count: int = 16,
) -> dict[str, Any]:
    """Return a serializable manifest and ``splits[split_name]`` case lists.

    Whole named worlds, name pools, and template IDs are disjoint across splits.
    The intermediate and final answer are exactly balanced at default sizes,
    including all 16 country/answer combinations per template. Shakedown uses
    its own seed, names, and wording; it is never quality/evaluation data.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    counts = dict(train=train_count, development=development_count,
                  evaluation=evaluation_count, shakedown=shakedown_count)
    for split, count in counts.items():
        capacity = math.comb(len(NAME_POOLS[split]), 4) * math.factorial(4) ** 2
        if type(count) is not int or not 0 <= count <= capacity:
            raise ValueError(f"{split} count must be an integer between 0 and {capacity}")
    seeds = {split: seed + 1_000_003 if split == "shakedown" else seed for split in counts}
    seen_worlds: set[str] = set()
    splits = {split: _make_split(split, count, seeds[split], seen_worlds) for split, count in counts.items()}
    abstract_ids = {split: {row["abstract_mapping_id"] for row in rows} for split, rows in splits.items()}
    manifest = {
        "seed": seed, "generation_seeds": seeds, "counts": counts,
        "countries": list(COUNTRIES), "colours": list(COLOURS),
        "name_pools": {key: list(value) for key, value in NAME_POOLS.items()},
        "template_families": {key: list(value) for key, value in TEMPLATE_FAMILIES.items()},
        "independent_unit": "named_mapping_pair_with_one_query_and_one_presentation",
        "world_signature_definition": "SHA256 of canonical JSON containing both actual mapping tables; query and template excluded",
        "anonymous_mapping_shape_count": math.factorial(4) ** 2,
        "anonymous_mapping_shapes_can_recur": True,
        "anonymous_shape_unique_counts": {split: len(ids) for split, ids in abstract_ids.items()},
        "anonymous_shape_overlap_counts": _overlaps(abstract_ids),
        "generalization_scope": "new named worlds, disjoint name pools and template families; not globally disjoint anonymous permutation shapes",
        "observables_policy": "Task labels are audit metadata only. Fit relations to actual continuous J-lens measurements; do not impose one-hot or Boolean constraints from these tables.",
        "shakedown_policy": "Separate seed, name pool and template; systems checks only, never quality data",
        "split_sha256": {split: _digest(rows) for split, rows in splits.items()},
        "balance": {split: {
            "intermediate": dict(sorted(Counter(row["intermediate"] for row in rows).items())),
            "answer": dict(sorted(Counter(row["answer"] for row in rows).items())),
            "template": dict(sorted(Counter(row["template_id"] for row in rows).items())),
        } for split, rows in splits.items()},
        "cases_sha256": _digest(splits),
    }
    dataset = {"schema_version": SCHEMA_VERSION, "study_id": STUDY_ID, "manifest": manifest, "splits": splits}
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: Mapping[str, Any]) -> None:
    """Audit labels, bijections, world identity, split boundaries and content hash."""
    if dataset["schema_version"] != SCHEMA_VERSION or dataset["study_id"] != STUDY_ID:
        raise ValueError("Unknown dataset schema/study")
    manifest, splits = dataset["manifest"], dataset["splits"]
    if set(splits) != set(NAME_POOLS):
        raise ValueError("Unexpected split names")
    all_worlds, all_ids, all_prompts = set(), set(), set()
    for split, rows in splits.items():
        if len(rows) != manifest["counts"][split]:
            raise ValueError("Split count mismatch")
        for row in rows:
            pmap, cmap = row["person_to_country"], row["country_to_colour"]
            if (len(pmap) != 4 or not set(pmap) <= set(NAME_POOLS[split])
                    or set(pmap.values()) != set(COUNTRIES) or set(cmap) != set(COUNTRIES)
                    or set(cmap.values()) != set(COLOURS)):
                raise ValueError("World must contain two bijections using its split's name pool")
            if row["split"] != split or row["template_id"] not in TEMPLATE_FAMILIES[split]:
                raise ValueError("Split/template mismatch")
            if row["quality_data"] != (split != "shakedown"):
                raise ValueError("Shakedown must not be quality data")
            if row["query_person"] not in pmap or pmap[row["query_person"]] != row["intermediate"]:
                raise ValueError("Incorrect intermediate lookup")
            if cmap[row["intermediate"]] != row["answer"]:
                raise ValueError("Incorrect composed answer")
            if (row["swap_to"] not in cmap or row["swap_to"] == row["intermediate"]
                    or row["swap_answer"] != cmap[row["swap_to"]] or row["swap_answer"] == row["answer"]):
                raise ValueError("Incorrect counterfactual lookup")
            if row["world_id"] != world_signature(pmap, cmap):
                raise ValueError("World signature mismatch")
            if row["abstract_mapping_id"] != _abstract_mapping_id(pmap, cmap):
                raise ValueError("Anonymous mapping signature mismatch")
            if (row["world_id"] in all_worlds or row["case_id"] in all_ids or row["prompt"] in all_prompts):
                raise ValueError("Repeated world, case identifier or prompt")
            all_worlds.add(row["world_id"])
            all_ids.add(row["case_id"])
            all_prompts.add(row["prompt"])
            if (set(row["person_display_order"]) != set(pmap)
                    or len(row["person_display_order"]) != 4
                    or set(row["country_display_order"]) != set(cmap)
                    or len(row["country_display_order"]) != 4):
                raise ValueError("Invalid table presentation order")
            expected_prompt, expected_question = _render(
                row["template_id"], [(p, pmap[p]) for p in row["person_display_order"]],
                [(c, cmap[c]) for c in row["country_display_order"]], row["query_person"])
            if (row["prompt"], row["question"]) != (expected_prompt, expected_question):
                raise ValueError("Rendered prompt disagrees with the audited mapping tables")
            if any(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", row["question"], re.IGNORECASE)
                   for value in (*COUNTRIES, *COLOURS)):
                raise ValueError("Question must not disclose a country or colour label")
        if manifest["split_sha256"][split] != _digest(rows):
            raise ValueError("Split content hash mismatch")
    if manifest["cases_sha256"] != _digest(splits):
        raise ValueError("Dataset content hash mismatch")
