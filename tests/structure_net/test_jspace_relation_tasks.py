from collections import Counter
from copy import deepcopy
import json
import random
import re

import pytest

from experiments.structure_net.jspace_relation_tasks import (
    COLOURS, COUNTRIES, NAME_POOLS, TEMPLATE_FAMILIES, build_dataset,
    validate_dataset, world_signature,
)


@pytest.fixture(scope="module")
def dataset():
    return build_dataset()


def test_default_dataset_is_deterministic_json_and_does_not_mutate_global_rng(dataset):
    before = random.getstate()
    duplicate = build_dataset()
    assert random.getstate() == before
    assert duplicate == dataset
    assert json.loads(json.dumps(dataset)) == dataset
    assert dataset["manifest"]["counts"] == {
        "train": 512, "development": 128, "evaluation": 128, "shakedown": 16,
    }
    changed = build_dataset(seed=8)
    assert changed["manifest"]["cases_sha256"] != dataset["manifest"]["cases_sha256"]
    assert changed["splits"]["evaluation"] != dataset["splits"]["evaluation"]


def test_splits_have_unique_named_worlds_queries_and_disjoint_names_templates(dataset):
    worlds, identifiers, prompts, previous_names, previous_templates = set(), set(), set(), set(), set()
    for split, rows in dataset["splits"].items():
        names = {name for row in rows for name in row["person_to_country"]}
        templates = {row["template_id"] for row in rows}
        assert names == set(NAME_POOLS[split])
        assert templates == set(TEMPLATE_FAMILIES[split])
        assert not names & previous_names
        assert not templates & previous_templates
        previous_names |= names
        previous_templates |= templates
        for row in rows:
            assert row["world_id"] not in worlds
            assert row["case_id"] not in identifiers
            assert row["prompt"] not in prompts
            worlds.add(row["world_id"])
            identifiers.add(row["case_id"])
            prompts.add(row["prompt"])
            assert row["world_id"] == world_signature(row["person_to_country"], row["country_to_colour"])
    assert len(worlds) == 784
    assert dataset["manifest"]["anonymous_mapping_shape_count"] == 576
    assert dataset["manifest"]["anonymous_mapping_shapes_can_recur"] is True
    assert sum(dataset["manifest"]["anonymous_shape_overlap_counts"].values()) > 0


def test_two_hop_and_counterfactual_answers_follow_the_supplied_bijections(dataset):
    for rows in dataset["splits"].values():
        for row in rows:
            people, colours = row["person_to_country"], row["country_to_colour"]
            assert len(people) == 4
            assert set(people.values()) == set(COUNTRIES)
            assert set(colours) == set(COUNTRIES)
            assert set(colours.values()) == set(COLOURS)
            assert row["intermediate"] == people[row["query_person"]]
            assert row["answer"] == colours[people[row["query_person"]]]
            assert row["swap_to"] in COUNTRIES
            assert row["swap_to"] != row["intermediate"]
            assert row["swap_answer"] == colours[row["swap_to"]]
            assert row["swap_answer"] != row["answer"]


def test_question_contains_no_target_or_intermediate_copy_but_body_contains_lookup_data(dataset):
    for rows in dataset["splits"].values():
        for row in rows:
            assert row["query_person"] in row["question"]
            assert row["prompt"].endswith("\n\n" + row["question"])
            body = row["prompt"][:-len(row["question"])]
            for label in (*COUNTRIES, *COLOURS):
                pattern = r"(?<!\w)" + re.escape(label) + r"(?!\w)"
                assert re.search(pattern, row["question"], re.IGNORECASE) is None
                assert re.search(pattern, body) is not None
            assert all(name in body for name in row["person_to_country"])


def test_intermediate_answer_and_joint_pairs_are_balanced_within_every_template(dataset):
    for split, rows in dataset["splits"].items():
        for template in TEMPLATE_FAMILIES[split]:
            selected = [row for row in rows if row["template_id"] == template]
            assert len(selected) % 16 == 0
            countries = Counter(row["intermediate"] for row in selected)
            colours = Counter(row["answer"] for row in selected)
            pairs = Counter((row["intermediate"], row["answer"]) for row in selected)
            assert set(countries.values()) == {len(selected) // 4}
            assert set(colours.values()) == {len(selected) // 4}
            assert len(pairs) == 16
            assert set(pairs.values()) == {len(selected) // 16}


def test_small_nonmultiple_counts_have_balanced_marginals_and_independent_split_sampling():
    kwargs = dict(train_count=5, development_count=7, evaluation_count=9, shakedown_count=3)
    first = build_dataset(**kwargs)
    second = build_dataset(**{**kwargs, "train_count": 29})
    for split in ("development", "evaluation", "shakedown"):
        assert first["splits"][split] == second["splits"][split]
    for rows in first["splits"].values():
        for key, values in (("intermediate", COUNTRIES), ("answer", COLOURS)):
            counts = Counter(row[key] for row in rows)
            numbers = [counts[value] for value in values]
            assert max(numbers) - min(numbers) <= 1


def test_shakedown_is_separate_seed_and_never_quality_data(dataset):
    seeds = dataset["manifest"]["generation_seeds"]
    assert seeds["train"] == seeds["development"] == seeds["evaluation"] == 7
    assert seeds["shakedown"] != 7
    for split, rows in dataset["splits"].items():
        assert all(row["generation_seed"] == seeds[split] for row in rows)
        assert all(row["quality_data"] == (split != "shakedown") for row in rows)


def test_world_identity_ignores_table_presentation_and_changes_when_relations_change(dataset):
    row = dataset["splits"]["train"][0]
    pmap, cmap = row["person_to_country"], row["country_to_colour"]
    assert world_signature(dict(reversed(list(pmap.items()))), dict(reversed(list(cmap.items())))) == row["world_id"]
    altered = dict(cmap)
    altered["China"], altered["France"] = altered["France"], altered["China"]
    assert world_signature(pmap, altered) != row["world_id"]


@pytest.mark.parametrize("mutation,match", [
    ("answer", "composed answer"), ("swap_answer", "counterfactual"),
    ("prompt", "Rendered prompt"), ("world_id", "World signature"),
    ("template_id", "Split/template"),
])
def test_audit_rejects_corrupted_labels_and_rendering(mutation, match):
    data = build_dataset(train_count=2, development_count=1, evaluation_count=1, shakedown_count=1)
    row = data["splits"]["train"][0]
    if mutation == "answer":
        row["answer"] = row["swap_answer"]
    elif mutation == "swap_answer":
        row["swap_answer"] = row["answer"]
    elif mutation == "prompt":
        row["prompt"] += " The answer is " + row["answer"]
    elif mutation == "world_id":
        row["world_id"] = "false-signature"
    else:
        row["template_id"] = TEMPLATE_FAMILIES["evaluation"][0]
    with pytest.raises(ValueError, match=match):
        validate_dataset(data)


def test_repeating_a_world_under_another_case_id_is_not_an_independent_example():
    data = build_dataset(train_count=2, development_count=0, evaluation_count=0, shakedown_count=0)
    copied = deepcopy(data["splits"]["train"][0])
    copied["case_id"] = "different-id-same-maps"
    data["splits"]["train"][1] = copied
    with pytest.raises(ValueError, match="Repeated world"):
        validate_dataset(data)


@pytest.mark.parametrize("kwargs", [
    {"seed": True}, {"seed": 1.5}, {"train_count": -1}, {"evaluation_count": 2.5},
    {"development_count": True}, {"shakedown_count": 300_000},
])
def test_invalid_generation_arguments_fail_before_sampling(kwargs):
    with pytest.raises(ValueError):
        build_dataset(**kwargs)
