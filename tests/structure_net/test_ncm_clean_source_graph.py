from collections import Counter
from itertools import product
import random

import pytest

from experiments.structure_net import ncm_clean_source_graph as run


def row(name, text, *, family=None, blocked=False, band=0):
    sid = family or name
    return {"id": name, "source_id": sid, "group_id": sid, "source": text,
        "source_sha256": run.digest(text), "blocked": blocked, "candidate": not blocked,
        "bin": band, "spans": []}


def test_components_propagate_training_exclusion_through_alias_and_text_edges():
    graph = run.SourceGraph()
    words = [f"term{i}" for i in range(100)]
    graph.add(row("TRAIN", " ".join(words), family="sharegpt_a_0", blocked=True))
    graph.add(row("middle", "other material", family="sharegpt_a_2"))
    graph.add(row("alias", "other material", family="sharegpt_b_0"))
    graph.add(row("tail", "different passage", family="sharegpt_b_4"))
    graph.add(row("contained", " ".join(words[20:80]), family="different_family"))
    components = graph.components()
    assert len(components) == 1
    assert next(iter(components.values()))["blocked"]


def test_generic_short_text_is_not_inserted_by_caller_but_distinct_ultrachat_stays_distinct():
    graph = run.SourceGraph()
    graph.add(row("a", "first unique passage", family="ultrachat_11"))
    graph.add(row("b", "second unique passage", family="ultrachat_22"))
    assert len(graph.components()) == 2


def test_quota_matching_recovers_a_feasible_assignment_when_greedy_can_fail():
    components = {"flexible": {"blocked": False, "rows": [row("a", "a", band=0), row("b", "b", band=1)]},
        "only0": {"blocked": False, "rows": [row("c", "c", band=0)]}}
    chosen, _ = run.assign_components(components, {0: 1, 1: 1}, "fixed")
    assert chosen["flexible"]["bin"] == 1 and chosen["only0"]["bin"] == 0
    with pytest.raises(ValueError, match="insufficient"):
        run.assign_components(components, {0: 2, 1: 1}, "fixed")


def test_blocked_components_are_never_used_to_fill_a_quota():
    components = {"blocked": {"blocked": True, "rows": [row("a", "a", band=0)]}}
    with pytest.raises(ValueError, match="insufficient"):
        run.assign_components(components, {0: 1}, "fixed")


def test_fixed_splits_keep_full_counts_and_one_component_per_source():
    selected = {str(i): row(str(i), str(i), band=i % 2) for i in range(8)}
    parts = run.split_assignment(selected, {0: 2, 1: 2}, {0: 2, 1: 2}, "fixed")
    assert Counter({k: len(v) for k, v in parts.items()}) == {k: 2 for k in parts}
    assert len({r["source_component"] for values in parts.values() for r in values}) == 8
    assert run.split_assignment(dict(reversed(list(selected.items()))), {0: 2, 1: 2}, {0: 2, 1: 2}, "fixed") == parts


def test_quota_matching_agrees_with_exhaustive_feasibility_on_small_graphs():
    rng = random.Random(719)
    for _ in range(60):
        masks = [rng.randrange(1, 8) for _ in range(rng.randrange(3, 7))]
        quotas = {b: rng.randrange(1, 3) for b in range(3)}
        feasible = any(all(sum(a == b for a in assignment) == quotas[b] for b in quotas)
            for assignment in product(*[[-1] + [b for b in quotas if mask & (1 << b)] for mask in masks]))
        components = {str(i): {"blocked": False, "rows": [row(f"{i}-{b}", f"{i}-{b}", band=b)
            for b in quotas if mask & (1 << b)]} for i, mask in enumerate(masks)}
        if feasible:
            selected, _ = run.assign_components(components, quotas, "fixed")
            assert Counter(r["bin"] for r in selected.values()) == Counter(quotas)
        else:
            with pytest.raises(ValueError, match="insufficient"):
                run.assign_components(components, quotas, "fixed")
