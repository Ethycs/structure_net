from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.structure_net import ncm_clean_exchange_partitions as run
from experiments.neural_architecture_lab import finalize_ncm_clean_exchange_partitions as audit


def fixture():
    samples = []
    for name in ("a", "b"):
        session = [{"role": "assistant", "content": "Thanks!"}]
        for i in range(4):
            session += [{"role": "user", "content": f"Question {name}{i}"},
                {"role": "assistant", "content": " ".join(f"{name}{i}_{j}" for j in range(400))}]
        samples.append({"haystack_session_ids": ["ultrachat_" + name], "haystack_sessions": [session]})
    enc = run.CountingEncoding(SimpleNamespace(encode=lambda text, **kwargs: text.split()))
    turns, sessions = run.inventory(samples, enc)
    known = run.fragments.windows("ultrachat_a", samples[0]["haystack_sessions"][0], enc)[0]
    return samples, enc, turns, sessions, known


def test_pool_excludes_known_families_without_joining_unrelated_greetings():
    samples, enc, turns, sessions, known = fixture()
    nodes, _ = run.pool(samples, [known], [], enc, turns, sessions)
    graph = run.CompactSourceGraph()
    for row in nodes:
        graph.add(row)
    components = graph.components()
    free = [r for value in components.values() if not value["blocked"] for r in value["rows"]]
    assert free and all(r["source_id"] == "ultrachat_b" for r in free)
    greeting_sha = run.text_hash("Thanks!")
    assert all(greeting_sha not in {s["turn_text_sha256"] for s in row["spans"]} for row in nodes)
    row = next(r for r in free if r["source"].startswith("Thanks!"))
    exported = run.export(row, "component", "development_long")
    assert exported["spans"][0]["turn_text_sha256"] == greeting_sha
    assert exported["source"].startswith("Thanks!")
    assert run.selected_checks([known], [exported], samples, turns) == {exported["id"]: exported["source_sha256"]}


def test_origin_audit_checks_every_literal_turn_and_exchange_boundary():
    samples, enc, turns, sessions, _ = fixture()
    nodes, _ = run.pool(samples, [], [], enc, turns, sessions)
    row = next(r for r in nodes if r["source"].startswith("Thanks!") and r["tokens"] >= 512)
    origin = run.choose_origin(list(row["origins"].values()))
    inv = {"turns": turns, "sessions": sessions}
    audit.check_origin(origin, inv, len(row["source"]))
    broken = deepcopy(origin)
    broken["end_turn"] = 4
    with pytest.raises(ValueError, match="exchange boundaries"):
        audit.check_origin(broken, inv, len(row["source"]))
    broken = deepcopy(origin)
    broken["spans"][0]["end_char"] += 1
    with pytest.raises(ValueError, match="complete public spans"):
        audit.check_origin(broken, inv, len(row["source"]))


def test_public_origin_choice_is_stable_across_input_order():
    values = [{"source_id": "a", "kind": "whole_turn"}, {"source_id": "b", "kind": "exchange_window"}]
    assert run.choose_origin(values) == run.choose_origin(list(reversed(values)))


def test_metadata_replay_keeps_full_short_and_long_cohort_allocation(monkeypatch):
    monkeypatch.setattr(run, "FINAL", {0: 1, 1: 1})
    monkeypatch.setattr(run, "DEVELOPMENT", {0: 1, 1: 1})
    index = []
    for i in range(4):
        sha = run.text_hash(str(i))
        index.append({"id": run.core.digest(run.NAMESPACE + ":node:" + sha), "source_sha256": sha,
            "source_component": sha, "tokens": 250 if i % 2 == 0 else 600, "bin": i % 2,
            "blocked": False, "candidate": True})
    parts, _ = audit.replay_assignment(index)
    assert {k: len(v) for k, v in parts.items()} == {k: 1 for k in parts}
    index[0]["source_component"] = "changed"
    with pytest.raises(ValueError, match="component identity"):
        audit.replay_assignment(index)
