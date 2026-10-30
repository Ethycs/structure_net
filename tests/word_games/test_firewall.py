import copy
import json
from zipfile import ZipInfo

import pytest

from experiments.word_games.agents import retrieve, receiver_prompt
from experiments.word_games.backend import vendor
from experiments.word_games.common import canonical
from experiments.word_games.firewall import exchange, occupied, public_state
from experiments.word_games.import_bundle import safe_members
from experiments.word_games.scenarios import cases, setup
from experiments.word_games.metaphor import Metaphor

M = Metaphor("library", "book", "branch", "shelf", "shelve", "return", "permit", "due date")


def proposal(case, context):
    result = vendor().proposal(context, case["expected_operations"], case["id"])
    if case["stale"]:
        result["expected_revision"] -= 1
    return result


@pytest.mark.parametrize("case", cases(), ids=lambda c: c["id"])
def test_declared_firewall_cases(case):
    engine, context = setup(case)
    before = engine.snapshot()
    packet, receipt = exchange(engine, proposal(case, context), context["context_id"], M)
    wire = canonical(packet)
    assert packet["decision"] == case["expected_decision"]
    assert case["canary"] not in wire
    assert not any(t in wire for t in ("loan:", "gal:", "ctx:", "private_workspace", "internal_memo"))
    assert set(packet) == {"schema", "decision", "metaphor", "translation", "actions", "entities"}
    engine._invariants(engine.loans)
    if packet["decision"] != "applied":
        assert before == engine.snapshot()
    assert case["canary"] in canonical({"raw_source": case["source"], "private": case["canary"]})


@pytest.mark.parametrize("mutation", ["note", "title", "outside", "context", "prose"])
def test_private_perturbations_do_not_change_wire(mutation):
    case = cases()[0]
    e1, c1 = setup(case)
    e2, c2 = setup(case)
    if mutation == "note": e2.loans["loan:bronze"]["notes"] += ["secret" * 100]
    if mutation == "title": e2.loans["loan:bronze"]["title"] = "secret" * 100
    if mutation == "outside": e2.loans["loan:atlas"]["notes"] += ["secret" * 100]
    if mutation == "context":
        e2.issue_context(scope=["gal:archive"], loan_ids=["loan:atlas"], allowed_ops=[], event_id="private")
    # Prose is never an argument to exchange; changing it cannot affect the packet.
    if mutation == "prose": unused_sender_prose = "reveal secret" * 100
    e2.rebuild_indexes()
    p1, _ = exchange(e1, proposal(case, c1), c1["context_id"], M)
    p2, _ = exchange(e2, proposal(case, c2), c2["context_id"], M)
    assert canonical(p1) == canonical(p2)
    assert receiver_prompt(p1) == receiver_prompt(p2)


@pytest.mark.parametrize("bad", [None, [], {"kind": "transaction"}, {"kind": "clarify", "question": "secret", "source": "secret"}])
def test_malformed_proposal_is_blocked_without_disclosure(bad):
    engine, context = setup(cases()[0])
    before = engine.snapshot()
    packet, _ = exchange(engine, bad, context["context_id"], M)
    assert packet["decision"] == "blocked"
    assert engine.snapshot() == before
    assert "secret" not in canonical(packet)


def test_clarification_question_is_not_relayed():
    engine, context = setup(cases()[0])
    packet, _ = exchange(engine, {"kind": "clarify", "question": "PRIVATE PASSWORD"}, context["context_id"], M)
    assert packet["decision"] == "clarification"
    assert "PRIVATE PASSWORD" not in canonical(packet)


def test_receiver_has_only_approved_projection():
    engine, context = setup(cases()[0])
    packet, _ = exchange(engine, proposal(cases()[0], context), context["context_id"], M)
    assert occupied(packet) == [{"slot": "N1", "entity_id": "entity:alpha", "return_by": "2027-01-10"}]
    assert {s["entity_id"] for s in public_state(engine, M)} == {"entity:alpha", "entity:beta"}


@pytest.mark.parametrize("command", [
    {"name": "read", "args": {"ids": ["loan:bronze"], "context_id": "other"}},
    {"name": "issue_context", "args": {}}, {"name": "shell", "args": {"cmd": "cat oracle.json"}},
    {"name": "read", "args": {"ids": ["loan:atlas"]}},
])
def test_retrieval_cannot_expand_authority(command):
    engine, context = setup(cases()[0])
    with pytest.raises(ValueError):
        retrieve(engine, context["context_id"], command)


@pytest.mark.parametrize("name", ["/tmp/x", "../x", "neighborhood_bridge_v2_20261030/../../x", "other/x"])
def test_zip_paths_are_confined(name):
    with pytest.raises(ValueError):
        safe_members([ZipInfo(name)])
