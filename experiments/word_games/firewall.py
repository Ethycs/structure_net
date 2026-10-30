"""Explicit declassification policy and deterministic receiver rendering.

Language-model prose never enters this serializer. This constrains the channel;
it does not eliminate covert channels in permitted choices or timing.
"""
from dataclasses import dataclass
from datetime import date
import copy

from .backend import vendor
from .common import canonical


@dataclass(frozen=True)
class DisclosurePolicy:
    seeds: tuple = (("loan:bronze", "entity:alpha", "Alpha"),
                    ("loan:clock", "entity:beta", "Beta"))
    plots: tuple = (("gal:north", "location:north", "North"),)
    notes: tuple = (("Condition checked", "condition_checked"),)
    operations: tuple = ("reserve", "transfer", "cancel", "annotate")

    def __post_init__(self):
        for values in (self.seeds, self.plots):
            if len({x[0] for x in values}) != len(values) or len({x[1] for x in values}) != len(values):
                raise ValueError("unique source and target identities required")


POLICY = DisclosurePolicy()


def allowed_actions(proposal, policy=POLICY):
    backend = vendor()
    if not isinstance(proposal, dict):
        raise ValueError("invalid proposal")
    if proposal.get("kind") == "clarify":
        if set(proposal) != {"kind", "question"} or not isinstance(proposal["question"], str):
            raise ValueError("invalid clarification")
        return []  # Free-form question is retained host-side; receiver gets a fixed label.
    ops = proposal.get("operations")
    if not isinstance(ops, list) or not 1 <= len(ops) <= 20:
        raise ValueError("invalid operations")
    seeds = {source: (target, label) for source, target, label in policy.seeds}
    plots = {source: (target, label) for source, target, label in policy.plots}
    notes = dict(policy.notes)
    actions = []
    for op in ops:
        if not isinstance(op, dict) or op.get("op") not in policy.operations:
            raise ValueError("operation not disclosed")
        typ = op["op"]
        if set(op) != backend.Engine.OP_FIELDS[typ] or op.get("loan_id") not in seeds:
            raise ValueError("fields or identity not disclosed")
        target, label = seeds[op["loan_id"]]
        action = {"entity_id": target}
        if typ in ("reserve", "transfer"):
            destination = op["gallery_id"] if typ == "reserve" else op["to_gallery_id"]
            if destination not in plots or op["slot"] not in ("N1", "N2"):
                raise ValueError("destination not disclosed")
            due = op["due"]
            if not isinstance(due, str) or date.fromisoformat(due).isoformat() != due:
                raise ValueError("invalid date")
            action.update(action="place" if typ == "reserve" else "move", location_id=plots[destination][0],
                          slot=op["slot"], return_by=due)
        elif typ == "cancel":
            action["action"] = "return_home"
        else:
            if not isinstance(op["text"], str) or op["text"] not in notes:
                raise ValueError("free text not disclosed")
            action.update(action="record_condition", note_code=notes[op["text"]])
        actions.append(action)
    return actions


def public_state(engine, metaphor, policy=POLICY):
    plots = dict((s, t) for s, t, _ in policy.plots)
    result = []
    for source, target, label in sorted(policy.seeds):
        record = engine.loans.get(source)
        if record is None or engine.location(record) not in plots:
            continue
        planting = record["reservation"]
        result.append({"entity_id": target, "label": f"{metaphor.entity} {label}", "location_id": plots[engine.location(record)],
                       "placement": None if planting is None else {
                           "slot": planting["slot"], "return_by": planting["due"]}})
    return result


def render(actions, decision, metaphor):
    if decision == "blocked":
        return f"The requested {metaphor.name} change was blocked. The authorized state is unchanged."
    if decision == "clarification":
        return f"The requested {metaphor.name} change requires clarification. No change was made."
    phrases = []
    for action in actions:
        seed = action["entity_id"]
        if action["action"] in ("place", "move"):
            verb = metaphor.place_verb if action["action"] == "place" else "move"
            phrases.append(f"Completed request to {verb} {metaphor.entity} {seed} at {metaphor.location} {action['location_id']}, {metaphor.slot} {action['slot']}; {metaphor.deadline}: {action['return_by']}.")
        elif action["action"] == "return_home":
            phrases.append(f"Completed request to {metaphor.return_verb} {metaphor.entity} {seed}; it occupies no {metaphor.slot}.")
        else:
            phrases.append(f"Recorded the approved condition check for {seed}.")
    return " ".join(phrases)


def exchange(engine, proposal, context_id, metaphor, *, policy=POLICY):
    """Return separate receiver packet and private receipt; never relay model prose."""
    before, receipts = engine.snapshot(), copy.deepcopy(engine.receipts)
    actions = []
    try:
        actions = allowed_actions(proposal, policy)
        receipt = engine.apply(proposal, expected_context_id=context_id)
        decision = "applied" if receipt["status"] == "committed" else "clarification"
    except (ValueError, TypeError, KeyError) as exc:
        receipt = {"status": "rejected", "reason": str(exc)}
        decision, actions = "blocked", []
    if decision != "applied" and (engine.snapshot() != before or engine.receipts != receipts):
        raise RuntimeError("rejected transaction modified private state")
    from dataclasses import asdict
    packet = {"schema": "word-games.receiver.v1", "decision": decision,
              "metaphor": asdict(metaphor), "translation": render(actions, decision, metaphor), "actions": actions,
              "entities": public_state(engine, metaphor, policy)}
    # Serialization is a host-created object, with no copied context, request or prose.
    canonical(packet)
    return packet, receipt


def occupied(packet):
    return sorted([{"slot": s["placement"]["slot"], "entity_id": s["entity_id"],
                    "return_by": s["placement"]["return_by"]}
                   for s in packet["entities"] if s["placement"]], key=lambda x: (x["slot"], x["entity_id"]))
