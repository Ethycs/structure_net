from copy import deepcopy

import pytest

from experiments.structure_net import ncm_additional_training_sources as run


def fixture(kind="whole_turn"):
    session = [{"role": "user", "content": "Alpha " * 200}, {"role": "assistant", "content": "Beta " * 400}]
    end = 1 if kind == "whole_turn" else 2
    text = "\n".join(t["content"] for t in session[:end])
    encoding = run.storage_encoding()
    count = len(encoding.encode(text, disallowed_special=()))
    spans = []
    for turn in session[:end]:
        sha = run.text_hash(turn["content"])
        spans.append({"source_id": "sid", "role": turn["role"], "turn_id": "public:" + run.text_hash("sid|" + turn["role"] + "|" + sha),
            "turn_text_sha256": sha, "span_text_sha256": sha, "start_char": 0, "end_char": len(turn["content"])})
    origin = {"source_id": "sid", "start_turn": 0, "end_turn": end, "kind": kind, "spans": spans, "session_sha256": "session"}
    row = {"id": "id", "source_component": "component", "source_sha256": run.text_hash(text),
        "source_characters": len(text), "tokens": count, "bin": run.reservation.graph.bucket(count), "public_origins": [origin]}
    return row, origin, session, encoding


@pytest.mark.parametrize("kind", ("whole_turn", "exchange_window"))
def test_literal_full_training_source_and_provenance(kind):
    row, origin, session, encoding = fixture(kind)
    value = run.materialize(row, origin, session, encoding)
    assert value["source"] == "\n".join(t["content"] for t in session[:origin["end_turn"]])
    assert not value["source_truncated"] and value["source_component"] == value["group_id"] == "component"
    assert value["spans"] == origin["spans"] and value["split"] == "additional_train"


def test_partial_exchange_is_rejected_before_any_source_replacement():
    row, origin, session, encoding = fixture("exchange_window")
    origin["end_turn"] = 1
    with pytest.raises(ValueError, match="exchange boundaries"):
        run.materialize(row, origin, session, encoding)


@pytest.mark.parametrize("field", ("source_sha256", "source_characters", "tokens", "bin"))
def test_identity_or_length_mismatch_is_not_repaired(field):
    row, origin, session, encoding = fixture()
    row[field] = "wrong" if field == "source_sha256" else row[field] + 1
    with pytest.raises(ValueError, match="exact public source"):
        run.materialize(row, origin, session, encoding)


def test_original_public_span_metadata_cannot_be_changed():
    row, origin, session, encoding = fixture()
    origin = deepcopy(origin)
    origin["spans"][0]["end_char"] -= 1
    with pytest.raises(ValueError, match="exact public source"):
        run.materialize(row, origin, session, encoding)
