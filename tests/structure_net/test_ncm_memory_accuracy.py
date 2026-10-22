import copy
import json

import pytest

from experiments.structure_net.ncm_memory_accuracy import CONTROLS, packet, parse, passes, wilson


def fixture():
    p = packet("case", "The meeting is cancelled. No new date is set.",
               [{"candidate_id": "a", "text": "The meeting was cancelled without a replacement date."}])
    g = {"candidate_id": "a", "faithful": True, "salient": True, "readable": True,
         "claims": [{"claim": "The meeting was cancelled without a replacement date.",
                     "verdict": "supported", "evidence_ids": ["L0"]}], "reason": "Faithful paraphrase."}
    return p, g


def test_faithful_paraphrase_passes_with_exact_reconstructed_evidence():
    p, g = fixture()
    result = parse(json.dumps({"evaluations": [g]}), p)[0]
    assert result["claims"][0]["source_quotes"] == [p["source_lines"][0]["text"]]
    assert passes(result, p["candidates"][0]["text"], 128)
    assert not passes(result, p["candidates"][0]["text"], 129)
    assert not passes(result, "", 0)


def test_tangential_but_supported_text_fails_whole_summary_task():
    p, g = fixture()
    g["salient"] = False
    parsed = parse(json.dumps({"evaluations": [g]}), p)[0]
    assert not passes(parsed, p["candidates"][0]["text"], 10)


def test_unsupported_claim_cannot_coexist_with_faithful_verdict():
    p, g = fixture()
    g["claims"][0].update(verdict="insufficient evidence", evidence_ids=[])
    with pytest.raises(ValueError, match="inconsistent"):
        parse(json.dumps({"evaluations": [g]}), p)
    g["faithful"] = False
    assert not passes(parse(json.dumps({"evaluations": [g]}), p)[0], "Monday meeting", 3)


def test_fabricated_evidence_or_candidate_binding_is_rejected():
    p, g = fixture()
    g["claims"][0]["evidence_ids"] = ["L999"]
    with pytest.raises(ValueError, match="evidence"):
        parse(json.dumps({"evaluations": [g]}), p)
    _, g = fixture()
    with pytest.raises(ValueError, match="candidate binding"):
        parse(json.dumps({"evaluations": [g, copy.deepcopy(g)]}), p)


def test_goal_and_confidence_are_distinct():
    assert 243/256 < .95 <= 244/256
    assert wilson(244, 256)[0] < .95
    assert wilson(256, 256)[0] > .95
    assert wilson(0, 256)[1] < .02
    assert len(CONTROLS) == 12 and all(len(x) == 3 for x in CONTROLS)


def test_conservative_unfaithful_verdict_remains_a_failure():
    from experiments.structure_net.ncm_memory_accuracy_v2 import parse as corrected_parse
    p, g = fixture()
    g.update(faithful=False, readable=False)
    result = corrected_parse(json.dumps({"evaluations": [g]}), p)[0]
    assert result["faithful"] is False
    assert result["claims"][0]["verdict"] == "supported"
    assert not passes(result, "meeting cancelled nonsense", 4)
    g["faithful"] = True
    g["claims"][0]["verdict"] = "insufficient evidence"
    with pytest.raises(ValueError, match="inconsistent"):
        corrected_parse(json.dumps({"evaluations": [g]}), p)
