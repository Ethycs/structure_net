import copy
import json

import pytest

from experiments.structure_net import ncm_targeted_label_recovery as study


def fixture():
    cases = [{"id": str(i), "source_id": "s"+str(i), "group_id": "g"+str(i),
        "source_sha256": "h"+str(i), "source": "source "+str(i), "spans": [i], "bin": i} for i in range(4)]
    failures = [{"label": {**r, "text": "draft"}, "score": {"id": r["id"], "text": "draft", "valid": True, "passed": False},
        "format_capacity_fit": True} for r in cases]
    return cases, failures


def test_only_saved_rejections_are_selected_and_pilot_covers_all_four_bands():
    cases, failures = fixture()
    failures[-1]["score"]["passed"] = True
    failures[-1]["format_capacity_fit"] = False
    selected = study.select_targets(cases, failures)
    assert selected == cases
    assert study.pilot_cases(selected) == cases


@pytest.mark.parametrize("change", ["accepted", "invalid", "text", "source", "duplicate", "unknown_verdict"])
def test_changed_or_previously_accepted_labels_cannot_enter_targeted_pass(change):
    cases, failures = fixture()
    if change == "accepted": failures[0]["score"]["passed"] = True
    elif change == "invalid": failures[0]["score"]["valid"] = False
    elif change == "text": failures[0]["score"]["text"] = "different"
    elif change == "source": failures[0]["label"]["source"] = "different"
    elif change == "duplicate": failures.append(copy.deepcopy(failures[0]))
    elif change == "unknown_verdict": failures[0]["score"]["passed"] = None
    with pytest.raises(ValueError): study.select_targets(cases, failures)


def test_strict_four_of_four_gate_and_existing_input_roots_are_preserved():
    result = {"endpoint": {"cases": 4, "valid": 4, "passed": 4}, "accepted_labels": 4, "pilot_gate": True}
    assert study.pilot_gate(result)
    result["endpoint"]["passed"] = 3
    assert not study.pilot_gate(result)
    assert study.root(True, "draft") == study._original_root(False, "draft")
    assert study.root(True, "critique") == study._original_root(False, "critique")
    assert study.root(True, "revision") != study.root(False, "revision")
    with study.limited_pool(max_workers=4) as pool: assert pool._max_workers == 2


def test_exhaustion_watchdog_requires_three_invalid_attempts_of_one_request(tmp_path):
    path = tmp_path / "requests" / "one" / "attempts";path.mkdir(parents=True)
    for i in range(2): (path / f"case_{i}.json").write_text(json.dumps({"valid": False}))
    assert study.exhausted(tmp_path) == []
    (path / "case_2.json").write_text(json.dumps({"valid": False}))
    assert len(study.exhausted(tmp_path)) == 3
    (path / "case_2.json").write_text(json.dumps({"valid": True}))
    assert study.exhausted(tmp_path) == []
