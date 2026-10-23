from copy import deepcopy

from experiments.structure_net.ncm_converged_native import quality_gate, score_case, summarize_scores, run


def scores(count=64):
    return [{"passed": True, "valid": True, "text": "A faithful summary.", "tokens": 8,
             "evaluations": [{"faithful": True, "salient": True, "readable": True}]} for _ in range(count)]


def test_joint_gate_inclusive_and_invalids_remain_in_all_case_denominator():
    rows = scores()
    for i in range(3):
        rows[i]["passed"] = False
        rows[i]["evaluations"][0]["faithful"] = False
    result = summarize_scores(rows)
    assert result["task_accuracy"] == 61 / 64
    assert result["components"] == {"faithful": 61, "salient": 64, "readable": 64}
    assert quality_gate(result)
    invalid = deepcopy(rows)
    invalid[-1].update(valid=False, passed=None, evaluations=None)
    result = summarize_scores(invalid)
    assert result["task_accuracy"] is None
    assert result["all_case_pass_lower_bound"] == 60 / 64
    assert result["all_case_pass_upper_bound"] == 61 / 64
    assert not quality_gate(result)
    assert not quality_gate(summarize_scores(scores(63)))


def test_empty_and_overflow_are_reported_without_dropping_cases():
    rows = scores()
    rows[0].update(text="", tokens=0, passed=False)
    rows[1].update(tokens=129, passed=False)
    result = summarize_scores(rows)
    assert result["cases"] == 64
    assert result["empty"] == 1 and result["stored_overflow"] == 1
    assert result["passed"] == 62


def test_actual_whole_summary_decision_requires_all_components_and_storage_cap():
    class Encoding:
        def encode(self, text, disallowed_special):
            return text.split()
    case = {"id": "example", "source_id": "source", "text": "word " * 128, "native_output_ids": [1, 2]}
    packet = run.q.packet(case["id"], "Source passage.", [{"candidate_id": "summary", "text": case["text"]}])
    entry = {"role": "secondary", "origin": "first", "row": {"status": "valid", "attempts": [],
        "evaluations": [{"faithful": True, "salient": True, "readable": True}]}}
    assert score_case(case, packet, entry, Encoding())["passed"]
    case["text"] += "extra"
    assert not score_case(case, packet, entry, Encoding())["passed"]
    case["text"] = ""
    assert not score_case(case, packet, entry, Encoding())["passed"]
    case["text"] = "A readable summary."
    entry["row"]["evaluations"][0]["faithful"] = False
    assert not score_case(case, packet, entry, Encoding())["passed"]
