from experiments.structure_net.ncm_oracle_codec_semantic import metric, qualifies, select


def test_selection_ignores_outputs_and_input_order():
    rows = [{"id": str(i), "text": "irrelevant"} for i in range(252)]
    expected = [r["id"] for r in select(rows)]
    changed = [{**r, "text": "different", "score": i} for i, r in enumerate(reversed(rows))]
    assert [r["id"] for r in select(changed)] == expected
    assert len(set(expected)) == 64


def test_gate_keeps_all_cases_and_never_labels_oracle_as_task_accuracy():
    rows = [{"passed": i < 61, "valid": True, "tokens": 25, "text": "some text",
             "evaluations": [{"faithful": i < 61, "salient": True, "readable": True}]}
            for i in range(64)]
    result = metric(rows)
    assert qualifies(result)
    assert result["task_accuracy"] is None
    assert result["oracle_semantic_pass_rate"] == 61 / 64
    rows[-1] = {**rows[-1], "valid": False, "passed": None}
    result = metric(rows)
    assert not qualifies(result)
    assert result["cases"] == 64 and result["passed"] == 61
    assert result["oracle_semantic_pass_rate"] is None
    assert result["all_case_pass_upper_bound"] == 62 / 64
