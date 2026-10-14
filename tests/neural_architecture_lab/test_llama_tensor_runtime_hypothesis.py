import pytest
from neural_architecture_lab.llama_tensor_runtime_hypothesis import classification, RuntimeEvidence


def test_joint_task_and_both_native_speed_endpoints():
    assert classification(19, 19, 19, 4, 2, 2) == "proxy_quality_and_double_native_speed_supported"
    assert classification(18, 19, 18, 10, 10, 10) == "runtime_final_quality_failed"
    assert classification(19, 20, 19, 10, 10, 10) == "runtime_final_quality_failed"
    assert classification(20, 20, 19, 4, 2.1, 1.9) == "prototype_speed_recovered_native_target_unmet"
    assert classification(20, 19, 19, 1.9, 1.1, 1.1) == "runtime_speed_target_unmet"


def test_strict_counts_and_finite_measurements():
    with pytest.raises(ValueError):
        classification(21, 20, 20, 2, 2, 2)
    with pytest.raises(ValueError):
        classification(20, 20, 20, float("nan"), 2, 2)
    e = RuntimeEvidence("campaign", "a"*64, "report", "b"*64, "prototype_speed_recovered_native_target_unmet", 396)
    assert e.record()["confirmed"] is False


def test_quality_uses_worst_distinct_repeated_output_and_checks_source(tmp_path):
    import json
    from experiments.neural_architecture_lab.finalize_llama_tensor_runtime import review_counts
    from experiments.structure_net.llama_tensor_runtime import digest
    fragment = {"id":"F01", "reference_fact":"Final owner Maya.", "source":"Final owner Maya."}
    examples = {"case":{"fragments":[fragment]}}
    rows = [{"arm":"graph", "id":"case", "text":t} for t in ("Maya", "Omar")]
    reviews = [{"arm":"graph", "id":"case", "text_sha256":digest(r["text"]),
                "source_sha256":digest([fragment]), "source_quotes":[{"id":"F01","quote":"Final owner Maya."}],
                "passed":r["text"] == "Maya", "rationale":"Compare owner with final source decision."} for r in rows]
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps({"items":reviews}))
    assert review_counts(rows,examples,path) == {"graph":0}
    reviews[0]["source_sha256"] = "changed"
    path.write_text(json.dumps({"items":reviews}))
    with pytest.raises(ValueError,match="source identity"):
        review_counts(rows,examples,path)
