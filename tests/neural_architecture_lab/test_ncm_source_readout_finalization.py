from experiments.neural_architecture_lab.finalize_ncm_source_readout import passed_gate


def test_task_gate_requires_the_correct_denominator_and_valid_endpoint():
    assert passed_gate({"status": "completed", "cases": 64, "passed": 61})
    assert not passed_gate({"status": "completed", "cases": 64, "passed": 60})
    assert not passed_gate({"status": "completed", "cases": 20, "passed": 20})
    assert not passed_gate({"status": "invalid_judgments", "cases": 64, "passed": 61})
    assert not passed_gate(None)
