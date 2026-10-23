import pytest

from experiments.neural_architecture_lab.finalize_ncm_native_query_controls import joint_teacher_gate


def test_teacher_task_point_estimate_cannot_rescue_capacity_gate():
    labels = [{"text": "A supported summary.", "claims": ["supported"],
               "stored_tokens": 80, "native_tokens_including_eos": 81} for _ in range(128)]
    assert joint_teacher_gate(123, 128, 128, labels) == (True, True)
    labels[0]["native_tokens_including_eos"] = 129
    assert joint_teacher_gate(123, 128, 128, labels) == (False, False)
    labels[0]["native_tokens_including_eos"] = 128
    assert joint_teacher_gate(121, 128, 128, labels) == (True, False)
    assert joint_teacher_gate(123, 127, 128, labels) == (True, False)
    with pytest.raises(ValueError):
        joint_teacher_gate(10, 10, 10, labels[:10])
