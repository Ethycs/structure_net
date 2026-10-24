from dataclasses import replace

import pytest

from neural_architecture_lab.ncm_native_final_hypothesis import NativeFinalEvidence


def test_final_metadata_cannot_turn_partial_or_failed_grade_into_success():
    value = NativeFinalEvidence('native655_sealed320_quality_qualified', True,
        'campaign', 'a' * 64, 'report', 'b' * 64, 'c' * 64, 320)
    assert value.record()['quality_gate_passed'] and value.record()['final_task_accuracy_measured']
    with pytest.raises(ValueError):
        replace(value, joint_valid_cases=319).record()
    with pytest.raises(ValueError):
        replace(value, classification='native655_sealed320_quality_below_95').record()
    stopped = replace(value, classification='native655_sealed320_quality_below_95', quality_gate_passed=False, joint_valid_cases=0)
    assert not stopped.record()['final_task_accuracy_measured']
    assert not value.record()['speed_tested'] and not value.record()['confirmed']
