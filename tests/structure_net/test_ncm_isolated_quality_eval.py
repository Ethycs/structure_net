import pytest

from experiments.structure_net.ncm_isolated_quality_eval import require_generated


def test_runtime_validation_keeps_empty_candidates_for_whole_summary_grading():
    rows = [{"full_frame_fixed": True, "source_truncated": False, "text": ""} for _ in range(64)]
    require_generated(rows, 64)
    with pytest.raises(ValueError):
        require_generated(rows[:-1], 64)
    rows[0]["full_frame_fixed"] = False
    with pytest.raises(ValueError):
        require_generated(rows, 64)
    rows[0]["full_frame_fixed"] = True
    rows[0]["source_truncated"] = True
    with pytest.raises(ValueError):
        require_generated(rows, 64)
