from experiments.structure_net.ncm_budget128_profile import RUBRIC, calibration, install
from experiments.structure_net import ncm_memory_accuracy_v2 as q


def test_budget128_profile_preserves_all_controls_and_actual_request_globals():
    old = q.RUBRIC
    packets, expected = calibration()
    assert len(expected) == 50 and sum(len(p["candidates"]) for p in packets) == 50
    assert "approximately 32-word budget" not in RUBRIC
    assert "Do not penalize" in RUBRIC and "Every material semantic claim must remain supported" in RUBRIC
    try:
        install()
        assert q.request.__globals__["RUBRIC"] == RUBRIC == q.RUBRIC
    finally:
        q.RUBRIC = old
        q.original.RUBRIC = old
