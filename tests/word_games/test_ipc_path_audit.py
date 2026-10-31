import pytest

from experiments.word_games.audit_ipc_path import timestamp_us, token_intervals


def test_timestamp_crosses_minute_boundary():
    assert timestamp_us("1.00.000.010 D x") - timestamp_us("0.59.999.990 D x") == 20
    assert timestamp_us("unstructured text") is None


def test_intervals_exclude_warmup_and_request_boundaries():
    log = "\n".join([
        "0.00.000.001 D slot process_toke: id 0 | task 0 | n_gen = 1",
        "0.00.000.002 D srv        decode: batch",
        "0.00.000.010 D que yield: done yielding to queue",
        "0.00.000.011 D slot process_toke: id 0 | task 0 | n_gen = 2",
        "0.00.001.000 D slot process_toke: id 0 | task 8 | n_gen = 1",
        "0.00.001.010 D srv        decode: batch",
        "0.00.001.090 D que yield: done yielding to queue",
        "0.00.001.100 D slot process_toke: id 0 | task 8 | n_gen = 2",
        "0.00.009.000 D slot process_toke: id 0 | task 10 | n_gen = 1",
    ])
    intervals, counts, warmup = token_intervals(log)
    assert warmup == 0 and counts == {0: 2, 8: 2, 10: 1}
    assert intervals == [{"task": 8, "generated": 2, "cycle_us": 100,
                          "before_decode_us": 10, "decode_region_us": 80, "after_decode_us": 10}]


def test_missing_timestamps_fail_instead_of_reporting_zero_cost():
    with pytest.raises(ValueError, match="no token timestamps"):
        token_intervals("no instrumentation")
