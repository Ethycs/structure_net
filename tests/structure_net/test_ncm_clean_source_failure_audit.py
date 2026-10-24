import pytest

from experiments.neural_architecture_lab import finalize_ncm_clean_source_partition_failure as run


def test_failure_parser_retains_the_full_required_counts_and_both_deficits():
    values = {b: {"available": 16 if b >= 3 else 10000, "required": run.run.FINAL[b] + run.run.DEVELOPMENT[b]}
        for b in run.run.FINAL}
    text = "ValueError: insufficient disjoint source components for fixed quotas: " + repr(values)
    assert run.logged_counts(text) == values
    with pytest.raises(ValueError, match="one original"):
        run.logged_counts(text + "\n" + text)
    values[3]["required"] = 16
    with pytest.raises(ValueError, match="full quotas"):
        run.logged_counts("ValueError: insufficient disjoint source components for fixed quotas: " + repr(values))
