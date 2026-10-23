from experiments.neural_architecture_lab.finalize_ncm_intact_language import gates


def test_readable_language_does_not_pass_summary_quality_or_ncm():
    actual = gates(64, 50, 64)
    assert actual["language_component"]
    assert not actual["secondary_summary_quality"]
    assert not actual["NCM_success_established"]
    assert gates(64, 61, 61)["secondary_summary_quality"]
    assert not gates(63, 61, 61)["language_component"]
    assert not gates(64, 60, 60)["language_component"]
