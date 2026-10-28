import pytest

from experiments.neural_architecture_lab import ncm_recovered_review_milestones as reporting


def test_label_firsts_are_separate_from_author_rendering_and_model_accuracy():
    rows = reporting.extension_records(5190, 5263, 5519)
    assert [r['added_firsts'] for r in rows] == [499, 73, 256]
    assert [(r['before_firsts'], r['after_firsts']) for r in rows] == [(4691, 5190), (5190, 5263), (5263, 5519)]
    assert all(r['included_in_application_accuracy'] is False for r in rows)
    for values in ((4690, 5263, 5519), (5190, 5189, 5519), (5190, 5263, 5262)):
        with pytest.raises(ValueError): reporting.extension_records(*values)
