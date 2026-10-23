import pytest

from experiments.structure_net.ncm_oracle_codec_semantic_recovery import merge_canonical


def test_conflicting_first_decisions_cannot_be_selected_by_outcome():
    negative = {"packet": {"id": "same"}, "row": {"passed": False}}
    positive = {"packet": {"id": "same"}, "row": {"passed": True}}
    assert merge_canonical([{"k": negative}, {"k": negative, "j": positive}]) == {"k": negative, "j": positive}
    with pytest.raises(ValueError, match="conflicting"):
        merge_canonical([{"k": negative}, {"k": positive}])
