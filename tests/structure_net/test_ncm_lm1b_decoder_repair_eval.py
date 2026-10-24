from copy import deepcopy

import pytest

from experiments.neural_architecture_lab.finalize_ncm_lm1b_decoder_repair import replay_numeric
from experiments.structure_net.ncm_lm1b_codec import LEVELS


class Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(v) for v in ids if v not in (0, 101))


def fixture():
    gold = [101, 500, 501, 101]
    row = {"id": "a", "target_ids": gold, "packed_ids": [gold + [0] * 124],
        "native_round_trip": "500 501", "fits_one_native_frame": True}
    output = {"id": "a", "target_ids": gold, "raw_ids": gold + [0] * 124,
        "codes": [[0] * sum(LEVELS)], "text": "500 501", "native_round_trip": "500 501",
        "correct": 4, "total": 4, "exact_native_ids": True, "first_EOS_correct": True,
        "frame_EOS_positions": [3], "all_frame_BOS_correct": True, "active_invalid_ids": 0,
        "fits_one_native_frame": True}
    return row, output


def test_numeric_audit_replays_predictions_without_gpu():
    row, output = fixture()
    result = replay_numeric([row], [output], {"a": deepcopy(output)}, Tokenizer())
    assert result["correct"] == result["total"] == 4
    assert result["cases"] == result["exact_native_ids"] == result["first_EOS_correct"] == 1


@pytest.mark.parametrize("field,value", [("text", "unearned repair"), ("correct", 3),
    ("first_EOS_correct", False), ("raw_ids", [101, 500, 501, 101]), ("active_invalid_ids", 1)])
def test_numeric_audit_rejects_fabricated_or_incomplete_metrics(field, value):
    row, output = fixture()
    old = {"a": deepcopy(output)}
    output[field] = value
    with pytest.raises(ValueError):
        replay_numeric([row], [output], old, Tokenizer())


def test_numeric_audit_rejects_changed_codes_and_omitted_cases():
    row, output = fixture()
    old = {"a": deepcopy(output)}
    output["codes"][0][0] = 1
    with pytest.raises(ValueError, match="codes changed"):
        replay_numeric([row], [output], old, Tokenizer())
    with pytest.raises(ValueError, match="complete ordered"):
        replay_numeric([row], [], old, Tokenizer())
