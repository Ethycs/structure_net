import pytest
from experiments.structure_net.ncm_native_word_mechanism import ARMS, aggregate, score


class Tokenizer:
    def decode(self, ids, skip_special_tokens):
        return " ".join(str(i) for i in ids if i != 9)


def test_stops_masked_tokens_and_teacher_forced_denominator():
    value = score([1, 9, 3, 4], [1, 2, 3, -100], {9}, Tokenizer())
    assert value["active_ids"] == [1, 9]
    assert value["correct"] == 2 and value["active_correct"] == 1 and value["total"] == 3
    assert value["text"] == "1"
    assert ARMS == {"A": (False, False), "B": (True, False), "C": (False, True), "D": (True, True)}
    rows = [{"cohort": "train", "arms": {arm: {**value, "valid_position_CE": 2.0} for arm in ARMS}}]
    result = aggregate(rows)["train"]
    assert result["A"]["token_recovery"] == 2 / 3
    assert result["D"]["task_accuracy"] is None
    assert result["D"]["oracle_plan"] and result["D"]["teacher_forced"]
    with pytest.raises(ValueError):
        score([1], [-100], {9}, Tokenizer())
