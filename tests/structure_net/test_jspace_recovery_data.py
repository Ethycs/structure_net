"""Untouched selection excludes historical roles and preserves prompt semantics."""
from types import SimpleNamespace

import pytest

from experiments.structure_net.jspace_recovery_data import (
    SYSTEM, all_identities, candidate_rows, identity, select_confirmation,
)


def row(i):
    return dict(sentence=f"The _ case {i}.", option1="first", option2="second", answer="1")


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return list(range(385 if "case 9." in messages[1]["content"] else 12))


def test_all_historical_splits_and_exclusion_lists_are_excluded():
    value = dict(splits={"train": [{"identity": "train"}], "evaluation": [{"identity": "test"}]},
        previous_identities=["old_test", "old_train"], nested={"splits": {"development": [{"identity": "dev"}]}})
    assert all_identities(value) == {"train", "test", "old_test", "old_train", "dev"}


def test_selector_is_disjoint_length_limited_deduplicated_and_deterministic():
    rows = [row(i) for i in range(10)] + [row(1)]
    excluded = {identity("Fill the blank in this sentence.\nThe _ case 0.", ["first", "second"])}
    tokenizer = Tokenizer()
    selected, report = select_confirmation("winogrande", rows, excluded, tokenizer, count=8)
    assert len(selected) == len({r["identity"] for r in selected}) == 8
    assert not excluded & {r["identity"] for r in selected}
    assert all(len(r["input_ids"]) == 12 for r in selected)
    assert {r["source_row"] for r in selected} == set(range(1, 9))
    again, _ = select_confirmation("winogrande", rows, excluded, Tokenizer(), count=8)
    assert selected == again
    for messages, options in tokenizer.calls:
        assert messages[0] == {"role": "system", "content": SYSTEM}
        assert options == dict(tokenize=True, add_generation_prompt=True)
        assert messages[1]["content"].endswith("A. first\nB. second")


def test_insufficient_eligible_confirmation_fails_instead_of_reusing_old_rows():
    with pytest.raises(ValueError, match="Only 2 eligible untouched"):
        select_confirmation("winogrande", [row(1), row(2), row(9)], set(), Tokenizer(), count=3)
