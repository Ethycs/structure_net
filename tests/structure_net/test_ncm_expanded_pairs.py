import pytest

from experiments.structure_net.ncm_expanded_pairs import select_sources, matches_public, public_index
from experiments.structure_net.ncm_memory_data import text_hash


class Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt):
        assert add_generation_prompt
        return list(range(len(messages[1]["content"])))


def row(i, group=None, source="public source", digest=None, split="train"):
    return {"id": str(i), "group_id": group or str(i), "source_id": str(i),
        "source": source, "source_sha256": digest or str(i), "split": split,
        "native_eligible": True, "source_gpt2_tokens": 20, "spans": [],
        "target": "old target must never be included", "target_sha256": "old"}


def test_fresh_selection_excludes_groups_hashes_test_and_long_inputs_without_old_targets():
    forbidden = {"group_id": {"old"}, "source_id": set(), "source_sha256": {"blocked"}}
    records = [row(0, group="old"), row(1, digest="blocked"), row(2, source="x"*513),
        row(3, split="test"), row(4), row(5, group="4"), row(6)]
    selected = select_sources(records, forbidden, Tokenizer(), 2)
    assert len(selected) == 2
    assert {x["group_id"] for x in selected} == {"4", "6"}
    assert all("target" not in x and "target_sha256" not in x for x in selected)
    assert select_sources(list(reversed(records)), forbidden, Tokenizer(), 2) == selected


def test_selection_fails_instead_of_reusing_or_truncating_sources():
    forbidden = {"group_id": set(), "source_id": set(), "source_sha256": set()}
    with pytest.raises(ValueError, match="not enough"):
        select_sources([row(0, source="x"*513)], forbidden, Tokenizer(), 1)


def test_canonical_span_filter_checks_full_turn_range_hash_and_reconstruction():
    turn = "First. Public sentence. End."
    index = public_index([{"haystack_session_ids": ["session"],
        "haystack_sessions": [[{"role": "assistant", "content": turn}]]}])
    source = turn[7:23]
    span = {"source_id": "session", "role": "assistant", "turn_text_sha256": text_hash(turn),
        "start_char": 7, "end_char": 23, "span_text_sha256": text_hash(source)}
    case = {"source": source, "source_sha256": text_hash(source), "spans": [span]}
    assert matches_public(case, index)
    assert not matches_public({**case, "source": source+" invented"}, index)
    assert not matches_public({**case, "spans": [{**span, "role": "user"}]}, index)
    assert not matches_public({**case, "spans": [{**span, "span_text_sha256": "wrong"}]}, index)
