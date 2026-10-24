from types import SimpleNamespace

import pytest

from experiments.structure_net import ncm_public_exchange_fragments as run
from experiments.structure_net import ncm_source_split_integrity as guard
from experiments.structure_net.ncm_frozen_body_public_data import verify_cases


def encoding():
    return SimpleNamespace(encode=lambda text, **kwargs: text.split())


def test_windows_preserve_complete_user_led_exchanges_and_public_provenance():
    session = []
    for i in range(4):
        session += [{"role": "user", "content": f"Question {i}"},
            {"role": "assistant", "content": " ".join(f"term{i}_{j}" for j in range(550))}]
    values = run.windows("sharegpt_x_0", session, encoding())
    assert {r["bin"] for r in values} >= {1, 3, 4}
    for row in values:
        assert 512 <= row["source_cl100k_tokens"] <= 2048
        assert row["exchange_start_turn"] % 2 == row["exchange_end_turn"] % 2 == 0
        assert row["source"] == "\n".join(t["content"] for t in session[row["exchange_start_turn"]:row["exchange_end_turn"]])
        assert not row["source_truncated"]
    samples = [{"haystack_session_ids": ["sharegpt_x_0"], "haystack_sessions": [session]}]
    assert verify_cases(samples, values) == {r["id"]: r["source_sha256"] for r in values}


def test_oversized_exchange_is_never_clipped_to_fill_a_band():
    session = [{"role": "user", "content": " ".join(["word"] * 2500)},
        {"role": "assistant", "content": "reply"}]
    assert run.windows("session", session, encoding()) == []


def test_short_turn_hashes_do_not_connect_unrelated_long_sources():
    greeting = {"source_id": "a", "turn_text_sha256": "greeting"}
    a = {"id": "a", "source_id": "ultrachat_1", "group_id": "a", "source_sha256": "a",
        "source": "first distinct substantial passage", "spans": [greeting]}
    b = {**a, "id": "b", "source_id": "ultrachat_2", "group_id": "b", "source_sha256": "b",
        "source": "other novel substantive material", "spans": [{**greeting, "source_id": "b"}]}
    assert guard.overlap(a, b)["shared_public_turn"]
    av, bv = (run.comparison_view(r, {"greeting": 2}) for r in (a, b))
    assert guard.overlap(av, bv) is None
    assert a["spans"] and b["spans"]
    with pytest.raises(ValueError, match="inventory"):
        run.comparison_view(a, {})
