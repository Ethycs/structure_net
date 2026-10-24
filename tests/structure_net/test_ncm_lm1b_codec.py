import pytest

from experiments.structure_net import ncm_lm1b_codec as run


def test_native_packing_preserves_all_words_and_one_EOS_without_truncation():
    for length in (1, 127, 128, 148, 300):
        ids = [1024] * length
        full, windows = run.pack(ids)
        assert full == ids + [101]
        assert all(len(w) == 128 for w in windows)
        assert [v for w in windows for v in w if v != 0] == full
        assert sum(v == 101 for w in windows for v in w) == 1


def test_native_packing_rejects_embedded_eos_pad_and_outside_vocabulary():
    for ids in ([], [0], [101], [30522], [-1]):
        with pytest.raises(ValueError):
            run.pack(ids)


def test_native_round_trip_and_special_tokens_use_matching_release():
    t = run.tokenizer()
    assert (t.eos_token_id, t.pad_token_id, len(t)) == (101, 0, 30522)
    ids = t.encode("Budget $2,500; due 2026-11-02.", add_special_tokens=False)
    full, windows = run.pack(ids)
    assert len(full) <= 128 and len(windows) == 1
    # BERT inserts punctuation spaces and lowercases; numeric content stays intact.
    decoded = t.decode(ids)
    assert "2,500" in decoded.replace(" ", "") and "2026-11-02" in decoded.replace(" ", "")
    assert t.encode(decoded, add_special_tokens=False) == ids
