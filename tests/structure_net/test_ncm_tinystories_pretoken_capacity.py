import pickle

import pytest

from experiments.structure_net import ncm_tinystories_pretoken_capacity as r


def test_inclusive_capacity_and_explicit_boundary_uncertainty():
    assert all(r.capacity(126).values())
    assert r.capacity(127) == {"fits128_no_frame_overhead": True, "fits127_plus_one_boundary": True,
        "fits126_plus_two_boundaries": False}
    assert r.capacity(128) == {"fits128_no_frame_overhead": True, "fits127_plus_one_boundary": False,
        "fits126_plus_two_boundaries": False}
    assert not any(r.capacity(129).values())
    with pytest.raises(ValueError):
        r.capacity(-1)


def test_plain_asset_reader_rejects_global_references():
    assert r.plain_load(pickle.dumps({0: b"a"})) == {0: b"a"}
    with pytest.raises(pickle.UnpicklingError):
        r.plain_load(pickle.dumps(eval))


def test_author_byte_encoder_can_preserve_case_and_unicode_on_fixture():
    vocab = {0: b"<|endoftext|>", 1: b"<|pad|>", **{i + 2: bytes([i]) for i in range(256)}}
    tokenizer = r.bpe(vocab, [])
    for text in ("USB File.py 10 MB", "Café and naïve UTF-8", "x != X; A + a = 2"):
        assert tokenizer.decode(tokenizer.encode(text), skip_special_tokens=True) == text
    merged = r.bpe({**vocab, 258: b"Ab"}, {(b"A", b"b"): 0})
    assert merged.encode("Ab") == [258]
    assert merged.decode([258]) == "Ab"
