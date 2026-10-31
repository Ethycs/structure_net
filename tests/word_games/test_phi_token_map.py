import pytest

from experiments.word_games.phi_token_map import Vocabulary, TokenMapBundle, build_map


def vocab(name, pieces, attrs=None):
    return Vocabulary(name, pieces, attrs or [4] * len(pieces), {"tokenizer": "test"})


def test_mapping_never_assumes_equal_ids_or_equal_bpe_boundaries():
    small = vocab("small", [b"a", b"b", b"ab"])
    large = vocab("large", [b"ab", b"b", b"a"])
    m = build_map(small, large, lambda b: {b"a": [2], b"b": [1], b"ab": [0]}[b])
    assert m.translate([0, 1]) == [2, 1]
    assert large.decode(m.translate([0, 1])) == b"ab"
    assert m.translate([0, 1]) != [0]  # whole-span canonical tokenization


def test_partial_utf8_and_special_spelling_are_not_lost_or_promoted():
    small = vocab("small", [b"\xe2\x82", b"<end>", b"<end>", b""], [4, 4, 8, 2])
    large = vocab("large", [b"<end>", b"\xe2", b"\x82", b"<", b"end", b">"], [8, 4, 4, 4, 4, 4])
    m = build_map(small, large, lambda b: [3, 4, 5])
    assert m.translate([0]) == [1, 2]
    assert large.decode(m.translate([0])) == b"\xe2\x82"
    assert m.direct[1] == -1 and m.translate([1]) == [3, 4, 5]
    for bad in [-1, 2, 3, 100]:
        with pytest.raises(ValueError): m.translate([bad])


def test_native_canonical_alias_and_multi_token_decomposition():
    small = vocab("small", [b"a", b"ab"])
    large = vocab("large", [b"a", b"a", b"b"])
    m = build_map(small, large, lambda b: [1] if b == b"a" else [1, 2])
    assert m.direct == [1, -1]
    assert m.translate([1], limit=1) == [1]
    assert m.translate([1], limit=0) == []
    with pytest.raises(ValueError): m.translate([0], limit=-1)


def test_artifact_integrity_and_runtime_vocab_binding(tmp_path):
    a = vocab("small", [b"a", b"b"])
    b = vocab("large", [b"b", b"a"])
    bundle = TokenMapBundle(build_map(a, b, lambda x: [b.pieces.index(x)]),
                            build_map(b, a, lambda x: [a.pieces.index(x)]))
    path = tmp_path / "map"
    bundle.save(path, {"test": True})
    loaded = TokenMapBundle.load(path)
    assert loaded.small_to_large.translate([0, 1]) == [1, 0]
    loaded.validate_tokenizers(a, b)
    with pytest.raises(ValueError): loaded.validate_tokenizers(a, vocab("wrong", [b"x", b"b"]))
    with (path / "small_to_large.json.gz").open("ab") as f: f.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"): TokenMapBundle.load(path)


def test_reject_native_encoder_that_changes_bytes():
    with pytest.raises(ValueError, match="changed bytes"):
        build_map(vocab("small", [b"a"]), vocab("large", [b"b"]), lambda _: [0])
