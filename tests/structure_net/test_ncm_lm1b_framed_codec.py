import pytest

from experiments.structure_net import ncm_lm1b_framed_codec as run


def test_every_native_frame_preserves_words_and_author_CLS_boundaries():
    for length in (1, 125, 126, 127, 149, 300):
        ids = [1000] * length
        full, frames = run.pack(ids)
        assert all(len(v) == 128 and v[0] == 101 for v in frames)
        assert [v for v in full if v != 101] == ids
        assert [v for frame in frames for v in frame if v != 0] == full
        assert len(frames) == (length + 125) // 126


def test_renderer_consumes_leading_BOS_and_stops_on_next_predicted_CLS():
    class Tokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return ",".join(map(str, ids))
    t = Tokenizer()
    frames = [[101, 1000, 1001, 101] + [2000] * 124, [101, 1002, 101] + [2001] * 125]
    assert run.render(frames, t) == ("1000,1001,1002", [3, 2], [True, True])
    # A missing BOS is measured, and no lexical token is silently removed.
    text, ends, bos = run.render([[1000, 101] + [2000] * 126], t)
    assert (text, ends, bos) == ("1000", [1], [False])
    with pytest.raises(ValueError):
        run.render([[101, 1000]], t)
