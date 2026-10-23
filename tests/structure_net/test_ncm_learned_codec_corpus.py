import pytest

from experiments.structure_net.ncm_learned_codec_corpus import pack


def test_packing_preserves_cross_block_stream_and_only_true_end_EOS():
    ids = [11, 12, 13, 14, 15, 2]
    blocks = pack(ids, width=4, pad=0)
    assert blocks.tolist() == [[11, 12, 13, 14], [15, 2, 0, 0]]
    assert blocks.flatten()[:len(ids)].tolist() == ids
    assert blocks.eq(2).sum().item() == 1


def test_fixed_summary_canvas_never_truncates():
    assert pack([11, 12, 2], 4, 0, 8).tolist() == [[11, 12, 2, 0], [0, 0, 0, 0]]
    with pytest.raises(ValueError):
        pack(list(range(9)), 4, 0, 8)
    with pytest.raises(ValueError):
        pack([1, 2], 4, 0, 6)
