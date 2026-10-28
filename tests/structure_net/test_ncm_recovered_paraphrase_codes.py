import hashlib
import struct

import pytest

from experiments.structure_net import ncm_recovered_paraphrase_codes as study


def row(ident, token):
    ids=[101,token,101]+[0]*125
    return {'id':ident,'native_word_ids':ids,'native_lexical_tokens':1,
        'native_frame_sha256':hashlib.sha256(struct.pack('<128H',*ids)).hexdigest()}


def test_frame_dedup_preserves_all_pair_bindings_and_has_stable_order():
    rows=[row('b',2000),row('a',2000),row('c',2001)]
    frames,pairs=study.unique_frames(rows)
    assert pairs==3 and len(frames)==2
    assert sorted(r['pair_ids'] for r in frames)==[['a','b'],['c']]
    assert study.unique_frames(reversed(rows))==(frames,pairs)


def test_partial_frame_wrong_hash_and_duplicate_pair_are_rejected():
    value=row('a',2000)
    with pytest.raises(ValueError):study.unique_frames([value,value])
    with pytest.raises(ValueError):study.unique_frames([{**value,'native_frame_sha256':'bad'}])
    with pytest.raises(ValueError):study.unique_frames([{**value,'native_word_ids':value['native_word_ids'][:-1]}])


def test_controls_are_unique_and_cover_complete_short_and_long_frames():
    frames=[{'native_lexical_tokens':i+1} for i in range(40)]
    positions=study.controls(frames)
    assert len(positions)==len(set(positions))==32
    assert 0 in positions and 39 in positions
