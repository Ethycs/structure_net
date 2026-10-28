import pytest

from experiments.structure_net.ncm_available_pair_supplement import resolve_pointer
from experiments.structure_net.ncm_available_pair_census import digest


def test_global_source_join_and_corrupt_pointer_rejection():
    whole='an exact source sentence';fragment=whole[3:8]
    p={'turn_text_sha256':digest(whole),'span_text_sha256':digest(fragment),'start_char':3,'end_char':8}
    lookup={digest(whole):whole,digest(fragment):fragment}.get
    assert resolve_pointer(p,lookup)==(fragment,'whole_turn_and_span_verified')
    with pytest.raises(ValueError,match='span hash'):
        resolve_pointer({**p,'end_char':9},lookup)


def test_independently_verified_fragment_and_missing_source():
    whole='some raw data';fragment=whole[5:]
    p={'turn_text_sha256':digest(whole),'span_text_sha256':digest(fragment),'start_char':5,'end_char':len(whole)}
    assert resolve_pointer(p,{digest(fragment):fragment}.get)==(fragment,'span_content_verified_without_whole_turn')
    with pytest.raises(ValueError,match='unavailable'):
        resolve_pointer(p,{}.get)
