import pytest
import torch

from experiments.structure_net.ncm_word_final_data import align,overlaps
from experiments.structure_net.ncm_prefix_geometry import source_frame


def test_unicode_overlap_and_native_prompt_offsets_preserve_every_query_token():
    # Two byte-token pieces can share a Unicode character offset.
    native=[(0,0),(10,12),(12,13),(12,13),(13,16),(0,0)]
    query=[(0,2),(2,3),(2,3),(3,6)]
    mapping=overlaps(query,native,10)
    hidden=torch.arange(12,dtype=torch.float32).reshape(6,2)
    result=align(hidden,mapping)
    assert torch.equal(result[0],hidden[1])
    assert torch.equal(result[1],(hidden[2]+hidden[3])/2)
    assert torch.equal(result[1],result[2])
    assert torch.equal(result[3],hidden[4])
    assert source_frame(2744)==2816
    with pytest.raises(ValueError,match='clipping'):
        source_frame(8193,maximum=8192)
    with pytest.raises(ValueError,match='overlap'):
        overlaps([(0,1)],[(10,12)],0)
