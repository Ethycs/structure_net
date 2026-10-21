import pytest
from experiments.structure_net.run_ncm_batching import static_inputs


def test_static_tail_duplicates_compute_without_inventing_real_requests():
    examples=[{'prefix_ids':[i]+[50257]*511} for i in (1,2,3,4)]
    padded=static_inputs(examples,9)
    assert len(examples)==4 and len(padded)==9
    assert padded[:4]==[e['prefix_ids'] for e in examples]
    assert padded[4:]==[examples[0]['prefix_ids']]*5
    with pytest.raises(ValueError):static_inputs([],9)
    with pytest.raises(ValueError):static_inputs(examples,1)
