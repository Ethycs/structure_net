import pytest

from experiments.structure_net import ncm_recovered_paraphrase_features as study


def test_feature_pilot_covers_metadata_controls_and_both_length_extremes():
    entries=[{'id':str(i),'source_positions':i+1,'native_prompt_positions':20-i} for i in range(12)]
    selected=study.select(entries,['5','6'])
    assert selected[:2]==[5,6]
    assert {0,1,2,3,8,9,10,11}<=set(selected)
    assert len(selected)==len(set(selected))


def test_duplicate_or_missing_source_control_is_rejected():
    entry={'id':'a','source_positions':1,'native_prompt_positions':2}
    with pytest.raises(ValueError):study.select([entry,entry],['a'])
    with pytest.raises(ValueError):study.select([entry],['b'])
    with pytest.raises(ValueError):study.select([entry],['a','a'])
