import pytest

from experiments.structure_net.ncm_broad_codec_reconstruction import gate, summarize


def test_codec_gate_keeps_invalids_and_number_regressions_in_denominator():
    original={"cases":252,"changed_number_cases":3}
    good={"cases":252,"token_recovery":.995,"first_EOS_correct":250,
          "active_invalid_ids":0,"changed_number_cases":3}
    assert gate(original,good)
    assert not gate(original,{**good,"first_EOS_correct":249})
    assert not gate(original,{**good,"active_invalid_ids":1})
    assert not gate(original,{**good,"changed_number_cases":4})
    with pytest.raises(ValueError):gate(original,{**good,"cases":251})
    with pytest.raises(ValueError):summarize([])
