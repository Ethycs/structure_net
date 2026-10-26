import pytest
from experiments.structure_net.ncm_codec_roundtrip import metrics, aggregate, disjoint


def frame(words):
    return [101,*words,101]+[0]*(126-len(words))


def test_boundary_and_number_errors_are_not_hidden_by_content_average():
    gold=frame([7,8,9])
    raw=gold.copy(); raw[4]=55
    value=metrics(gold,raw,'Ana paid 12.5 dollars','Ana paid 125 dollars')
    assert value['content_matches']==value['content_tokens']==3
    assert not value['exact_active_sequence']
    assert not value['exact_rendered']
    assert value['has_digits'] and not value['digit_sequence_preserved']


def test_padding_does_not_inflate_accuracy():
    gold=frame([7,8])
    raw=frame([17,18])
    value=metrics(gold,raw,'a b','c d')
    assert value['content_matches']==0 and value['content_tokens']==2
    assert value['active_matches']==2 and value['active_tokens']==4


def test_split_rejects_source_overlap_with_distinct_example_ids():
    first=dict(id='a',source_id='session',group_id='g1',source_sha256='h1')
    second=dict(id='b',source_id='session',group_id='g2',source_sha256='h2')
    with pytest.raises(ValueError,match='source_id'):
        disjoint([first],[second])


def test_numbers_use_only_nonempty_digit_denominator():
    a=metrics(frame([7]),frame([7]),'12','12')
    b=metrics(frame([8]),frame([8]),'hello','hello')
    rows=[dict(**x,split='reserved_component',arm='test') for x in (a,b)]
    result=aggregate(rows)['reserved_component']['test']
    assert result['cases']==2 and result['has_digits']==result['digit_cases_preserved']==1
    assert result['reconstruction_gate'] is None
