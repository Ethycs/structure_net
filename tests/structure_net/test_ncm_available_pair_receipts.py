import json
import pytest
from experiments.structure_net import ncm_available_pair_receipts as receipts


def test_reader_question_is_not_misclassified_as_a_merge_request():
    body={'fragments':[{'summary':'saved summary'}],'question':'a held-out question'}
    value={'messages':[{'role':'user','content':json.dumps(body)}]}
    assert receipts.request_shape(value) is None
    del body['question'];value['messages'][0]['content']=json.dumps(body)
    assert receipts.request_shape(value)[0]=='merge'


def test_raw_response_requires_correct_label_order_and_own_support():
    fragments=[{'label':'T0','fragment':'first source'},{'label':'T1','fragment':'second source'}]
    atoms=[{'label':'T0','summary':'first','support':['first']},{'label':'T1','summary':'second','support':['second']}]
    assert len(receipts.raw_pairs(fragments,json.dumps({'atoms':atoms})))==2
    with pytest.raises(ValueError):receipts.raw_pairs(fragments,json.dumps({'atoms':atoms[::-1]}))
    atoms[0]['support']=['second']
    with pytest.raises(ValueError):receipts.raw_pairs(fragments,json.dumps({'atoms':atoms}))
