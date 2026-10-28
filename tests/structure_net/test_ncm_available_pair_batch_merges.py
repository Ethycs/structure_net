import json
import pytest
from experiments.structure_net.ncm_available_pair_batch_merges import job_body,batch_outputs


def test_batched_merge_contract_rejects_reader_questions_and_cross_job_output():
    job={'kind':'user_spine','fragments':[{'role':'user','transcript_date':'2020-01-01','summary':'Source'}],
         'user_spine':None,'max_output_tokens':128,'label':'S0'}
    body=job_body(job)
    assert body is not None and 'label' not in body
    assert job_body({**job,'question':'Answer this'}) is None
    assert batch_outputs([body],json.dumps({'summaries':[{'label':'S0','summary':'Target'}]}))==['Target']
    with pytest.raises(ValueError,match='label/order'):
        batch_outputs([body],json.dumps({'summaries':[{'label':'S1','summary':'Wrong slot'}]}))
