import pytest
from experiments.neural_architecture_lab.finalize_ncm_llama import CONDITIONS,digest,review_counts,summarize,write_json


def test_request_estimator_uses_case_medians_and_common_text_tokens():
    rows=[{"id":"a","wall_seconds":s,"common_output_tokens":10,"text":"same"} for s in (1.,2.,100.)]
    rows += [{"id":"b","wall_seconds":s,"common_output_tokens":20,"text":"same"} for s in (3.,4.,100.)]
    r=summarize(rows)
    assert r['sum_median_seconds']==6 and r['common_tokens_per_second']==5
    assert r['all_repetitions_same_output']


def test_all_conditions_require_explicit_bound_source_reviews(tmp_path):
    ex={'case':{'source_sha256':'source','source':'The final decision is X.'}}
    rows=[{'condition':a,'id':'case','text':'X'} for a in CONDITIONS]
    items=[{'condition':a,'id':'case','text_sha256':digest('X'),'source_sha256':'source',
        'passed':True,'verdict':'supported','source_quote':'The final decision is X.',
        'rationale':'The reviewed claim matches the final source decision.'} for a in CONDITIONS]
    path=tmp_path/'reviews.json';write_json(path,{'items':items})
    assert review_counts(rows,ex,path)=={a:1 for a in CONDITIONS}
    items[0]['source_quote']='This quote does not exist.';write_json(path,{'items':items})
    with pytest.raises(ValueError,match='support missing'):review_counts(rows,ex,path)
    write_json(path,{'items':items[1:]})
    with pytest.raises(ValueError,match='incomplete'):review_counts(rows,ex,path)
