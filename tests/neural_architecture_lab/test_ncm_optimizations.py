import pytest
from neural_architecture_lab.ncm_optimization_hypotheses import replica_classification,factor_classification
from experiments.neural_architecture_lab.finalize_ncm_optimizations import review_count,write_json,digest


def test_replica_budget_and_parity_gate_precede_speed():
    assert replica_classification(1799,[1000.],77.)=='replica_output_parity_failed'
    assert replica_classification(1800,[],77.)=='replica_throughput_target_unmet'
    assert replica_classification(1800,[154.],77.)=='replica_throughput_target_met'
    assert replica_classification(1800,[153.],77.)=='replica_throughput_target_unmet'


def test_factor_quality_cannot_be_rescued_by_size_or_speed():
    assert factor_classification(99,100,.5,2.,2.)=='factorized_quality_failed'
    assert factor_classification(100,100,.91,2.,2.)=='factorized_compression_target_unmet'
    assert factor_classification(100,100,.9,1.05,1.05)=='factorized_smaller_and_faster_on_bounded_task'
    assert factor_classification(100,100,.8,1.,1.1)=='factorized_smaller_speed_target_unmet'


def test_source_reviews_require_binding_and_complete_coverage(tmp_path):
    p=tmp_path/'review.json';ex={'a':{'source_sha256':'s','source':'Final decision: correct.'}}
    rows=[{'id':'a','text':'correct'}]
    item={'id':'a','source_sha256':'s','text_sha256':digest('correct'),'source_quote':'Final decision: correct.','rationale':'matches','passed':True}
    write_json(p,{'items':[item]});assert review_count(rows,ex,p)==1
    item['text_sha256']='wrong';write_json(p,{'items':[item]})
    with pytest.raises(ValueError):review_count(rows,ex,p)
