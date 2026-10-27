import numpy as np
import pytest

from experiments.structure_net import ncm_public_code_prior as prior


def test_modes_use_lowest_code_tie_break_and_reject_incomplete_counts():
    counts=np.array([[1,2,2],[0,3,2]],dtype=np.uint32)
    assert prior.modes(counts)==[1,1]
    for bad in (np.zeros_like(counts),np.array([[1,0],[0,2]]),np.array([1,2])):
        with pytest.raises(ValueError):prior.modes(bad)


def test_accumulation_handles_equal_codes_at_distinct_positions_and_bounds():
    counts=np.zeros((2,3),dtype=np.uint32)
    prior.add(counts,[1,1]);prior.add(counts,[1,2])
    assert counts.tolist()==[[0,2,0],[0,1,1]]
    for codes in ([1],[1,3],[True,1],[1.,2.]):
        with pytest.raises(ValueError):prior.add(counts,codes)


def test_nonmodal_agreement_excludes_only_reference_modes_and_handles_empty_slice():
    result=prior.compare([[0,1],[1,1]],[[0,1],[2,1]],[0,1])
    assert result['correct']==3 and result['positions']==4
    assert result['nonmodal_positions']==1 and result['nonmodal_correct']==0
    assert prior.compare([[0]],[[0]],[0])['nonmodal_agreement'] is None


def test_entropy_distinguishes_constant_and_uniform_codes_without_nan():
    np.testing.assert_allclose(prior.entropy(np.array([[2,0],[1,1]])),[0,1])


def test_independent_counter_replay_excludes_diagnostic_and_rejects_binding_change():
    def pair(i):return {'id':str(i),'source_sha256':'s','target_sha256':'t','native_frame_sha256':'f','native_lexical_tokens':12}
    index=[pair(i) for i in range(3)]
    rows=[{**r,'codec_sha256':prior.bundle.CODEC_SHA,'codes':[i]*641} for i,r in enumerate(index)]
    cases=[{'id':'1','oracle_codes':[1]*641}]
    a,ap,ac=prior.accumulate(rows,index,cases,3)
    b,bp,bc=prior.accumulate(rows,index,cases,3,True)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ap,bp)
    assert ac==bc and ac['fitted']==2 and ac['excluded']==['1']
    assert a[:,1].sum()==0 and np.all(a.sum(1)==2)
    rows[0]['target_sha256']='different'
    with pytest.raises(ValueError):prior.accumulate(rows,index,cases,3)
