import pytest

from experiments.structure_net.run_ncm_replicas import Config, summarize, waves


def evidence(copies=9):
    rows=[];batches=[]
    for rep in range(3):
        for wave,part in enumerate(waves(list(range(100)),copies)):
            batches.append({'repeat':rep,'wave':wave,'ids':part,'wall_seconds':1.})
            rows.extend({'id':i,'repeat':rep,'wave':wave,'parity':True,'common_output_tokens':20} for i in part)
    return rows,batches


def test_partial_wave_counts_actual_requests_without_padding_or_dropping():
    parts=waves(list(range(100)),9)
    assert len(parts)==12 and len(parts[-1])==1
    assert [i for p in parts for i in p]==list(range(100))
    rows,batches=evidence()
    result=summarize(rows,batches,9)
    assert result['requests']==300 and result['common_output_tokens']==6000
    assert result['aggregate_tokens_per_second']==6000/36
    assert result['effective_tokens_per_second_per_copy']==6000/36/9


def test_failed_parity_missing_cells_and_double_counted_waves_reject():
    rows,batches=evidence()
    rows[0]['parity']=False
    with pytest.raises(ValueError):summarize(rows,batches,9)
    rows[0]['parity']=True
    with pytest.raises(ValueError):summarize(rows[:-1],batches,9)
    with pytest.raises(ValueError):summarize(rows,batches+[batches[0]],9)
    with pytest.raises(ValueError):Config(10)
