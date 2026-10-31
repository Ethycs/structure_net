from pathlib import Path
import pytest
from experiments.word_games.hybrid_experts import command, validate_stats


def test_partition_keeps_attention_primary_and_protected_gpu_hidden():
    args=command('pins64',18088,'test',Path('/tmp/word-games-test'))
    for key,value in {'--device':'CUDA0,CUDA1','--tensor-split':'1,0','--n-cpu-moe':'26',
                      '--cache-type-k':'q8_0','--cache-type-v':'q8_0','--threads':'12'}.items():
        assert args[args.index(key)+1]==value
    assert 'WORD_GAMES_HYBRID_PINS=64' in args
    assert 'GPU-8cf440df' not in str(args)
    assert all('WORD_GAMES_HYBRID_PINS' not in a for a in command('experts8',18088,'test',Path('/tmp/x')))


def test_routed_accounting_and_numerics_fail_closed():
    good=dict(pins=64,layers=26,validated_layers=26,calls=100,gpu_hits=200,cpu_misses=600,
              validation_max_relative_l2=.001)
    validate_stats(good,'pins64')
    for change in ({'cpu_misses':599},{'gpu_hits':0,'cpu_misses':800},{'validated_layers':25},
                   {'validation_max_relative_l2':.03},{'validation_max_relative_l2':float('nan')}):
        with pytest.raises(ValueError):validate_stats({**good,**change},'pins64')
    zero={**good,'pins':0,'gpu_hits':0,'cpu_misses':800,'validation_max_relative_l2':0}
    validate_stats(zero,'pins0')


def test_unknown_partition_rejected():
    with pytest.raises(ValueError):command('pins256',1,'test',Path('/tmp/x'))


def test_registry_keeps_speed_separate_from_quality():
    import runpy
    cls=runpy.run_path(str(Path(__file__).resolve().parents[2]/
        'src/neural_architecture_lab/word_games_expert_split_hypothesis.py'))['ExpertSplitEvidence']
    args=dict(hypothesis_id='word-games-three-gpu-experts-v1',
        classification='full_speedup_quality_unestablished',description='measured',
        baseline_tok_s=30,candidate_tok_s=40,full_passes=2,completed_requests=72,service_restored=True,
        campaign='a',campaign_sha256='a'*64,report='b',report_sha256='b'*64)
    assert cls(**args).record()['confirmed'] is False
    assert cls(**args).record()['accuracy_noninferiority_established'] is False
    for change in ({'candidate_tok_s':29},{'full_passes':1},{'service_restored':False}):
        with pytest.raises(ValueError):cls(**{**args,**change})
