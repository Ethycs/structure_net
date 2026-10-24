"""Assert complete exact packet inputs and source-only accuracy boundaries."""
from experiments.structure_net import ncm_sol64_codec_semantics as run


def test_actual64_reconstructions_are_bound_to_exact_sources():
    run.configure()
    cases = run.require_numeric()
    outputs = {r['id']: r for r in run.q.read(run.numeric.OUTPUTS / 'outputs.json')}
    assert len(cases) == 64 and len({r['source_component'] for r in cases}) == 64
    for case in cases:
        assert set(case['texts']) == set(run.ARMS)
        for arm in run.ARMS:
            assert case['texts'][arm] == outputs[case['id']]['conditions'][arm]['text']
    scores = [{'id': str(i), 'valid': True, 'passed': True, 'text': 'summary', 'tokens': 1,
        'evaluations': [{'faithful': True, 'salient': True, 'readable': True}]} for i in range(64)]
    metric = run.metric(scores)
    assert metric['student_task_accuracy'] is None
    assert metric['source_prediction'] is False and metric['oracle_target_codes'] is True
