import pytest

from experiments.structure_net.ncm_saved_recovered_code_readout import fixed_inputs


def inputs():
    controls = [{'id': str(i), 'population': 'recovered'} for i in range(4)]
    before = [{'codes': [i]*641} for i in range(4)]
    after = [{'codes': [i+4]*641} for i in range(4)]
    pairs = {str(i): {'oracle_codes': [99]*641} for i in range(4)}
    return controls, before, after, pairs


def test_saved_predictions_preserve_order_and_never_use_reference_codes():
    args = inputs()
    rows = fixed_inputs(*args)
    assert [r['id'] for r in rows] == ['0', '1', '2', '3']
    assert [r['before_source_prediction'] for r in rows] == [[i]*641 for i in range(4)]
    assert [r['after_source_prediction'] for r in rows] == [[i+4]*641 for i in range(4)]
    rows[0]['after_source_prediction'][0] = 8191
    assert args[2][0]['codes'][0] == 4


def test_missing_case_and_incomplete_prediction_are_rejected():
    controls, before, after, pairs = inputs()
    with pytest.raises(ValueError, match='all four'):
        fixed_inputs(controls[:-1], before, after, pairs)
    after[0]['codes'] = [0]*640
    with pytest.raises(ValueError, match='hierarchy'):
        fixed_inputs(controls, before, after, pairs)
