import copy
import pytest

from experiments.neural_architecture_lab.audit_ncm_recovered_decoder_readout import ARMS, replay


def fixture():
    gold = [101, 17, 101] + [0]*125
    row = dict(id='a', split='reserved_component', target_sha256='abc', native_reference='17',
               target_text='17', native_word_ids=gold)
    outputs = [dict(id='a', arm=a, split=row['split'], target_sha256='abc', reference='17',
        target_text='17', text='17', raw_word_ids=gold[:], codes=[0]*641, active_matches=3,
        active_tokens=3, content_matches=1, content_tokens=1, exact_active_sequence=True,
        exact_rendered=True, has_digits=True, digit_sequence_preserved=True, empty=False) for a in ARMS]
    return [row], outputs


def test_independent_counts_and_missing_cells():
    cases, outputs = fixture()
    metrics, paired = replay(cases, outputs, lambda raw: str(raw[1]))
    assert metrics['reserved_component']['new_codec_only']['digit_cases_preserved'] == 1
    assert paired['reserved_component']['new_source_effect']['both_exact'] == 1
    with pytest.raises(ValueError, match='exactly one'):
        replay(cases, outputs[:-1] + [outputs[0]], lambda raw: str(raw[1]))


@pytest.mark.parametrize('field,value,error', [
    ('content_matches', 0, 'arithmetic'), ('codes', [1]*641, 'codes differ'),
    ('text', '18', 'rendering'), ('raw_word_ids', [30528]*128, 'vocabulary'),
])
def test_rejects_tampered_outputs(field, value, error):
    cases, outputs = fixture()
    outputs = copy.deepcopy(outputs); outputs[1][field] = value
    with pytest.raises(ValueError, match=error):
        replay(cases, outputs, lambda raw: str(raw[1]))
