import json
import pytest

from experiments.structure_net import ncm_evidence_teacher_pilot_v3 as run


def test_single_first_block_retained_without_commentary_or_second_candidate():
    row = {'id': 'a', 'source': 'One line.'}
    value = {'summary': 'The passage discusses one line.', 'support': []}
    text = '```json\n' + json.dumps(value) + '\n```'
    assert run.parse(text + '\nReview explanation.', row) == value
    with pytest.raises(ValueError):
        run.parse(text + '\n' + text, row)
    with pytest.raises(ValueError):
        run.parse(text + '\n{"another":"candidate"}', row)
