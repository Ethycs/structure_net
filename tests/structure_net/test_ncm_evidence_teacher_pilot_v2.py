import json
import pytest

from experiments.structure_net import ncm_evidence_teacher_pilot_v2 as run


def test_fence_amendment_preserves_all_summary_and_support_bytes():
    value = {'summary': 'The passage describes blue skies.',
        'support': [{'summary_span': 'blue skies', 'source_lines': ['L0']}]}
    row = {'id': 'a', 'source': 'The sky is blue.'}
    text = json.dumps(value)
    assert run.parse(text, row) == run.parse('```json\n' + text + '\n```', row) == value
    assert run.parse('```\n' + text + '\n```', row) == value
    with pytest.raises(ValueError):
        run.parse('Comment\n```json\n' + text + '\n```', row)
    with pytest.raises(ValueError):
        run.parse('```json\n' + text, row)
