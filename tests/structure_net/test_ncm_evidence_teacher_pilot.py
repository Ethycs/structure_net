import json
import pytest

from experiments.structure_net import ncm_evidence_teacher_pilot as run


def test_support_checks_only_literal_syntax_not_semantic_claims():
    row = {'id': 'a', 'source': 'The sky is blue.'}
    # An unsupported claim must remain the first candidate for later semantic judgment.
    value = {'summary': 'The sky is red.', 'support': [{'summary_span': 'red', 'source_lines': ['L0']}]}
    assert run.parse(json.dumps(value), row) == value
    assert run.parse(json.dumps({'summary': '', 'support': []}), row)['summary'] == ''
    value['support'][0]['source_lines'] = ['L999']
    with pytest.raises(ValueError):
        run.parse(json.dumps(value), row)


def test_review_sees_same_complete_source_and_exact_draft():
    row = {'id': 'a', 'source': 'A source sentence.\nA second complete sentence.'}
    draft = {'summary': 'A draft.', 'support': []}
    body = run.payload(row, 'review', draft)
    value = json.loads(body['messages'][1]['content'])
    assert value['source_lines'] == run.source_lines(row)
    assert value['untrusted_draft'] == draft
    assert body['model'] == run.MODELS['review'] and body['max_tokens'] == 512


def test_selection_excludes_every_preceding_component():
    rows = [{'id': f'{band}-{i}', 'bin': band, 'source_component': f'{band}-{i}'}
        for band in range(5) for i in range(40)]
    prior = [r for r in rows if r['id'].endswith('-0')]
    chosen = run.select(rows, prior)
    assert len(chosen) == 64
    assert not {r['source_component'] for r in chosen} & {r['source_component'] for r in prior}
