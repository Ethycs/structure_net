from copy import deepcopy

import pytest

from experiments.structure_net import ncm_engineering_summary_inventory as run


def raw(source, group, turn='turn', tokens=200):
    span = {'text': source, 'start_char': 0, 'end_char': len(source),
        'span_text_sha256': run.text_hash(source), 'turn_text_sha256': run.text_hash(turn), 'source_id': group}
    return {'source_group_id': group, 'evidence': [span], 'input': {'text': source, 'turns': [span]},
        'source_sha256': run.text_hash(source), 'input_tokens': tokens}


def test_shared_inputs_turns_and_session_aliases_cannot_cross_partitions():
    a = raw('An informative original source', 'a', 'shared turn')
    b = raw('A different fragment of that turn', 'b', 'shared turn')
    c = raw('Other material from the same session', 'b', 'another turn')
    d = raw('A wholly separate discussion', 'd', 'separate turn')
    roots = run.components([a, b, c, d])
    assert roots[0] == roots[1] == roots[2] and roots[3] != roots[0]
    # Small repeated bypass text does not join unrelated sessions.
    assert len(set(run.components([raw('Thanks', 'x', 'thanks', 1), raw('Thanks', 'y', 'thanks', 1)]))) == 2


def test_near_duplicate_input_joins_renamed_sessions():
    text = ' '.join(f'word{i}' for i in range(100))
    rows = [raw(text, 'a', 'one'), raw(text + ' added context', 'b', 'two'),
        raw(' '.join(f'unrelated{i}' for i in range(100)), 'c', 'three')]
    roots = run.components(rows)
    assert roots[0] == roots[1] and roots[2] != roots[0]


def test_span_and_frozen_database_tampering_rejected():
    row = raw('The exact source', 'a')
    row.update(split='unassigned', id='receipt', task='raw_summary_candidate', section_kind='atomic',
        target={'summary': 'Stored summary'}, provenance={'section_receipt_sha256': 'receipt',
            'target_sha256': run.text_hash('Stored summary'), 'label_origin': 'existing_stored_summary',
            'source_bundle': 'source', 'summary_bundle': 'summary', 'store': 'store',
            'summary_file': 'summary.db', 'source_database_sha256': 'source-sha', 'summary_database_sha256': 'summary-sha'})
    databases = {('source', 'store/memory.db'): 'source-sha', ('summary', 'store/summary.db'): 'summary-sha'}
    run.validate(row, databases)
    changed = deepcopy(row)
    changed['evidence'][0]['text'] += ' changed'
    with pytest.raises(ValueError, match='character span/hash'):
        run.validate(changed, databases)
    with pytest.raises(ValueError, match='frozen artifact index'):
        run.validate(row, {**databases, ('source', 'store/memory.db'): 'different'})


def test_component_partition_is_deterministic_and_requires_independence():
    split = run.split_components(['a', 'b', 'c', 'd', 'e'])
    assert split == run.split_components(['e', 'd', 'c', 'b', 'a'])
    assert list(split.values()).count('test') == 1
    with pytest.raises(ValueError, match='at least two'):
        run.split_components(['same', 'same'])
