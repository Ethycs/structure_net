import pytest

from experiments.structure_net import ncm_core_cached_target_admission as run


def test_rejects_target_or_source_binding_changes():
    source = 'Complete original source'
    target = 'Stored summary'
    row = {'id': 'old', 'source_id': 's', 'group_id': 'g', 'source': source, 'target': target,
        'source_sha256': run.text_hash(source), 'target_sha256': run.text_hash(target),
        'input_tokens': 200, 'target_tokens': 3, 'split': 'train', 'spans': []}
    pair = {k: row[k] for k in ('id', 'source_id', 'group_id', 'source_sha256', 'target_sha256', 'input_tokens', 'target_tokens')}
    pair['native_target_tokens'] = 2
    cached = {'id': 'cached', 'source_component': 'component', 'alignment_sha256': 'a', 'native_prompt_sha256': 'p'}
    prepared = {**cached, 'source': source, 'source_sha256': row['source_sha256'], 'source_truncated': False}
    protected = {k: set() for k in ('families', 'groups', 'hashes', 'turns')}
    run.replay_binding(row, pair, cached, prepared, [7, 8], protected, set())
    with pytest.raises(ValueError, match='source/target'):
        run.replay_binding({**row, 'target': 'Changed'}, pair, cached, prepared, [7, 8], protected, set())
    with pytest.raises(ValueError, match='cached complete source'):
        run.replay_binding(row, pair, cached, {**prepared, 'source_component': 'other'}, [7, 8], protected, set())
    with pytest.raises(ValueError, match='protected'):
        run.replay_binding(row, pair, cached, prepared, [7, 8], {**protected, 'families': {'s'}}, set())
