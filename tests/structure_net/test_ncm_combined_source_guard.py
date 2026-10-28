from experiments.structure_net.ncm_combined_source_guard import SourceGroups, aliases


def row(source, component, **extra):
    return {'source_sha256': source, 'source_component': component, **extra}


def test_shared_source_propagates_blocking_across_original_components():
    groups = SourceGroups({})
    groups.add(row('a', 'first'))
    groups.add(row('a', 'second'))
    groups.add(row('b', 'second'), text_conflict=True)
    groups.add(row('c', 'independent'))
    assert groups.root('a') == groups.root('b')
    assert groups.root('a') in groups.blocked_roots()
    assert groups.root('c') not in groups.blocked_roots()


def test_sealed_metadata_conflict_blocks_without_source_text():
    groups = SourceGroups({'turns': {'reserved-turn'}})
    groups.add(row('a', 'one', spans=[{'turn_text_sha256': 'reserved-turn'}]))
    assert groups.root('a') in groups.blocked_roots()


def test_original_family_and_partial_span_aliases_are_retained():
    value = aliases(row('a', 'one', source_id='sharegpt_session_3',
        group_id='original-group', spans=[{'source_id': 'sharegpt_session_8', 'span_text_sha256': 'part'}]))
    assert ('families', 'sharegpt_session') in value
    assert ('groups', 'original-group') in value
    assert ('turns', 'part') in value
