import pytest

from experiments.structure_net import ncm_evidence_teacher_bulk as run


def test_all_retained_firsts_preserved_even_with_failed_quality():
    rows = [{'id': 'bad'}, {'id': 'good'}, {'id': 'new'}]
    retained = {'bad': {'id': 'bad', 'known_quality_failed': True}, 'good': {'id': 'good'}}
    generated = {'new': {'id': 'new'}}
    values = run.combine(rows, retained, generated)
    assert values[0] is retained['bad'] and len(values) == 3
    with pytest.raises(ValueError):
        run.combine(rows, retained, {**generated, 'bad': {'id': 'bad', 'rerolled': True}})
    with pytest.raises(ValueError):
        run.combine(rows, retained, {})


def test_eight_systems_cases_exclude_all_qualified_sources_and_include_longest():
    rows = [{'id': f'{b}-{i}', 'bin': b, 'source_cl100k_tokens': b * 400 + i}
        for b in range(5) for i in range(10)]
    retained = {'0-1': {}, '4-9': {}}
    selected = run.pilot_ids(rows, retained)
    assert len(set(selected)) == 8 and not set(selected) & set(retained)
    assert '4-8' in selected
