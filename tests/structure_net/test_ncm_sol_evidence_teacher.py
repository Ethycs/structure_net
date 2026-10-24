"""Metadata selection and historical model binding must survive the switch."""
from collections import Counter
import pytest

from experiments.structure_net import ncm_sol_evidence_teacher as run


def population():
    rows = [{'id': str(i), 'source_component': 'family' + str(i), 'bin': i % 5,
        'source_cl100k_tokens': 200 + i} for i in range(10000)]
    return rows, [r['id'] for r in rows[1117:]], rows[:1117]


def test_selection_is_fixed_fresh_and_keeps_bin_quotas():
    rows, unseen, firsts = population()
    chosen = run.select(rows, unseen, firsts)
    assert chosen == run.select(list(reversed(rows)), list(reversed(unseen)), list(reversed(firsts)))
    assert Counter(r['bin'] for r in chosen) == run.QUOTAS
    assert {r['id'] for r in chosen} <= set(unseen)
    assert not {r['source_component'] for r in chosen} & {r['source_component'] for r in firsts}
    ids = run.pilot_ids(chosen)
    assert len(ids) == len(set(ids)) == 8
    assert chosen[0]['id'] in ids and max(chosen, key=lambda r: r['source_cl100k_tokens'])['id'] in ids


def test_population_omission_overlap_or_insufficient_bin_fails():
    rows, unseen, firsts = population()
    with pytest.raises(ValueError, match='complete original'):
        run.select(rows, unseen[:-1], firsts)
    with pytest.raises(ValueError, match='partition'):
        run.select(rows, [firsts[0]['id'], *unseen[1:]], firsts)
    for row in rows[1117:]:
        row['bin'] = 0
    with pytest.raises(ValueError, match='bin quotas'):
        run.select(rows, unseen, firsts)


def test_archived_model_identity_changes_key_without_changing_live_mapping():
    lineage = run.lineage
    packet = {'case_id': 'fixed', 'source_lines': [], 'candidates': []}
    mapping = dict(run.q.JUDGES)
    a = lineage.archived_key(packet, 'primary', lineage.OLD_PRIMARY)
    b = lineage.archived_key(packet, 'primary', lineage.qualification.MODEL)
    assert a != b and run.q.JUDGES == mapping
    with pytest.raises(ValueError, match='unapproved'):
        lineage.archived_key(packet, 'secondary', lineage.qualification.MODEL)
