import pytest

from experiments.structure_net import ncm_recovered_review_lineage as lineage


def test_union_retains_firsts_and_rejects_extra_missing_or_changed():
    prior = {'old': {'passed': False}}
    current = {**prior, 'new': {'passed': True}}
    assert lineage.preserve_union(prior, current, {'new'}) == current
    for bad in ({'new': {'passed': True}}, {**current, 'extra': {}}, prior,
                {'old': {'passed': True}, 'new': {'passed': True}}):
        with pytest.raises(ValueError):
            lineage.preserve_union(prior, bad, {'new'})
    copied = lineage.preserve_union(prior, current, {'new'})
    copied['old']['passed'] = True
    assert prior['old']['passed'] is False


def test_insert_once_after_author_epoch2_only(monkeypatch):
    monkeypatch.setattr(lineage, 'INSTALLED', False)
    monkeypatch.setattr(lineage.author, 'install', lambda: None)
    original = lineage.author.parent.original
    monkeypatch.setattr(original, 'append_grade', lambda cache, root, evaluation: [*cache, root, 'author'])
    monkeypatch.setattr(original, 'lineage_paths', lambda roots: ['base'])
    for key in ('PLAN', 'TEST', 'PREFLIGHT'):
        monkeypatch.setattr(original, key, getattr(original, key))
    monkeypatch.setattr(lineage, 'extend', lambda cache: [*cache, 'labels'])
    monkeypatch.setattr(lineage, 'branch_inputs', lambda: ['base', 'labels'])
    lineage.install(); first = original.append_grade; lineage.install()
    assert first is original.append_grade
    assert first([], lineage.author.roots(2)['grade'], None)[-2:] == ['author', 'labels']
    for epoch in (1, 4, 8):
        assert first([], lineage.author.roots(epoch)['grade'], None)[-1] == 'author'
    assert original.lineage_paths([lineage.author.roots(2)['grade']]) == ['base', 'labels']
    assert original.lineage_paths([lineage.author.roots(1)['grade']]) == ['base']
