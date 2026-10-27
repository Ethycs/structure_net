import pytest

from experiments.structure_net import ncm_public_author_lineage as lineage
from experiments.neural_architecture_lab import ncm_public_author_milestones as reporting


def test_both_branches_extend_in_order_without_changing_parent(monkeypatch):
    calls = []
    def original(cache):
        calls.append(('original', cache))
        return [*cache, 'original-firsts']
    def corrected(cache):
        calls.append(('author', cache))
        return [*cache, 'corrected-firsts']
    monkeypatch.setattr(lineage.parent, 'branch_cache', original)
    monkeypatch.setattr(lineage, 'author_cache', corrected)
    assert lineage.branch_cache(['epoch2']) == ['epoch2', 'original-firsts', 'corrected-firsts']
    assert calls == [('original', ['epoch2']), ('author', ['epoch2', 'original-firsts'])]
    assert lineage.parent.branch_cache is original


def test_install_inserts_author_once_after_original_epoch2(monkeypatch):
    monkeypatch.setattr(lineage, 'INSTALLED', False)
    monkeypatch.setattr(lineage.parent, 'install', lambda: None)
    monkeypatch.setattr(lineage.parent.original, 'append_grade',
        lambda cache, root, evaluation: [*cache, root, 'original-branch'] if root == lineage.roots(2)['grade'] else [*cache, root])
    monkeypatch.setattr(lineage.parent.original, 'lineage_paths', lambda roots: ['old'])
    monkeypatch.setattr(lineage, 'author_cache', lambda cache: [*cache, 'author-branch'])
    monkeypatch.setattr(lineage, 'branch_inputs', lambda: ['old', 'author-inputs'])
    for key in ('PLAN', 'TEST', 'PREFLIGHT'):
        monkeypatch.setattr(lineage.parent.original, key, getattr(lineage.parent.original, key))
    parent_branch = lineage.parent.branch_cache
    lineage.install(); first = lineage.parent.original.append_grade; lineage.install()
    assert first is lineage.parent.original.append_grade
    assert lineage.parent.branch_cache is parent_branch
    assert first([], lineage.roots(2)['grade'], None) == [lineage.roots(2)['grade'], 'original-branch', 'author-branch']
    for epoch in (1, 4, 8):
        assert first([], lineage.roots(epoch)['grade'], None) == [lineage.roots(epoch)['grade']]
    assert lineage.parent.original.lineage_paths([lineage.roots(2)['grade']]) == ['old', 'author-inputs']
    assert lineage.parent.original.lineage_paths([lineage.roots(1)['grade']]) == ['old']


def test_extensions_assign_each_first_to_its_actual_branch():
    values = lineage.extension_records(5190, 5263)
    assert [v['added_firsts'] for v in values] == [499, 73]
    assert [(v['before_firsts'], v['after_firsts']) for v in values] == [(4691, 5190), (5190, 5263)]
    assert all(v['included_in_application_accuracy'] is False for v in values)
    for original, author in ((4690, 5263), (5190, 5189)):
        with pytest.raises(ValueError): lineage.extension_records(original, author)


def test_author_cache_rejects_wrong_ancestor_before_loading_branch(monkeypatch):
    monkeypatch.setattr(lineage.author, 'prior_cache', lambda: {'retained': False})
    with pytest.raises(ValueError, match='exact original'):
        lineage.author_cache({'retained': True})


@pytest.mark.parametrize('epoch', [1, 2])
def test_completed_milestones_remain_historical(epoch):
    with pytest.raises(ValueError, match='historical'): lineage.configure(epoch, 1)
    with pytest.raises(ValueError, match='unchanged'): reporting.run('replay', epoch, 1)


def test_reporting_inherits_both_branches_only_at_epoch4():
    assert reporting.inherited_for_milestone(1, 4435, 5263) == 4435
    assert reporting.inherited_for_milestone(2, 4563, 5263) == 4563
    assert reporting.inherited_for_milestone(4, 4691, 5263) == 5263
    assert reporting.inherited_for_milestone(8, 5391, 5263) == 5391
    with pytest.raises(ValueError): reporting.inherited_for_milestone(4, 4690, 5263)
