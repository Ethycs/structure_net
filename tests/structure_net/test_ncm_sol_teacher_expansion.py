"""The collector stops new submissions and never repeats or silently drops firsts."""
import pytest
from threading import Barrier

from experiments.structure_net import ncm_sol_teacher_expansion as run


def test_exhausted_pipeline_stops_replacements_and_preserves_first(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'WORKERS', 1)
    (tmp_path / 'cases').mkdir()
    seen = []

    def candidate(row, *args):
        seen.append(row['id'])
        return {'id': row['id'], 'teacher_valid': False, 'capacity_fit': False}

    with pytest.raises(RuntimeError, match='inflight jobs drained'):
        run.candidate_jobs([{'id': str(i)} for i in range(3)], tmp_path, None, None, candidate)
    assert seen == ['0']
    assert run.q.read(tmp_path / 'partial_firsts.json') == [
        {'id': '0', 'teacher_valid': False, 'capacity_fit': False}]
    assert run.q.read(tmp_path / 'cases' / '0.json')['teacher_valid'] is False


def test_capacity_failure_is_retained_without_quality_reroll(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'WORKERS', 1)
    (tmp_path / 'cases').mkdir()
    rows = [{'id': str(i)} for i in range(3)]
    seen = []

    def candidate(row, *args):
        seen.append(row['id'])
        return {'id': row['id'], 'teacher_valid': True, 'capacity_fit': False}

    values = run.candidate_jobs(rows, tmp_path, None, None, candidate)
    assert seen == ['0', '1', '2']
    assert list(values) == seen
    assert len(run.q.read(tmp_path / 'partial_firsts.json')) == 3


def test_duplicate_source_rejected_before_any_request(tmp_path):
    with pytest.raises(ValueError, match='unique first-candidate'):
        run.candidate_jobs([{'id': 'a'}, {'id': 'a'}], tmp_path, None, None,
            lambda *args: pytest.fail('duplicate must be rejected before request'))


def test_inflight_firsts_are_drained_after_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'WORKERS', 2)
    (tmp_path / 'cases').mkdir()
    inflight = Barrier(2)

    def candidate(row, *args):
        inflight.wait(timeout=10)
        return {'id': row['id'], 'teacher_valid': False, 'capacity_fit': False}

    with pytest.raises(RuntimeError):
        run.candidate_jobs([{'id': str(i)} for i in range(10)], tmp_path, None, None, candidate)
    assert [r['id'] for r in run.q.read(tmp_path / 'partial_firsts.json')] == ['0', '1']
    assert sorted(p.stem for p in (tmp_path / 'cases').glob('*.json')) == ['0', '1']
