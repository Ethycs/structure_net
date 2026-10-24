"""Exercise the writer boundary that stopped v1, without external requests."""
import json
from pathlib import Path
import pytest

from experiments.structure_net import ncm_sol_evidence_teacher_recovery as run


def test_producing_creates_case_directory_and_real_writer_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'PILOT', tmp_path / 'pilot')
    monkeypatch.setattr(run, 'FAILED_WORKFLOW', tmp_path / 'failed')
    monkeypatch.setattr(run, 'FAILED_PILOT', tmp_path / 'failed_pilot')
    monkeypatch.setattr(run.q, 'read', lambda p: {'actual_tool_exit_code': 1, 'exit_code': 1})
    def producing(root, stage, inputs):
        root.mkdir()
        (root / 'producer_source').mkdir()
        return {'config': {}, 'source_hashes': {}}
    monkeypatch.setattr(run, '_producing', producing)
    p = run.producing(run.PILOT, 'test')
    value = {'id': 'first', 'target_text': 'An unchanged original first summary.'}
    run.save_candidate(run.PILOT, value)
    assert json.loads((run.PILOT / 'cases/first.json').read_text()) == value
    assert p['config']['candidate_text_edited'] is False


def test_incomplete_original_stage_gets_no_new_attempt_allowance(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'FAILED_PILOT', tmp_path)
    destination = tmp_path / 'requests/case/draft'
    destination.mkdir(parents=True)
    (destination / 'attempt_0.json').write_text('{"valid":false}')
    with pytest.raises(ValueError, match='unfinished'):
        run.descriptor({'id': 'case'}, 'draft')
