import json
import sys

import pytest

from experiments.structure_net import ncm_generated_history_workflow as workflow


def test_nonzero_child_exit_cannot_be_hidden_by_a_passing_result(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'DEST', tmp_path)
    monkeypatch.setattr(workflow.study.data, 'verify', lambda p: None)
    (tmp_path / 'stage_receipts').mkdir()
    destination = tmp_path / 'child'
    code = ("import pathlib,sys,json; p=pathlib.Path(sys.argv[1]); p.mkdir(); "
        "(p/'result.json').write_text(json.dumps({'status':'passed'})); sys.exit(5)")
    journal = {'stages': []}
    with pytest.raises(RuntimeError, match='observed exit'):
        workflow.run_stage('failing', [sys.executable, '-c', code, str(destination)],
            destination, journal, {}, model_worker=False)
    receipt = json.loads((tmp_path / 'stage_receipts/failing.json').read_text())
    assert receipt['exit_code'] == 5 and receipt['observer'] == 'subprocess.wait'
    assert journal['stages'][0]['exit_code'] == 5
    assert json.loads((destination / 'worker_terminal_receipt.json').read_text())['exit_code'] == 5


def test_existing_stage_cannot_be_duplicated_and_status_file_is_not_liveness(tmp_path):
    destination = tmp_path / 'already_exists'
    destination.mkdir()
    with pytest.raises(ValueError, match='freshness'):
        workflow.run_stage('must_not_launch', ['does-not-exist'], destination, {'stages': []}, {})
    (tmp_path / 'progress.json').write_text(json.dumps({'status': 'running'}))
    with pytest.raises(RuntimeError, match='upstream terminal'):
        workflow.wait_for_existing(tmp_path, None, lambda: False)
