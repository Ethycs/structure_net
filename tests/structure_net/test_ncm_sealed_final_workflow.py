import json
import sys

import pytest

from experiments.structure_net import ncm_sealed_final_workflow as workflow


def test_final_workflow_preserves_real_child_failure_despite_passing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'DEST', tmp_path)
    monkeypatch.setattr(workflow.final.base, 'verify', lambda _: None)
    (tmp_path / 'stage_receipts').mkdir()
    destination = tmp_path / 'worker'
    command = ("import pathlib,sys,json; p=pathlib.Path(sys.argv[1]); p.mkdir(); "
        "(p/'result.json').write_text(json.dumps({'quality_gate':True})); sys.exit(6)")
    journal = {'stages': []}
    with pytest.raises(RuntimeError, match='observed exit'):
        workflow.run_stage('failed_final_stage', [sys.executable, '-c', command, str(destination)],
            destination, journal, {}, model_worker=False)
    assert journal['stages'][0]['exit_code'] == 6
    assert json.loads((tmp_path / 'stage_receipts/failed_final_stage.json').read_text())['exit_code'] == 6
    assert json.loads((destination / 'worker_terminal_receipt.json').read_text())['exit_code'] == 6


def test_existing_final_stage_is_never_duplicated(tmp_path):
    destination = tmp_path / 'existing'
    destination.mkdir()
    with pytest.raises(ValueError, match='freshness'):
        workflow.run_stage('must_not_launch', ['does-not-exist'], destination, {'stages': []}, {})
