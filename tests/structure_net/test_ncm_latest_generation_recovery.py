import json

import pytest

from experiments.structure_net import ncm_latest_generation_recovery as recovery


def exclusive(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, allow_nan=False)


def test_active_progress_replacement_preserves_every_latest_value(tmp_path):
    path = tmp_path/'outputs.json'; write = recovery.active_writer(exclusive, path)
    for rows in ([{'id': 'a'}], [{'id': 'a'}, {'id': 'b'}]):
        write(path, {'parent_with_source': rows, 'new_codec_only': []})
        assert json.loads(path.read_text())['parent_with_source'] == rows
        assert not path.with_name('outputs.json.pending').exists()


def test_other_and_completed_artifacts_cannot_be_overwritten(tmp_path):
    path = tmp_path/'outputs.json'; write = recovery.active_writer(exclusive, path)
    other = tmp_path/'producer.json'; write(other, {'fixed': True})
    with pytest.raises(FileExistsError): write(other, {'fixed': False})
    write(path, {'fixed': True}); (tmp_path/'result.json').write_text('{}')
    with pytest.raises(ValueError, match='immutable'): write(path, {'fixed': False})
    assert json.loads(path.read_text()) == {'fixed': True}


def test_failed_serialization_preserves_the_last_complete_progress(tmp_path):
    path = tmp_path/'outputs.json'; write = recovery.active_writer(exclusive, path)
    write(path, {'valid': [1, 2]})
    with pytest.raises(ValueError): write(path, {'invalid': float('nan')})
    assert json.loads(path.read_text()) == {'valid': [1, 2]}
    assert path.with_name('outputs.json.pending').exists()


def test_only_the_observed_no_judge_io_failure_is_admitted():
    receipt = {'actual_tool_exit_code': 1, 'new_judge_calls': 0}
    failure = {'type': 'FileExistsError', 'message': str(recovery.FAILED/'outputs.json')}
    outputs = {'parent_with_source': [{'id': 'fixed'}], 'new_codec_only': []}
    recovery.check_failure(receipt, failure, outputs)
    for bad in (dict(receipt, actual_tool_exit_code=0), dict(receipt, new_judge_calls=1)):
        with pytest.raises(ValueError): recovery.check_failure(bad, failure, outputs)
    with pytest.raises(ValueError): recovery.check_failure(receipt, dict(failure, type='RuntimeError'), outputs)
    assert recovery.ROOTS['preflight'] == recovery.latest.GENERATION['preflight']
    assert recovery.ROOTS['inference_pilot'] != recovery.FAILED
