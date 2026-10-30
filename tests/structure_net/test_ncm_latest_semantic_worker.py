import json
import os
import time

import pytest

from experiments.structure_net import ncm_latest_semantic_worker as worker


def request(stage='grade', arm='new_codec_only'):
    return {'stage': stage, 'arm': arm, 'request_id': 'a'*32}


def test_cpu_whitelist_rejects_generation_and_every_persistent_audit():
    assert worker.validate(request()) == ('grade', 'new_codec_only')
    for stage in ('audit', 'preflight', 'inference_pilot', 'outputs', 'train'):
        with pytest.raises(ValueError): worker.validate(request(stage))
    for value in (request('comparison'), request(arm=None), dict(request(), request_id='bad'),
                  dict(request(), unexpected='field'), request(arm='different')):
        with pytest.raises(ValueError): worker.validate(value)
    assert worker.validate(request('comparison', None)) == ('comparison', None)
    assert worker.validate(request('probe', None)) == ('probe', None)


def test_only_measured_hashes_share_and_file_changes_are_still_detected(tmp_path, monkeypatch):
    worker.SHARED_HASHES.clear()
    # Exercise the original descriptor/stat implementation, with old-enough files.
    clock = time.time_ns()
    from experiments.structure_net import ncm_recovered_training_inputs as original
    monkeypatch.setattr(original.time, 'time_ns', lambda: clock+10_000_000_000)
    path = tmp_path/'artifact'; path.write_bytes(b'a'*(1024**2))
    first, second = worker.WorkerProofs(), worker.WorkerProofs()
    assert not first.cache and first.inputs is not second.inputs
    initial = first.bind(path)
    assert second.sha(path) == initial and second.cache is first.cache
    assert not second.inputs
    path.write_bytes(b'b'*(1024**2))
    assert second.sha(path) != initial
    with pytest.raises(ValueError): first.bind(path, initial)
    assert not worker.OriginalProofs().cache  # Fresh ordinary audit object.
    worker.SHARED_HASHES.clear()


def test_completion_requires_exact_worker_request_result_and_current_bytes(tmp_path, monkeypatch):
    result = tmp_path/'result.json'; result.write_text('{"status":"completed"}')
    monkeypatch.setattr(worker, 'destination', lambda req: result)
    req = request(); start = {'invocation': 'fixed', 'pid': 123}
    reply = {'request': req, 'worker_invocation': 'fixed', 'worker_pid': 123,
             'exit_code': 0, 'result': str(result), 'result_sha256': worker.sha(result)}
    worker.check_response(req, reply, start)
    for changed in (dict(reply, worker_pid=124), dict(reply, worker_invocation='other'),
                    dict(reply, request=request('judge_pilot')), dict(reply, result='elsewhere')):
        with pytest.raises(ValueError): worker.check_response(req, changed, start)
    result.write_text('{"status":"changed"}')
    with pytest.raises(ValueError): worker.check_response(req, reply, start)


def test_failed_execution_is_returned_once_without_a_completion_claim():
    calls = []
    def fail(req):
        calls.append(req)
        raise ValueError('preserved failure')
    req = request(); reply = worker.execute(req, 'worker', fail)
    assert calls == [req] and reply['exit_code'] == 1
    assert 'result_sha256' not in reply and reply['message'] == 'preserved failure'
    with pytest.raises(RuntimeError):
        worker.check_response(req, reply, {'invocation': 'worker', 'pid': os.getpid()})


def test_successful_completion_observes_returned_function_and_result(tmp_path, monkeypatch):
    path = tmp_path/'result.json'; calls = []
    monkeypatch.setattr(worker, 'destination', lambda req: path)
    def finish(req):
        calls.append(req)
        path.write_text(json.dumps({'status': 'complete', 'cases': 128}))
    req = request(); reply = worker.execute(req, 'worker', finish)
    assert calls == [req] and reply['exit_code'] == 0
    worker.check_response(req, reply, {'invocation': 'worker', 'pid': os.getpid()})


def test_no_warmup_or_grading_without_an_actual_current_worker_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    probe = {'status': 'private_CPU_worker_roundtrip_passed', 'worker_pid': os.getpid(),
             'measured_hashes': 0, 'CUDA_initialized': False, 'new_model_calls': 0,
             'new_judge_requests': 0}
    (tmp_path/'probe_result.json').write_text(json.dumps(probe))
    (tmp_path/'worker_start.json').write_text(json.dumps({'pid': os.getpid()}))
    receipt = {'actual_tool_exit_code': 0, 'artifacts': {str(tmp_path/name): worker.sha(tmp_path/name)
               for name in ('probe_result.json', 'worker_start.json')}}
    path = tmp_path/'probe_actual_tool_receipt.json'; path.write_text(json.dumps(receipt))
    worker.require_probe()
    path.write_text(json.dumps({**receipt, 'actual_tool_exit_code': 1}))
    with pytest.raises(ValueError, match='successful actual'): worker.require_probe()
    path.write_text(json.dumps(receipt))
    (tmp_path/'probe_result.json').write_text(json.dumps({**probe, 'CUDA_initialized': True}))
    with pytest.raises(ValueError, match='bind'): worker.require_probe()
    receipt['artifacts'][str(tmp_path/'probe_result.json')] = worker.sha(tmp_path/'probe_result.json')
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='empty-cache'): worker.require_probe()
