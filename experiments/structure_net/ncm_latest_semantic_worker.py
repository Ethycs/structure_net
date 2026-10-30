"""Persistent CPU execution with hashes shared only inside the live worker.

Independent audits are deliberately unavailable through this service.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import traceback
import uuid

from experiments.structure_net import ncm_latest_recovered_evaluation as latest

BASE = latest.BASE
ROOT = BASE/'20261029_latest_semantic_worker_v1'
RUNTIME = Path('/tmp/structure-net-ncm-latest-semantic-20261029-v1')
SOCKET = RUNTIME/'worker.sock'
PLAN = Path('docs/07 - Status Reports/2026-10-29_ncm-persistent-latest-semantics.md')
TEST = Path('tests/structure_net/test_ncm_latest_semantic_worker.py')
CPU = BASE/'20261029_latest_semantic_worker_CPU_v1.json'
STAGES = ('probe', 'warmup', 'export_pilot', 'export_outputs', 'judge_pilot', 'grade', 'comparison', 'shutdown')
SHARED_HASHES = {}
OriginalProofs = latest.cache.ProcessProofs


class WorkerProofs(OriginalProofs):
    def __init__(self):
        super().__init__()
        self.cache = SHARED_HASHES


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate(request):
    if set(request) != {'stage', 'arm', 'request_id'}:
        raise ValueError('exact CPU request fields required')
    stage, arm, identity = request['stage'], request['arm'], request['request_id']
    if stage not in STAGES or not isinstance(identity, str) or len(identity) != 32:
        raise ValueError('known CPU stage and unique hexadecimal request identity required')
    try:
        int(identity, 16)
    except ValueError as error:
        raise ValueError('hexadecimal request identity required') from error
    if stage in ('probe', 'warmup', 'comparison', 'shutdown'):
        if arm is not None:
            raise ValueError('this CPU stage has no readout arm')
    elif arm not in latest.semantic.ARMS:
        raise ValueError('fixed readout arm required')
    return stage, arm


def destination(request):
    stage, arm = validate(request)
    if stage in ('probe', 'warmup', 'shutdown'):
        return ROOT/(stage+'_result.json')
    if stage == 'comparison':
        return latest.COMPARISON/'result.json'
    target = {'export_pilot': 'inference_pilot', 'export_outputs': 'outputs'}.get(stage, stage)
    return latest.SEMANTIC[arm][target]/'result.json'


def execute(request, invocation, callback):
    """One attempt, with an explicit completion digest or preserved exception."""
    response = {'request': request, 'worker_invocation': invocation, 'worker_pid': os.getpid()}
    try:
        validate(request)
        callback(request)
        result = destination(request)
        response.update(exit_code=0, result=str(result), result_sha256=sha(result))
    except Exception as error:
        response.update(exit_code=1, exception=type(error).__name__, message=str(error))
        traceback.print_exc()
    return response


def check_response(request, response, start):
    if (response.get('request') != request
            or response.get('worker_invocation') != start['invocation']
            or response.get('worker_pid') != start['pid']):
        raise ValueError('response must identify this exact request and worker invocation')
    if response.get('exit_code') != 0:
        raise RuntimeError('worker stage failed: '+str(response.get('message')))
    result = destination(request)
    if response.get('result') != str(result) or response.get('result_sha256') != sha(result):
        raise ValueError('completed worker response must bind the actual result bytes')


def install(start):
    checked = json.loads(CPU.read_text())
    if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != 9:
        raise ValueError('six worker and three original hash-cache tests required')
    for path, expected in checked['artifacts'].items():
        if sha(path) != expected:
            raise ValueError('qualified worker source changed')
    if SHARED_HASHES:
        raise ValueError('the live worker must start with no measured hashes')
    latest.cache.ProcessProofs = WorkerProofs
    latest.install()
    original_write = latest.previous.write_proof
    def write_proof(root, proof, files):
        value = {**proof, 'config': {**proof['config'], 'persistent_CPU_execution': {
            'worker_invocation': start['invocation'], 'worker_PID': start['pid'],
            'hashes_shared_only_within_live_process': True,
            'cache_loaded_from_disk': False, 'independent_audits_use_separate_processes': True}},
            'input_hashes': dict(proof['input_hashes'])}
        for path in (Path(__file__).resolve(), PLAN, TEST, CPU, ROOT/'worker_start.json'):
            name, digest = str(path), latest.data.sha(path)
            if name in value['input_hashes'] and value['input_hashes'][name] != digest:
                raise ValueError('persistent execution conflicts with a producer input')
            value['input_hashes'][name] = digest
        return original_write(root, value, files)
    latest.previous.write_proof = write_proof


def require_probe():
    observed = json.loads((ROOT/'probe_actual_tool_receipt.json').read_text())
    if observed['actual_tool_exit_code'] != 0:
        raise ValueError('successful actual startup-probe client receipt required')
    for name in ('probe_result.json', 'worker_start.json'):
        path = ROOT/name
        if observed['artifacts'].get(str(path)) != sha(path):
            raise ValueError('startup probe must bind the current result and worker identity')
    probe = json.loads((ROOT/'probe_result.json').read_text())
    start = json.loads((ROOT/'worker_start.json').read_text())
    if (probe['status'] != 'private_CPU_worker_roundtrip_passed'
            or probe['worker_pid'] != os.getpid() or start['pid'] != os.getpid()
            or probe['measured_hashes'] != 0 or probe['CUDA_initialized']
            or probe['new_model_calls'] != 0 or probe['new_judge_requests'] != 0):
        raise ValueError('completed empty-cache neutral probe from this live worker required')


def run_request(request):
    stage, arm = validate(request)
    if stage not in ('probe', 'shutdown'):
        require_probe()
    if stage == 'probe':
        import torch
        if SHARED_HASHES or torch.cuda.is_initialized():
            raise ValueError('neutral startup probe requires an empty cache and no CUDA initialization')
        write(destination(request), {'status': 'private_CPU_worker_roundtrip_passed',
            'worker_pid': os.getpid(), 'measured_hashes': 0, 'CUDA_initialized': False,
            'new_model_calls': 0, 'new_judge_requests': 0, 'task_accuracy': None,
            'systems_only_not_quality_evidence': True, 'final_test_opened': False})
    elif stage == 'warmup':
        if destination(request).exists():
            raise ValueError('warmup runs once; no reroll')
        _, _, _, _, judgments, trained, _, cohorts, _, _ = latest.generated.context()
        write(destination(request), {'status': 'CPU_training_admission_warmed',
            'cases': sum(map(len, cohorts.values())), 'canonical_first_judgments': len(judgments),
            'checkpoint_sha256': trained['checkpoint_sha256'], 'new_model_calls': 0,
            'new_judge_requests': 0, 'task_accuracy': None, 'final_test_opened': False,
            'systems_only_not_quality_evidence': True})
    elif stage == 'shutdown':
        write(destination(request), {'status': 'CPU_worker_shutdown_requested',
                                     'new_model_calls': 0, 'new_judge_requests': 0})
    else:
        argv = sys.argv
        try:
            sys.argv = [latest.__file__, stage, *(['--arm', arm] if arm else [])]
            latest.main()
        finally:
            sys.argv = argv


def serve():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('PIXI_ENVIRONMENT_NAME') != 'ncm':
        raise ValueError('CPU-only locked Pixi worker required')
    if ROOT.exists() or RUNTIME.exists():
        raise ValueError('fresh append-only worker invocation and private socket directory required')
    start = {'invocation': uuid.uuid4().hex, 'pid': os.getpid(), 'socket': str(SOCKET),
             'executable': sys.executable, 'source_sha256': sha(__file__),
             'initial_measured_hashes': 0, 'persistent_audits_allowed': False}
    install(start)
    ROOT.mkdir(); (ROOT/'requests').mkdir(); RUNTIME.mkdir(mode=0o700)
    write(ROOT/'worker_start.json', start)
    seen = set()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(SOCKET)); os.chmod(SOCKET, 0o600); server.listen(1)
        print(json.dumps({'status': 'CPU_worker_ready', **start}), flush=True)
        while True:
            with server.accept()[0] as connection:
                with connection.makefile('rb') as stream:
                    raw = stream.readline(65537)
                if len(raw) > 65536 or not raw.endswith(b'\n'):
                    raise ValueError('bounded newline-terminated request required')
                request = json.loads(raw); validate(request)
                if request['request_id'] in seen:
                    raise ValueError('duplicate request identity; no automatic retry')
                seen.add(request['request_id'])
                write(ROOT/'requests'/(request['request_id']+'_request.json'), request)
                response = execute(request, start['invocation'], run_request)
                write(ROOT/'requests'/(request['request_id']+'_response.json'), response)
                connection.sendall((json.dumps(response, allow_nan=False)+'\n').encode())
            if request['stage'] == 'shutdown' and response['exit_code'] == 0:
                break
    SOCKET.unlink(); RUNTIME.rmdir()


def client(stage, arm):
    request = {'stage': stage, 'arm': arm, 'request_id': uuid.uuid4().hex}
    validate(request)
    start = json.loads((ROOT/'worker_start.json').read_text())
    if start['socket'] != str(SOCKET) or start['source_sha256'] != sha(__file__):
        raise ValueError('current exact worker source and private endpoint required')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(SOCKET))
        connection.sendall((json.dumps(request)+'\n').encode())
        with connection.makefile('rb') as stream:
            response = json.loads(stream.readline(65537))
    check_response(request, response, start)
    print(json.dumps(response), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('serve', *STAGES))
    parser.add_argument('--arm', choices=latest.semantic.ARMS)
    args = parser.parse_args()
    if args.action == 'serve':
        if args.arm is not None:
            parser.error('serve has no arm')
        serve()
    else:
        client(args.action, args.arm)
