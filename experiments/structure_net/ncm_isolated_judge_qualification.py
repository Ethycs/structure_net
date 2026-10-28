"""Qualify both unchanged judges through the isolated nonpersistent gateway."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

from experiments.structure_net import ncm_isolated_judge_transport as transport
from experiments.neural_architecture_lab import ncm_isolated_judge_gateway_v2 as neutral

previous, q = transport.previous, transport.q
BASE = previous.ROOT
ROOTS = {(role, stage): BASE/f'20261028_isolated_judge_{role}_{stage}_v1'
         for role in transport.MODELS for stage in ('qualification', 'audit')}
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-isolated-judge-qualification.md')
TEST = Path('tests/structure_net/test_ncm_isolated_judge_transport.py')
CPU = BASE/'20261028_isolated_judge_transport_CPU_v1.json'
HISTORY = BASE/'20261026_public_curriculum_epoch4_DEV_grade_v1/judgment_cache.json'


def verify(mapping):
    for name, expected in mapping.items():
        if previous.file_hash(name) != expected: raise ValueError('bound artifact changed: '+name)


def completed(root):
    receipt = q.read(root/'actual_tool_receipt.json')
    if (receipt['actual_tool_exit_code'] != 0
            or receipt['artifacts'].get(str(root/'result.json')) != previous.file_hash(root/'result.json')):
        raise ValueError('actual terminal receipt must bind exact result')
    verify(receipt['artifacts'])
    result = q.read(root/'result.json'); verify(result.get('artifacts', {}))
    return result


def context():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU/network-only qualification required')
    previous.configure()
    import tiktoken
    packets, expected = previous.profile.calibration()
    public = previous.verify_controls(packets)
    return packets, expected, tiktoken.get_encoding('cl100k_base'), public


def score(role, packets, expected, rows, encoding):
    scores, result = previous.score(packets, expected, rows, encoding)
    result['judge'] = transport.MODELS[role]
    return scores, result


def freeze(root, role, stage, paths):
    files = {Path(__file__).resolve(), PLAN.resolve(), TEST.resolve(), previous.TEST.resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            p = Path(filename).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()): files.add(p)
    proof = {'schema': 'nal.ncm-isolated-judge-qualification.v1', 'stage': stage, 'role': role,
        'config': {'judge': transport.MODELS[role], 'endpoint': transport.ENDPOINT,
            'transport_identity': transport.IDENTITY, 'rubric': previous.profile.RUBRIC,
            'controls': 50, 'required_correct': 50, 'workers': 1, 'timeout_seconds': 180, 'backend_timeout_seconds': 160,
            'max_attempts': 3, 'max_tokens': 4096, 'temperature': 0,
            'canonical_application_history_changed': False, 'first_valid_negative_is_terminal': True,
            'wire_model_alias_mapping_recorded': True, 'final_test_opened': False},
        'environment': {'executable': sys.executable, 'python': sys.version},
        'source_hashes': {str(p): previous.file_hash(p) for p in sorted(files)},
        'input_hashes': {str(p): previous.file_hash(p) for p in dict.fromkeys(
            [*paths, HISTORY, Path('pixi.lock'), Path('pyproject.toml')])}}
    proof['fingerprint'] = q.digest(proof)
    root.mkdir(); (root/'producer_source').mkdir()
    for i, p in enumerate(sorted(files)): (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    transport.write(root/'producer.json', proof)
    return proof


def finish(root, proof, value):
    verify(proof['source_hashes']); verify(proof['input_hashes'])
    value = {'schema': proof['schema'], **value, 'role': proof['role'],
        'task_accuracy': None, 'NCM_success_established': False, 'final_test_opened': False,
        'canonical_application_history_changed': False, 'producer_fingerprint': proof['fingerprint'],
        'artifacts': {str(p): previous.file_hash(p) for p in root.rglob('*.json') if 'producer_source' not in p.parts}}
    transport.write(root/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


def qualify(role):
    root = ROOTS[role, 'qualification']
    if root.exists(): raise ValueError('fresh qualification root required')
    checked = q.read(CPU); verify(checked['artifacts'])
    if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != 8:
        raise ValueError('focused transport and original control-gate tests required')
    probe = completed(neutral.ROOT)
    if (len(probe['results']) != 2 or any(not r['neutral_exact_OK'] for r in probe['results'])
            or [r['model'] for r in probe['results']] != list(transport.MODELS.values())):
        raise ValueError('both exact-model neutral probes required')
    packets, expected, encoding, public = context()
    paths = [CPU, public, neutral.SERVICE/'producer.json', neutral.ROOT/'producer.json', neutral.ROOT/'result.json', neutral.ROOT/'actual_tool_receipt.json']
    paths += [previous.profile.OLD_CALIBRATION/n for n in ('calibration_packets.json', 'calibration_expected.json')]
    if role == 'primary':
        completed(ROOTS['secondary', 'audit'])
        paths += [ROOTS['secondary', 'audit']/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
    key = transport.credential()  # Fail before creating a producer if unavailable.
    proof = freeze(root, role, 'qualification', paths)
    (root/'attempts').mkdir(); (root/'rows').mkdir()
    transport.write(root/'packets.json', packets); transport.write(root/'expected.json', expected)
    rows = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        jobs = [pool.submit(transport.request, packet, role, root, key=key) for packet in packets]
        for future in as_completed(jobs):
            row = future.result(); rows[row['case_id']] = row
            transport.write(root/'rows'/(row['case_id']+'.json'), row)
            print(json.dumps({'role': role, 'completed_packets': len(rows), 'total': len(packets), 'status': row['status']}), flush=True)
    scores, result = score(role, packets, expected, rows, encoding)
    transport.write(root/'scores.json', scores)
    finish(root, proof, {**result, 'new_generation_requests': sum(len(r['attempts']) for r in rows.values()),
        'new_application_judge_calls': 0, 'qualification_replication': True})


def audit(role):
    root, source = ROOTS[role, 'audit'], ROOTS[role, 'qualification']
    if root.exists(): raise ValueError('fresh independent raw-response replay required')
    recorded = completed(source); producer = q.read(source/'producer.json')
    verify(producer['source_hashes']); verify(producer['input_hashes'])
    packets, expected, encoding, public = context()
    if packets != q.read(source/'packets.json') or expected != q.read(source/'expected.json'):
        raise ValueError('fixed control bytes changed')
    rows = {}
    for packet in packets:
        row = q.read(source/'rows'/(packet['case_id']+'.json'))
        rows[packet['case_id']] = transport.replay_row(packet, role, source, row)
    scores, result = score(role, packets, expected, rows, encoding)
    if scores != q.read(source/'scores.json') or any(recorded.get(k) != v for k, v in result.items()):
        raise ValueError('all first decisions and scores must independently replay')
    paths = [p for p in source.rglob('*.json') if 'producer_source' not in p.parts]+[public]
    proof = freeze(root, role, 'audit', paths)
    transport.write(root/'scores.json', scores)
    finish(root, proof, {'status': 'local_judge_wire_firsts_and_controls_replayed',
        'judge': transport.MODELS[role], 'controls': 50, 'correct': result['correct'],
        'valid_requests': result['valid_requests'], 'cases': result['cases'],
        'qualification_passed': result['status'] == 'passed', 'new_generation_requests': 0,
        'new_application_judge_calls': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('stage', choices=('qualify', 'audit'))
    parser.add_argument('--role', choices=tuple(transport.MODELS), required=True); args = parser.parse_args()
    root = ROOTS[args.role, 'qualification' if args.stage == 'qualify' else 'audit']; existed = root.exists()
    try: qualify(args.role) if args.stage == 'qualify' else audit(args.role)
    except Exception as error:
        if root.exists() and not existed:
            transport.write(root/'failure.json', {'type': type(error).__name__, 'message': str(error),
                'first_judgments_preserved': True, 'automatic_retry': False})
        raise
