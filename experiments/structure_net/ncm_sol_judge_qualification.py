"""Qualify the explicitly authorized replacement judge on the unchanged controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from experiments.structure_net import ncm_budget128_profile as profile
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify

q = profile.q
MODEL = 'codex_sdk/gpt-5.6-sol'
DEST = ROOT / '20261024_budget128_judge_sol_qualification_v1'
AUDIT = ROOT / '20261024_budget128_judge_sol_qualification_audit_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-sol-reviewer-amendment.md')
TEST = Path('tests/structure_net/test_ncm_sol_judge_qualification.py')
CANARY = ROOT / '20261024_replacement_reviewer_canaries_v1/codex_sdk_gpt-5.6-sol.response.json'


def configure():
    profile.install()
    # Requests execute inside the original module. Change only this new process.
    q.JUDGES['primary'] = MODEL
    q.original.JUDGES['primary'] = MODEL


def score(packets, expected, rows, encoding):
    candidates = [c for p in packets for c in p['candidates']]
    if (len(expected) != 50 or len(candidates) != 50
            or {c['candidate_id'] for c in candidates} != set(expected)
            or set(rows) != {p['case_id'] for p in packets}):
        raise ValueError('all fixed50 controls and packets required')
    scores = []
    for packet in packets:
        row = rows[packet['case_id']]
        grades = {g['candidate_id']: g for g in row.get('evaluations', [])}
        for candidate in packet['candidates']:
            name = candidate['candidate_id']
            if candidate['text'] != expected[name]['text']:
                raise ValueError('control candidate text changed')
            tokens = len(encoding.encode(candidate['text'], disallowed_special=()))
            actual = q.passes(grades[name], candidate['text'], tokens) if row['status'] == 'valid' else None
            scores.append({'candidate_id': name, 'expected': expected[name]['expected_pass'],
                'actual': actual, 'correct': actual == expected[name]['expected_pass'] if actual is not None else False,
                'tokens': tokens, 'evaluation': grades.get(name)})
    valid = sum(r['status'] == 'valid' for r in rows.values())
    correct = sum(s['correct'] for s in scores)
    result = {'status': 'passed' if valid == len(packets) and correct == 50 else 'failed_qualification',
        'cases': len(packets), 'valid_requests': valid, 'controls': 50, 'correct': correct,
        'judge': MODEL, 'profile': 'budget128-v1', 'NCM_success_established': False,
        'task_accuracy': None, 'independent_human_gold': False}
    return scores, result


def verify_controls(packets):
    from experiments.structure_net.ncm_frozen_body_public_data import PUBLIC_FILE, PUBLIC_SHA256
    if file_hash(PUBLIC_FILE) != PUBLIC_SHA256:
        raise ValueError('public calibration corpus changed')
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for v in value.values():
                yield from strings(v)
        elif isinstance(value, list):
            for v in value:
                yield from strings(v)
    texts = set(strings(json.loads(PUBLIC_FILE.read_text())))
    if any('\n'.join(s['text'] for s in p['source_lines']) not in texts for p in packets):
        raise ValueError('control source bytes not present in public corpus')
    return PUBLIC_FILE


def qualify():
    configure()
    profile.labels.dependencies()
    import tiktoken
    encoding = tiktoken.get_encoding('cl100k_base')
    packets, expected = profile.calibration()
    public = verify_controls(packets)
    canary = q.read(CANARY)
    if canary.get('http_status') != 200 or canary.get('reported_model') != MODEL or not canary.get('ready'):
        raise ValueError('actual exact-model availability canary required')
    files = [Path(__file__), PLAN, TEST, Path(profile.__file__), profile.PLAN]
    inputs = [profile.OLD_CALIBRATION / n for n in ('calibration_packets.json', 'calibration_expected.json')]
    p = freeze(DEST, 'replacement_judge_controls_only_not_student_quality', [*inputs, *files, public, CANARY],
        {'judge': MODEL, 'rubric': profile.RUBRIC, 'controls': 50, 'workers': 4,
         'max_attempts': 3, 'semantic_retries': False, 'independent_human_gold': False,
         'reason': 'user-authorized replacement of quota-exhausted Sonnet',
         'old_judgments_preserved': True, 'sealed_final_source_files_read': False})
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (DEST / 'producer_source' / f'sol_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(DEST / 'producer.json', p)
    (DEST / 'attempts').mkdir()
    (DEST / 'rows').mkdir()
    write_json(DEST / 'packets.json', packets)
    write_json(DEST / 'expected.json', expected)
    rows = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(q.request, packet, 'primary', DEST) for packet in packets]
        for future in as_completed(jobs):
            row = future.result()
            rows[row['case_id']] = row
            write_json(DEST / 'rows' / (row['case_id'] + '.json'), row)
            print(json.dumps({'completed_control_packets': len(rows), 'total': len(packets), 'status': row['status']}), flush=True)
    scores, result = score(packets, expected, rows, encoding)
    result['producer_fingerprint'] = p['fingerprint']
    write_json(DEST / 'scores.json', scores)
    verify(p)
    write_json(DEST / 'result.json', result)
    print(json.dumps(result), flush=True)


def audit():
    configure()
    profile.labels.dependencies()
    import tiktoken
    encoding = tiktoken.get_encoding('cl100k_base')
    if AUDIT.exists() or q.read(DEST / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('fresh audit after actual whole-tool completion required')
    producer = q.read(DEST / 'producer.json')
    verify(producer)
    packets, expected = profile.calibration()
    if packets != q.read(DEST / 'packets.json') or expected != q.read(DEST / 'expected.json'):
        raise ValueError('original fixed control bytes changed')
    verify_controls(packets)
    rows = {}
    for packet in packets:
        row = q.read(DEST / 'rows' / (packet['case_id'] + '.json'))
        wanted = {'model': MODEL, 'temperature': 0, 'max_tokens': 4096,
            'messages': [{'role': 'system', 'content': profile.RUBRIC},
                {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False)}]}
        found, attempts = None, row['attempts']
        if not 1 <= len(attempts) <= 3:
            raise ValueError('one-to-three attempts required')
        for i, name in enumerate(attempts):
            path = DEST / 'attempts' / f"{packet['case_id']}_{i}.json"
            if str(path) != name or found is not None:
                raise ValueError('attempt order or first valid chronology changed')
            raw = q.read(path)
            if raw['request'] != wanted or raw['request_sha256'] != q.digest(wanted) or raw['attempt'] != i:
                raise ValueError('raw request/rubric/model binding changed')
            grades = None
            try:
                if raw.get('http_status') != 200 or raw['response'].get('model') != MODEL:
                    raise ValueError('invalid transport/model')
                grades = q.parse(raw['response']['choices'][0]['message']['content'], packet)
            except (ValueError, KeyError, TypeError, IndexError):
                pass
            if raw['valid'] != (grades is not None):
                raise ValueError('recorded validity differs from raw replay')
            if grades is not None:
                found = grades
        expected_row = {'case_id': packet['case_id'], 'packet_sha256': q.digest(packet),
            'status': 'valid' if found is not None else 'invalid', 'judge': MODEL, 'attempts': attempts}
        if found is not None:
            expected_row['evaluations'] = found
        if row != expected_row:
            raise ValueError('first judgment or preserved failure changed')
        rows[packet['case_id']] = row
    scores, result = score(packets, expected, rows, encoding)
    result['producer_fingerprint'] = producer['fingerprint']
    if scores != q.read(DEST / 'scores.json') or result != q.read(DEST / 'result.json'):
        raise ValueError('control score replay differs')
    artifact_paths = [p for p in DEST.rglob('*') if p.is_file() and 'producer_source' not in p.parts]
    AUDIT.mkdir()
    write_json(AUDIT / 'result.json', {'status': 'replacement_judge_firsts_and_scores_replayed',
        'qualification_passed': result['status'] == 'passed', 'controls': 50, 'correct': result['correct'],
        'judge': MODEL, 'task_accuracy': None, 'artifacts': {str(p): file_hash(p) for p in artifact_paths}})
    print(json.dumps({'status': 'replacement_judge_firsts_and_scores_replayed', 'correct': result['correct']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('qualify', 'audit'))
    args = parser.parse_args()
    {'qualify': qualify, 'audit': audit}[args.stage]()
