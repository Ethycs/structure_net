"""Preserve raw local gateway replies while validating their exact model alias."""
from copy import deepcopy
import json
from pathlib import Path
import time

import requests

from experiments.structure_net import ncm_sol_judge_qualification as previous

q = previous.q
ENDPOINT = 'http://127.0.0.1:8092/v1/chat/completions'
IDENTITY = 'ncm-isolated8092-judge-transport-v1'
MODELS = {'secondary': 'codex_sdk/gpt-5.6-terra', 'primary': 'codex_sdk/gpt-5.6-sol'}
from experiments.neural_architecture_lab.ncm_isolated_judge_gateway_v2 import credential


def normalize(wire, model):
    if model not in MODELS.values() or wire.get('model') not in (model, model.removeprefix('codex_sdk/')):
        raise ValueError('returned judge model differs from the exact requested model')
    result = deepcopy(wire); result['model'] = model
    return result


def payload_for(packet, role):
    return {'model': MODELS[role], 'temperature': 0, 'max_tokens': 4096, 'timeout': 160,
        'messages': [{'role': 'system', 'content': previous.profile.RUBRIC},
            {'role': 'user', 'content': json.dumps(packet, ensure_ascii=False)}]}


def write(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def request(packet, role, root, *, post=None, key=None):
    """Original three-attempt budget; first valid answer ends requests either way."""
    previous.configure()
    payload = payload_for(packet, role); attempts = []
    session = requests.Session(); session.trust_env = False
    send = post or session.post
    auth = key if key is not None else credential()
    try:
        for number in range(3):
            path = Path(root)/'attempts'/f"{packet['case_id']}_{number}.json"
            if path.exists(): raise ValueError('attempt overwrite')
            raw = {'attempt': number, 'request': payload, 'request_sha256': q.digest(payload),
                'endpoint': ENDPOINT, 'transport_identity': IDENTITY}
            started = time.monotonic()
            try:
                response = send(ENDPOINT, headers={'Authorization': 'Bearer '+auth}, json=payload, timeout=180)
                raw.update(http_status=response.status_code, wire_response=response.json())
                # Keep the unaltered wire response, including its unprefixed name.
                raw['response'] = deepcopy(raw['wire_response'])
                response.raise_for_status()
                raw['response'] = normalize(raw['wire_response'], MODELS[role])
                raw['model_alias_mapping'] = {'wire': raw['wire_response']['model'], 'canonical': MODELS[role]}
                parsed = q.parse(raw['response']['choices'][0]['message']['content'], packet)
                raw['valid'] = True
            except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as error:
                raw.update(valid=False, error_type=type(error).__name__)
            raw['seconds'] = time.monotonic()-started
            write(path, raw); attempts.append(str(path))
            if raw['valid']:
                return {'case_id': packet['case_id'], 'packet_sha256': q.digest(packet), 'status': 'valid',
                    'judge': MODELS[role], 'evaluations': parsed, 'attempts': attempts}
        return {'case_id': packet['case_id'], 'packet_sha256': q.digest(packet), 'status': 'invalid',
            'judge': MODELS[role], 'attempts': attempts}
    finally: session.close()


def replay_row(packet, role, root, row):
    """Check both untouched wire replies and canonical first-answer chronology."""
    wanted = payload_for(packet, role); found = None; attempts = row['attempts']
    if not 1 <= len(attempts) <= 3: raise ValueError('one to three attempts required')
    for number, name in enumerate(attempts):
        path = Path(root)/'attempts'/f"{packet['case_id']}_{number}.json"
        if name != str(path) or found is not None: raise ValueError('first valid chronology changed')
        raw = q.read(path)
        if (raw['request'] != wanted or raw['request_sha256'] != q.digest(wanted)
                or raw['attempt'] != number or raw.get('endpoint') != ENDPOINT
                or raw.get('transport_identity') != IDENTITY):
            raise ValueError('fixed request and endpoint binding changed')
        parsed = None
        try:
            if raw.get('http_status') == 200:
                expected = normalize(raw['wire_response'], MODELS[role])
                if (raw.get('response') != expected or raw.get('model_alias_mapping') !=
                        {'wire': raw['wire_response']['model'], 'canonical': MODELS[role]}):
                    raise RuntimeError('wire response or declared model mapping changed')
                parsed = q.parse(expected['choices'][0]['message']['content'], packet)
        except (ValueError, KeyError, TypeError, IndexError):
            pass
        if type(raw['valid']) is not bool or raw['valid'] != (parsed is not None):
            raise ValueError('recorded validity differs from wire replay')
        if parsed is not None: found = parsed
    if found is None and len(attempts) != 3: raise ValueError('all exhausted invalid attempts must remain')
    expected_row = {'case_id': packet['case_id'], 'packet_sha256': q.digest(packet),
        'status': 'valid' if found is not None else 'invalid', 'judge': MODELS[role], 'attempts': attempts}
    if found is not None: expected_row['evaluations'] = found
    if row != expected_row: raise ValueError('first valid decision or preserved invalid row changed')
    return row
