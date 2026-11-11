"""Small live deployment gate; not a benchmark or a task accuracy estimate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .client import Client
from .service import Service, packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ca', default='/etc/docker/certs.d/10.158.218.193:5000/ca.crt')
    parser.add_argument('--key-env-file', type=Path, help='Development only; read LITELLM_KEY without logging it')
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    key = os.environ.get('LITELLM_KEY')
    if args.key_env_file:
        key = next(x.split('=', 1)[1].strip().strip('"').strip("'")
                   for x in args.key_env_file.read_text().splitlines() if x.startswith('LITELLM_KEY='))
    if not key:
        raise SystemExit('LITELLM_KEY required')
    client = Client(lambda: key, ca=args.ca)
    plan = {'classification': 'systems_lifecycle_not_accuracy_estimate',
            'model': client.model, 'deployment_id': client.deployment_id,
            'checks': ['fresh_mapping', 'source_isolation', 'reserve', 'transfer', 'cancel',
                       'checkpoint_reopen', 'v4_recursive_checkpoint', 'resume_annotation'],
            'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in Path(__file__).parent.rglob('*')
                              if p.is_file() and p.suffix in ('.py', '.json', '.txt')}}
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    service = Service(args.output/'private.sqlite')
    rows = []
    started = time.monotonic()
    try:
        created = service.create(client, client.caller(key), {})
        (args.output/'private_session.json').write_text(packed(created))
        sid = created['session_id']
        print(json.dumps({'stage': 'created', 'target': created['target'],
                          'seconds': time.monotonic()-started}), flush=True)
        version = 0
        goals = [
            ('reserve', 'Reserve loan:bronze in gal:north slot N1, due 2027-01-10.',
             {'gallery_id': 'gal:north', 'slot': 'N1', 'due': '2027-01-10', 'approval_id': None}),
            ('transfer', 'Transfer loan:bronze to gal:archive slot A1, retaining due 2027-01-10.',
             {'gallery_id': 'gal:archive', 'slot': 'A1', 'due': '2027-01-10', 'approval_id': None}),
            ('cancel', 'Cancel the reservation for loan:bronze, returning it home.', None),
        ]

        def run_turn(name, goal):
            nonlocal version
            for role, action, extra in [
                ('doer', 'prepare', {'goal': goal, 'context': {
                    'scope': ['gal:north', 'gal:archive'], 'loan_ids': ['loan:bronze'],
                    'allowed_ops': ['reserve','transfer','cancel','annotate'], 'event_id': name}}),
                ('controller', 'solve', {}), ('doer', 'apply', {})]:
                result = service.mutate(sid, created[role+'_capability'], role, action,
                                       {'request_id': name+'-'+action, 'version': version, **extra}, client)
                version = result['version']
                if action == 'prepare':
                    public = service.inspect(sid, created['controller_capability'], 'controller')
                    text = packed(public)
                    assert 'Bronze Heron' not in text and 'loan:bronze' not in text
                    assert 'source_goal' not in text and 'key_map' not in text
                if action == 'apply':
                    assert result['result']['status'] == 'committed', result['result']
                    return result

        for name, goal, expected in goals:
            start = time.monotonic()
            run_turn(name, goal)
            snapshot = service.get(sid)['engine']['snapshot']
            assert snapshot['loans']['loan:bronze']['reservation'] == expected
            row = {'stage': name, 'passed': True, 'seconds': time.monotonic()-start,
                   'source_revision': snapshot['revision']}
            rows.append(row); print(json.dumps(row), flush=True)
        before = service.get(sid)['engine']
        service.db.close()
        service = Service(args.output/'private.sqlite')
        assert service.get(sid)['engine'] == before
        start = time.monotonic()
        response = service.mutate(sid, created['doer_capability'], 'doer', 'retranslate',
                                  {'request_id': 'recursive-1', 'version': version}, client)
        version = response['version']
        assert service.get(sid)['engine'] == before
        rows.append({'stage': 'recursive_checkpoint', 'passed': True,
                     'seconds': time.monotonic()-start, 'generation': response['generation']})
        print(json.dumps(rows[-1]), flush=True)
        run_turn('resume', 'Append exactly the note "Bridge resume check." to loan:bronze. Leave its placement unchanged.')
        snapshot = service.get(sid)['engine']['snapshot']
        assert snapshot['loans']['loan:bronze']['notes'] == ['Bridge resume check.']
        assert snapshot['loans']['loan:bronze']['reservation'] is None
        rows.append({'stage': 'translated_resume', 'passed': True, 'source_revision': snapshot['revision']})
        result = {'classification': plan['classification'], 'passed': True, 'checks': rows,
                  'seconds': time.monotonic()-started}
    except Exception as error:
        result = {'classification': plan['classification'], 'passed': False, 'checks': rows,
                  'error': type(error).__name__, 'detail': str(error), 'seconds': time.monotonic()-started}
    calls = [json.loads(row[0]) for row in service.db.execute('SELECT record FROM calls')]
    result['model_calls'] = []
    for call in calls:
        if 'raw_response' in call:
            data = json.loads(call['raw_response'])
            result['model_calls'].append({'stage': call['stage'], 'seconds': call['seconds'],
                'usage': data.get('usage'), 'model': data.get('model'),
                'deployment_id': call['deployment_id']})
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'model_calls'}), flush=True)
    service.db.close()
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
