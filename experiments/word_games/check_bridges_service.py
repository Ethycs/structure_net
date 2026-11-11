"""HTTP deployment check for Bridges, with private credentials excluded from evidence."""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://blackbox.zt:18092')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--key-env-file', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    key = os.environ.get('LITELLM_KEY')
    if args.key_env_file:
        key = next(x.split('=',1)[1].strip().strip('"').strip("'")
                   for x in args.key_env_file.read_text().splitlines() if x.startswith('LITELLM_KEY='))
    if not key:
        raise SystemExit('LITELLM_KEY required')
    def request(path, body=None, capability=None, bearer=key, expected=200):
        headers = {'Content-Type': 'application/json'}
        if bearer:
            headers['Authorization'] = 'Bearer ' + bearer
        if capability:
            headers['X-Bridge-Capability'] = capability
        req = urllib.request.Request(args.base.rstrip('/')+path,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=1500) as response:
                assert response.status == expected
                return json.load(response)
        except urllib.error.HTTPError as error:
            assert error.code == expected, (error.code, error.read().decode())
            return json.load(error)

    checks = []
    start = time.monotonic()
    try:
        assert request('/health', bearer=None)['status'] == 'ok'
        checks.append('health_without_inference')
        request('/v1/sessions', {}, bearer='invalid-key-for-bridge-smoke', expected=403)
        checks.append('invalid_inference_key_denied')
        created = request('/v1/sessions', {'profile': 'reservations-v4'}, expected=201)
        (args.output/'private_session.json').write_text(json.dumps(created))
        sid = created['session_id']; base = '/v1/sessions/'+sid
        doer = created['doer_capability']; controller = created['controller_capability']
        request(base+'/doer', capability=controller, expected=403)
        request(base+'/controller', capability=doer, expected=403)
        checks.append('both_role_boundaries_denied')
        prepared = request(base+'/doer/prepare', {
            'request_id': 'http-prepare', 'version': 0,
            'goal': 'Reserve loan:bronze in gal:north slot N1 due 2027-01-10.',
            'context': {'scope': ['gal:north'], 'loan_ids': ['loan:bronze'],
                        'allowed_ops': ['reserve'], 'event_id': 'http-private-event'}}, doer)
        public = request(base+'/controller', capability=controller)
        text = json.dumps(public)
        assert all(word not in text for word in ['Bronze Heron','loan:bronze','gal:north','http-private-event','key_map','source_goal'])
        checks.append('controller_view_omits_source')
        proposed = request(base+'/controller/solve', {'request_id':'http-solve', 'version':prepared['version']}, controller)
        state = request(base+'/doer', capability=doer)
        assert state['state']['revision'] == 0
        checks.append('controller_proposal_does_not_commit')
        body = {'request_id':'http-apply', 'version':proposed['version']}
        applied = request(base+'/doer/apply', body, doer)
        assert applied['result']['status'] == 'committed'
        replay = request(base+'/doer/apply', body, doer)
        assert replay == applied
        state = request(base+'/doer', capability=doer)['state']
        assert state['revision'] == 1
        assert state['loans']['loan:bronze']['reservation'] == {
            'gallery_id': 'gal:north', 'slot': 'N1', 'due': '2027-01-10', 'approval_id': None}
        checks += ['source_state_matches_request', 'apply_retry_idempotent']
        result = {'passed': True, 'checks': checks, 'session_id': sid, 'base': args.base,
                  'seconds': time.monotonic()-start, 'classification': 'systems_not_accuracy'}
    except Exception as error:
        result = {'passed': False, 'checks': checks, 'base': args.base,
                  'error': type(error).__name__, 'detail': str(error)}
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
