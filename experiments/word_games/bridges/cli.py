"""Client for the managed Bridges service; inference key stays in the environment."""
import argparse
import json
import os
from pathlib import Path
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['create', 'view', 'prepare', 'solve', 'submit', 'apply', 'retranslate'])
    parser.add_argument('--base', default='http://blackbox.zt:18092')
    parser.add_argument('--output', type=Path, help='New private directory for separate role configuration files')
    parser.add_argument('--session', type=Path, help='Only the configuration for this role')
    parser.add_argument('--body', type=Path, help='JSON body with version and unique request_id')
    args = parser.parse_args()
    key = os.environ.get('LITELLM_KEY')
    if not key:
        parser.error('Set LITELLM_KEY to an existing inference key that can access qwen3.8-35b-a3b')
    headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
    if args.command == 'create':
        if not args.output:
            parser.error('create requires --output (a new directory)')
        os.umask(0o077)
        args.output.mkdir(parents=True, exist_ok=False)
        url = args.base.rstrip('/') + '/v1/sessions'
        body = {'profile': 'reservations-v4'}
    else:
        if not args.session:
            parser.error('--session is required')
        config = json.loads(args.session.read_text())
        headers['X-Bridge-Capability'] = config['capability']
        url = config['base'].rstrip('/') + '/v1/sessions/' + config['session_id'] + '/' + config['role']
        body = None
        if args.command != 'view':
            if not args.body:
                parser.error('--body is required for a mutation (version + request_id)')
            url += '/' + args.command
            body = json.loads(args.body.read_text())
    request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                     headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=1500) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise SystemExit(f'Bridge HTTP {error.code}: {error.read().decode()}') from None
    if args.command == 'create':
        for role in ('doer', 'controller'):
            (args.output/(role+'.json')).write_text(json.dumps({
                'base': args.base, 'session_id': result['session_id'], 'role': role,
                'capability': result.pop(role+'_capability')}, indent=2)+'\n')
        result['role_configs'] = str(args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
