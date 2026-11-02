"""Load/unload WORD_GAMES through the regular shared Triton service."""
import argparse
import json
import os
import subprocess
import time
import urllib.request

from .regular_install import MODEL, verify_installed

URL = os.environ.get('WORD_GAMES_TRITON_URL', 'http://127.0.0.1:8000').rstrip('/')


def ready(path):
    try:
        with urllib.request.urlopen(URL + path, timeout=5) as response:
            return response.status == 200
    except OSError:
        return False


def status():
    return {'server_ready': ready('/v2/health/ready'),
            'model_ready': ready(f'/v2/models/{MODEL}/ready'),
            'model': MODEL, 'url': URL, 'container': 'triton'}


def api(action):
    request = urllib.request.Request(URL + f'/v2/repository/models/{MODEL}/{action}',
                                     data=b'{}', headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=300) as response:
        return {'status': response.status, 'body': response.read().decode()}


def wait_native_exit(timeout=60):
    deadline = time.monotonic() + timeout
    marker = f'/models/{MODEL}/1/runtime-assets/native/llama-server'
    while time.monotonic() < deadline:
        processes = subprocess.check_output(['docker', 'top', 'triton', '-eo', 'pid,args'], text=True)
        if not any(marker in line for line in processes.splitlines()[1:]):
            return
        time.sleep(.25)
    raise TimeoutError('Native child did not exit after unload')


def control(action):
    action = {'up': 'load', 'down': 'unload'}.get(action, action)
    if not ready('/v2/health/ready'):
        raise RuntimeError('Shared Triton is not ready at ' + URL + '; its service is managed by infrastructure')
    if action == 'load':
        verify_installed()
        if ready(f'/v2/models/{MODEL}/ready'):
            return {**status(), 'already_loaded': True}
    result = api(action)
    if action == 'unload' and URL == 'http://127.0.0.1:8000':
        wait_native_exit()
    result.update(status())
    if result['model_ready'] != (action == 'load'):
        raise RuntimeError('Triton model did not reach requested state: ' + json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('load', 'unload', 'up', 'down', 'status', 'verify'))
    args = parser.parse_args()
    if args.action == 'status':
        result = status()
    elif args.action == 'verify':
        result = verify_installed()
    else:
        result = control(args.action)
    print(json.dumps(result, indent=2))
