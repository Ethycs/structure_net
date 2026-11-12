"""Launch the installed OpenCode harness on Qwen; no custom agent loop.

This adapter supplies configuration/credentials and records the upstream CLI's
events. OpenCode owns planning, native tool calls, execution, history and resume.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'deployments/word_games_opencode/opencode.jsonc'
RUNTIME = ROOT/'.pixi/word-games-opencode-runtime'


def environment(key):
    env = dict(os.environ)
    for name, folder in [('XDG_CONFIG_HOME','config'), ('XDG_DATA_HOME','data'),
                         ('XDG_STATE_HOME','state'), ('XDG_CACHE_HOME','cache')]:
        path = RUNTIME/folder
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env.update(OPENCODE_CONFIG=str(CONFIG), OPENCODE_DISABLE_PROJECT_CONFIG='1',
               OPENCODE_DISABLE_AUTOUPDATE='1', OPENCODE_DISABLE_MODELS_FETCH='1',
               NODE_EXTRA_CA_CERTS='/etc/docker/certs.d/10.158.218.193:5000/ca.crt',
               LITELLM_KEY=key)
    return env


def run(workspace, prompt, output, key, session=None, timeout=240):
    workspace = Path(workspace).resolve()
    if not workspace.is_dir():
        raise ValueError('workspace must exist')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = environment(key)
    version = subprocess.check_output(['opencode', '--version'], env=env, text=True, timeout=30).strip()
    if version != '1.19.5':
        raise RuntimeError('Configuration verified for OpenCode 1.19.5; revalidate installed version '+version)
    command = ['opencode', 'run', '--pure', '--format', 'json', '--agent', 'actor',
               '--model', 'wordgames/qwen3.8-35b-a3b', '--dir', str(workspace)]
    if session:
        command += ['--session', session]
    command += [prompt]
    (output/'prompt.txt').write_text(prompt)
    (output/'config.json').write_bytes(CONFIG.read_bytes())  # Contains an environment reference, not the key.
    env['OPENCODE_CONFIG'] = str((output/'config.json').resolve())
    config_hash = hashlib.sha256((output/'config.json').read_bytes()).hexdigest()
    started = time.monotonic()
    with (output/'events.jsonl').open('w') as stdout, (output/'stderr.txt').open('w') as stderr:
        process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait(timeout=10)
    events = []
    for line in (output/'events.jsonl').read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    session_ids = sorted({e['sessionID'] for e in events if e.get('sessionID')})
    tools = [{'tool': e['part']['tool'], 'status': e['part']['state']['status']}
             for e in events if e.get('type') == 'tool_use']
    usage = [e['part']['tokens'] for e in events if e.get('type') == 'step_finish']
    result = {'harness': 'OpenCode', 'harness_version': version, 'model': 'qwen3.8-35b-a3b',
              'workspace': str(workspace), 'seconds': time.monotonic()-started,
              'returncode': returncode, 'timed_out': timed_out, 'session_ids': session_ids,
              'tools': tools, 'usage_by_step': usage,
              'errors': [e for e in events if e.get('type') == 'error'],
              'config_sha256': config_hash,
              'classification': 'native_actor_transport_only_task_success_requires_independent_check'}
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--session')
    parser.add_argument('--key-env-file', type=Path, help='Read existing LITELLM_KEY without storing it in OpenCode auth config')
    args = parser.parse_args()
    os.umask(0o077)
    key = os.environ.get('LITELLM_KEY')
    if args.key_env_file:
        key = next(x.split('=',1)[1].strip().strip('"').strip("'")
                   for x in args.key_env_file.read_text().splitlines() if x.startswith('LITELLM_KEY='))
    if not key:
        parser.error('LITELLM_KEY is required')
    result = run(args.workspace, args.prompt, args.output, key, args.session)
    print(json.dumps(result, indent=2))
    if result['returncode'] or result['timed_out'] or result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
