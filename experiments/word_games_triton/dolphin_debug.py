"""Run one explicitly selected Dolphin diagnostic cell and collect its evidence.

Launch from the repository root using Pixi. GPU cells require word-games-down;
this harness never changes the installed model or starts another model afterward.
"""
import argparse
import json
from pathlib import Path
import subprocess
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--cases', type=int, default=12)
    parser.add_argument('--max-tokens', type=int, default=64)
    args = parser.parse_args()
    if not args.cell.replace('_', '').isalnum():
        parser.error('cell must contain only letters, digits and underscores')
    if not 1 <= args.cases <= 12 or not 1 <= args.max_tokens <= 64:
        parser.error('initial cells require 1–12 cases and 1–64 output tokens')
    if args.backend == 'gpu':
        request = urllib.request.Request('http://localhost:8000/v2/repository/index',
                                         data=b'{}', headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            ready = [r['name'] for r in json.load(response)
                     if r.get('state') == 'READY' and r['name'].startswith('word-games')]
        if ready:
            raise RuntimeError(f'Drain and stop WORD_GAMES before GPU diagnostics: {ready}')
    args.root.mkdir(parents=True, exist_ok=True)
    config = {key: getattr(args, key) for key in ('cell', 'weights', 'backend', 'cases', 'max_tokens')}
    with (args.root / (args.cell + '_request.json')).open('x') as out:
        json.dump(config, out, indent=2)
    worker = Path(__file__).with_name('dolphin_debug_worker.py')
    remote = '/tmp/word-games-dolphin-debug-worker.py'
    subprocess.run(['docker', 'cp', str(worker), 'triton:' + remote], check=True)
    command = ['docker', 'exec', '-i', 'triton',
               '/opt/pixi-gemma4/.pixi/envs/default/bin/python', remote]
    try:
        with (args.root / (args.cell + '_stdout.log')).open('x') as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            process.stdin.write(json.dumps(config))
            process.stdin.close()
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    finally:
        subprocess.run(['docker', 'cp', 'triton:/tmp/word-games-dolphin-debug/' + args.cell,
                        str(args.root / args.cell)], check=True)
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    main()
