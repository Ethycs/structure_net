"""Download revision-pinned GGUFs to /data, checking space and SHA-256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / 'deployments/word_games_triton/fleet/catalog.json'
HASH_LOCK = threading.Lock()


def space_check(models):
    directory = Path('/data/models')
    required = 0
    for model in models:
        spec = model['weights']
        target = Path(spec['path'])
        partial = target.with_suffix('.gguf.part')
        present = target if target.exists() else partial
        required += max(0, spec['size'] - (present.stat().st_size if present.exists() else 0))
    # Runtime copies, partial transfers and headroom, on the destination filesystem.
    reserve = 10 * 1024**3
    free = shutil.disk_usage(directory).free
    if free < required + reserve:
        raise RuntimeError(f'Insufficient space: {free} free, {required + reserve} required')
    return {'free_bytes': free, 'download_bytes': required, 'reserve_bytes': reserve}


def download(model):
    spec = model['weights']
    target = Path(spec['path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.gguf.part')
    started = time.time()
    if not target.exists():
        url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{spec['file']}"
        print(json.dumps({'event': 'download_started', 'model': model['alias'], 'bytes': spec['size']}), flush=True)
        if not partial.exists() or partial.stat().st_size != spec['size']:
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                            '--retry', '5', '--retry-delay', '3', '--connect-timeout', '30',
                            '--continue-at', '-', '--output', str(partial), url], check=True)
        candidate = partial
    else:
        candidate = target
    if candidate.stat().st_size != spec['size']:
        raise ValueError('Incorrect download size: ' + str(candidate))
    # /data is rotational storage. Concurrent whole-file hashing turns
    # sequential reads into seeking, even when network downloads benefit
    # from concurrency.
    with HASH_LOCK, candidate.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest != spec['sha256']:
        raise ValueError('Checksum mismatch: ' + str(candidate))
    if candidate != target:
        candidate.rename(target)
    result = {'model': model['alias'], 'path': str(target), 'sha256': digest,
              'size': target.stat().st_size, 'elapsed_seconds': time.time() - started,
              'identity': {k: getattr(target.stat(), k) for k in
                           ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}}
    target.with_suffix('.verified.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'event': 'verified', **result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    models = [m for m in json.loads(CATALOG.read_text())['models'] if m.get('download')]
    print(json.dumps(space_check(models)), flush=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        result = list(pool.map(download, models))
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
