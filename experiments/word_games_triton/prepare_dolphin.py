"""Rebuild and verify the selected Dolphin mixed-precision GGUF from pinned inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .download_catalog import download, space_check

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / 'deployments/word_games_triton/fleet/dolphin-mixed-precision.json'


def identity(path):
    s = path.stat()
    return {k: getattr(s, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def verified(spec):
    path = Path(spec['path'])
    marker = path.with_suffix('.verified.json')
    if not path.exists() or not marker.exists():
        return False
    r = json.loads(marker.read_text())
    return r.get('sha256') == spec['sha256'] and r.get('identity') == identity(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    recipe = json.loads(RECIPE.read_text())
    spec = recipe['output']
    target = Path(spec['path'])
    if not target.exists():
        sources = [{'alias': 'dolphin-' + name, 'weights': recipe[name]}
                   for name in ('base', 'donor')]
        print(json.dumps(space_check([*sources, {'weights': spec}])), flush=True)
        for model in sources:
            if not verified(model['weights']):
                download(model)
        target.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-m', 'experiments.word_games_triton.mix_dolphin_tensors',
                        '--base', recipe['base']['path'], '--donor', recipe['donor']['path'],
                        '--group', recipe['group'], '--output', str(target),
                        '--receipt', str(args.receipt.with_suffix('.tensors.json'))],
                       cwd=ROOT, check=True)
    before = identity(target)
    if not verified(spec):
        with target.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != spec['sha256'] or before != identity(target):
            raise ValueError('Derived Dolphin checksum or file identity mismatch')
    if target.stat().st_size != spec['size']:
        raise ValueError('Derived Dolphin size mismatch')
    result = {'path': str(target), 'sha256': spec['sha256'], 'identity': before,
              'size': spec['size'], 'recipe_sha256': hashlib.sha256(RECIPE.read_bytes()).hexdigest()}
    target.with_suffix('.verified.json').write_text(json.dumps(result, indent=2) + '\n')
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x') as out:
        json.dump(result, out, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
