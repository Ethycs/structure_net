"""Build a credential-free, hash-manifested Bridges deployment archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    app = args.output/'app'
    source = Path('experiments/word_games/bridges')
    for path in source.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.txt','.json') and '__pycache__' not in path.parts:
            target = app/path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    for path in ['experiments/__init__.py', 'experiments/word_games/__init__.py']:
        (app/path).write_text('')
    deploy = Path(__file__).parent
    for name in ['pixi.toml', 'pixi.lock']:
        shutil.copyfile(deploy/name, app/name)
    for name in ['deploy.py', 'deploy_user.py', 'word-games-bridges.service',
                 'word-games-bridges.user.service', 'README.md', 'validation.json']:
        shutil.copyfile(deploy/name, args.output/name)
    shutil.copyfile('/etc/docker/certs.d/10.158.218.193:5000/ca.crt', args.output/'ca.crt')
    manifest = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in args.output.rglob('*') if p.is_file()}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    archive = args.output.with_suffix('.tar.gz')
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, 'w:gz') as handle:
        handle.add(args.output, arcname='word-games-bridges-release', filter=lambda item: normalized(item))
    print(json.dumps({'archive': str(archive), 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                      'files': len(manifest)}))


def normalized(item):
    item.uid = item.gid = 0
    item.uname = item.gname = 'root'
    item.mode = 0o755 if item.isdir() else 0o644
    return item


if __name__ == '__main__':
    main()
