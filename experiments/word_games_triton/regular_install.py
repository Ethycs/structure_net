"""Install the pinned native backend as an ordinary entry in Triton's repository."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'deployments/word_games_triton/regular'
MODEL = 'word-games-qwen38-native'
REPOSITORY = Path('/data/models/_triton')
COMPONENTS = {'native_runtime': 'native', 'native_sysroot': 'sysroot', 'cuda': 'cuda'}


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def lock():
    return json.loads((RELEASE / 'release.lock.json').read_text())


def identity(path):
    stat = path.stat()
    return {key: getattr(stat, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def verify_tree(directory, files, links=None):
    for name, expected in files.items():
        path = directory / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Missing or changed pinned file: ' + str(path))
    for name, target in (links or {}).items():
        path = directory / name
        if not path.is_symlink() or str(path.readlink()) != target or not path.exists():
            raise ValueError('Missing or changed pinned symlink: ' + str(path))


def verify_installed(repository=REPOSITORY):
    spec = lock()
    model = repository / MODEL
    verify_tree(model, spec['repository_files'])
    count = len(spec['repository_files'])
    for component, name in COMPONENTS.items():
        entry = spec[component]
        verify_tree(model / '1/runtime-assets' / name, entry['files'], entry['symlinks'])
        count += len(entry['files'])
    weight = Path(spec['weights']['path'])
    receipt = json.loads((model / 'installation.json').read_text())
    before = identity(weight)
    if receipt.get('weights') != {'identity': before, 'sha256': spec['weights']['sha256']}:
        if sha(weight) != spec['weights']['sha256'] or identity(weight) != before:
            raise ValueError('Checkpoint changed')
    return {'model': MODEL, 'repository': str(repository), 'verified_files': count,
            'profile': spec['profile'], 'weights_sha256': spec['weights']['sha256']}


def install(repository=REPOSITORY):
    spec = lock()
    source = RELEASE / 'models' / MODEL
    verify_tree(source, spec['repository_files'])
    destination = repository / MODEL
    if destination.exists():
        return {**verify_installed(repository), 'already_installed': True}
    repository.mkdir(parents=True, exist_ok=True)
    weight = Path(spec['weights']['path'])
    before = identity(weight)
    expected = {'identity': before, 'sha256': spec['weights']['sha256']}
    receipt = ROOT / '.pixi/word-games-triton-service/weights_verified.json'
    cached = json.loads(receipt.read_text()) if receipt.exists() else None
    if cached != expected and sha(weight) != spec['weights']['sha256']:
        raise ValueError('Checkpoint changed')
    if identity(weight) != before:
        raise ValueError('Checkpoint changed during verification')
    # Stage outside the repository so Triton never sees a partial installation.
    with tempfile.TemporaryDirectory(prefix='.word-games-stage-', dir=repository.parent) as temporary:
        stage = Path(temporary) / MODEL
        shutil.copytree(source, stage)
        for component, name in COMPONENTS.items():
            entry = spec[component]
            source_assets = ROOT / entry['path']
            verify_tree(source_assets, entry['files'], entry['symlinks'])
            target = stage / '1/runtime-assets' / name
            target.mkdir(parents=True)
            for file in entry['files']:
                shutil.copy2(source_assets / file, target / file)
            for file, link in entry['symlinks'].items():
                (target / file).symlink_to(link)
            verify_tree(target, entry['files'], entry['symlinks'])
        (stage / 'installation.json').write_text(json.dumps({
            'weights': expected, 'release_lock_sha256': sha(RELEASE / 'release.lock.json'),
            'source': str(RELEASE), 'native_assets_copied': True}, indent=2) + '\n')
        stage.rename(destination)
    return verify_installed(repository)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'verify'))
    parser.add_argument('--repository', type=Path, default=REPOSITORY)
    args = parser.parse_args()
    result = install(args.repository) if args.action == 'install' else verify_installed(args.repository)
    print(json.dumps(result, indent=2))
