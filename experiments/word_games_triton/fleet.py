"""Install, verify and control the on-demand Triton model catalog."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import urllib.error
import urllib.request

from . import regular_install as regular

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'deployments/word_games_triton/fleet'
BASE = ROOT / 'deployments/word_games_triton/regular/models/word-games-qwen38-native'
REPOSITORY = Path('/data/models/_triton')
ROUTER = 'word-games-model-router'
URL = 'http://127.0.0.1:8000'


def catalog():
    return json.loads((SOURCE / 'catalog.json').read_text())


def entry(alias):
    return next(m for m in catalog()['models'] if alias in (m['alias'], m.get('public_alias')))


def api(path, body=None):
    request = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        data = response.read()
        return json.loads(data) if data else {}


def ready(name):
    try:
        with urllib.request.urlopen(URL + f'/v2/models/{name}/ready', timeout=5) as response:
            return response.status == 200
    except urllib.error.HTTPError as error:
        if error.code in (400, 404, 503):
            return False
        raise


def runtime(model):
    result = json.loads((BASE / '1/runtime.json').read_text())
    args = result['native_args']
    changes = {'-m': model['weights']['path'].replace('/data/models/', '/model-weights/', 1),
               '--alias': model['alias'], '--port': str(model['native_port'])}
    if model['alias'] == 'dolphin-x1-trinity-nano':
        result['gpu_uuids'] = ['GPU-7b1db3cb-ec9c-beb3-e7db-826c2543edf9']
        changes.update({'--device': 'CUDA0', '--tensor-split': '1'})
    elif model['alias'] == 'orion-26b-a4b':
        changes['--tensor-split'] = '6,10,16'
    for flag, value in changes.items():
        args[args.index(flag) + 1] = value
    result.update(native_port=model['native_port'], public_chat_port=None,
                  profile=model['alias'] + '-' + model['quantization'].lower() + '-fa-q8kv-8192')
    return result


def install(alias):
    model = entry(alias) if alias != ROUTER else None
    name = model['triton_model'] if model else ROUTER
    if ready(name):
        raise RuntimeError('Unload before installing: ' + name)
    lock = regular.lock()
    weight_identity = None
    if model:
        weights = model['weights']
        path = Path(weights['path'])
        weight_identity = regular.identity(path)
        expected = {'identity': weight_identity, 'sha256': weights['sha256']}
        existing = REPOSITORY / name / 'installation.json'
        cached = json.loads(existing.read_text()).get('weights') if existing.exists() else None
        downloaded = path.with_suffix('.verified.json')
        if cached != expected and downloaded.exists():
            receipt = json.loads(downloaded.read_text())
            cached = {k: receipt.get(k) for k in expected}
        if (cached != expected and regular.sha(path) != weights['sha256']) or regular.identity(path) != weight_identity:
            raise ValueError('Missing or changed checkpoint: ' + str(path))
    with tempfile.TemporaryDirectory(prefix='.word-games-fleet-', dir=REPOSITORY.parent) as temporary:
        stage = Path(temporary) / name
        version = stage / '1'
        version.mkdir(parents=True)
        if model:
            config = (BASE / 'config.pbtxt').read_text().replace(regular.MODEL, name)
            (stage / 'config.pbtxt').write_text(config)
            if name == regular.MODEL:
                shutil.copy2(BASE / '1/model.py', version / 'model.py')
                # Preserve its legacy JSON research contract and release profile.
                shutil.copy2(BASE / '1/runtime.json', version / 'runtime.json')
            else:
                shutil.copy2(SOURCE / 'native_model.py', version / 'model.py')
                shutil.copy2(BASE / '1/model.py', version / 'native_base.py')
                (version / 'runtime.json').write_text(json.dumps(runtime(model), indent=2) + '\n')
            shutil.copy2(BASE / '1/chat_proxy.py', version / 'chat_proxy.py')
            for component, directory in regular.COMPONENTS.items():
                spec = lock[component]
                origin = ROOT / spec['path']
                regular.verify_tree(origin, spec['files'], spec['symlinks'])
                target = version / 'runtime-assets' / directory
                target.mkdir(parents=True)
                for file in spec['files']:
                    shutil.copy2(origin / file, target / file)
                for file, link in spec['symlinks'].items():
                    (target / file).symlink_to(link)
        else:
            (stage / 'config.pbtxt').write_text('''name: "word-games-model-router"
backend: "python"
max_batch_size: 0
input [{name: "control" data_type: TYPE_STRING dims: [1]}]
output [{name: "result" data_type: TYPE_STRING dims: [1]}]
instance_group [{count: 1 kind: KIND_CPU}]
parameters: {key: "EXECUTION_ENV_PATH" value: {string_value: "/opt/pixi-gemma4/.pixi/envs/default"}}
''')
            shutil.copy2(SOURCE / 'router_model.py', version / 'model.py')
            shutil.copy2(SOURCE / 'router.py', version / 'router.py')
            shutil.copy2(SOURCE / 'catalog.json', version / 'catalog.json')
            shutil.copy2(BASE / '1/chat_proxy.py', version / 'chat_proxy.py')
        files = {str(p.relative_to(stage)): regular.sha(p) for p in stage.rglob('*') if p.is_file() and not p.is_symlink()}
        links = {str(p.relative_to(stage)): str(p.readlink()) for p in stage.rglob('*') if p.is_symlink()}
        source_paths = [Path(__file__), SOURCE / 'catalog.json', BASE / '1/chat_proxy.py']
        source_paths += ([BASE / 'config.pbtxt', BASE / '1/model.py', BASE / '1/runtime.json',
                          SOURCE / 'native_model.py'] if model else
                         [SOURCE / 'router.py', SOURCE / 'router_model.py'])
        receipt = {'model': name, 'files': files, 'symlinks': links,
                   'source_files': {str(p.relative_to(ROOT)): regular.sha(p) for p in source_paths},
                   'weights': {'identity': weight_identity, 'sha256': model['weights']['sha256']} if model else None,
                   'release_lock_sha256': regular.sha(regular.RELEASE / 'release.lock.json'),
                   'catalog_sha256': regular.sha(SOURCE / 'catalog.json')}
        (stage / 'installation.json').write_text(json.dumps(receipt, indent=2) + '\n')
        target = REPOSITORY / name
        backup = Path(temporary) / 'previous'
        if target.exists():
            target.rename(backup)
        try:
            stage.rename(target)
        except BaseException:
            if backup.exists():
                backup.rename(target)
            raise
    return {'installed': name, 'files': len(files)}


def verify(alias):
    model = entry(alias) if alias != ROUTER else None
    name = model['triton_model'] if model else ROUTER
    directory = REPOSITORY / name
    receipt = json.loads((directory / 'installation.json').read_text())
    regular.verify_tree(directory, receipt['files'], receipt['symlinks'])
    if 'source_files' not in receipt:
        raise ValueError('Reinstall to record deployment source fingerprints')
    regular.verify_tree(ROOT, receipt['source_files'])
    if receipt['release_lock_sha256'] != regular.sha(regular.RELEASE / 'release.lock.json'):
        raise ValueError('Native release changed; reinstall to verify it')
    if receipt['catalog_sha256'] != regular.sha(SOURCE / 'catalog.json'):
        raise ValueError('Installed model uses a different catalog')
    if model:
        path = Path(model['weights']['path'])
        if regular.identity(path) != receipt['weights']['identity']:
            raise ValueError('Checkpoint identity changed; reinstall to verify the checksum')
    return {'verified': name, 'files': len(receipt['files'])}


def control(action, alias=None):
    if not ready(ROUTER):
        if action == 'status':
            return {'router_ready': False, 'models': [
                {'alias': m['alias'], 'triton_model': m['triton_model'],
                 'ready': ready(m['triton_model'])} for m in catalog()['models']]}
        verify(ROUTER)
        api(f'/v2/repository/models/{ROUTER}/load', {})
    if action == 'start':
        action = 'status'
    if action == 'load':
        model = entry(alias or 'qwen3.8-35b-a3b')
        verify(model['alias'])
        alias = model['alias']
    response = api(f'/v2/models/{ROUTER}/infer', {
        'inputs': [{'name': 'control', 'shape': [1], 'datatype': 'BYTES',
                    'data': [json.dumps({'action': action, 'model': alias})]}]})
    return {'router_ready': True, **json.loads(response['outputs'][0]['data'][0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'verify', 'start', 'load', 'unload', 'status', 'stop'))
    parser.add_argument('model', nargs='?')
    args = parser.parse_args()
    if args.action in ('install', 'verify'):
        names = [args.model] if args.model else [m['alias'] for m in catalog()['models']] + [ROUTER]
        for name in names:
            print(json.dumps((install if args.action == 'install' else verify)(name)), flush=True)
    else:
        print(json.dumps(control('unload' if args.action == 'stop' else args.action, args.model), indent=2))
        if args.action == 'stop':
            api(f'/v2/repository/models/{ROUTER}/unload', {})


if __name__ == '__main__':
    main()
