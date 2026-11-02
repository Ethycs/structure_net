"""Regular repository installation and model-only control contracts."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments.word_games_triton import regular_install as installer, service


def test_load_is_idempotent_and_never_starts_a_container(monkeypatch):
    monkeypatch.setattr(service, 'ready', Mock(return_value=True))
    monkeypatch.setattr(service, 'verify_installed', Mock())
    api, process = Mock(), Mock()
    monkeypatch.setattr(service, 'api', api)
    monkeypatch.setattr(service.subprocess, 'check_output', process)
    assert service.control('up')['already_loaded']
    api.assert_not_called()
    process.assert_not_called()


def test_down_unloads_only_its_model_and_keeps_shared_server(monkeypatch):
    monkeypatch.setattr(service, 'ready', lambda path: path == '/v2/health/ready')
    api, wait = Mock(return_value={'status': 200}), Mock()
    monkeypatch.setattr(service, 'api', api)
    monkeypatch.setattr(service, 'wait_native_exit', wait)
    result = service.control('down')
    assert result['server_ready'] and not result['model_ready']
    api.assert_called_once_with('unload')
    wait.assert_called_once()


def test_regular_backend_confines_gpu_order_to_native_child(monkeypatch, tmp_path):
    monkeypatch.setitem(__import__('sys').modules, 'triton_python_backend_utils', SimpleNamespace())
    path = Path('deployments/word_games_triton/regular/models/word-games-qwen38-native/1/model.py')
    spec = importlib.util.spec_from_file_location('regular_native_backend', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'parent-order')
    child = Mock()
    child.poll.return_value = None
    popen = Mock(return_value=child)
    monkeypatch.setattr(module.subprocess, 'Popen', popen)
    model = module.TritonPythonModel()
    model.call = Mock(return_value={})
    model.initialize({})
    runtime = json.loads(path.with_name('runtime.json').read_text())
    command = popen.call_args.args[0]
    assert command[4:] == runtime['native_args']
    assert popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'] == ','.join(runtime['gpu_uuids'])
    assert module.os.environ['CUDA_VISIBLE_DEVICES'] == 'parent-order'
    assert str(popen.call_args.kwargs['cwd']).endswith('runtime-assets/native')
    assert command[0].endswith('runtime-assets/sysroot/ld-linux-x86-64.so.2')
    model.log.close()
    model.http.close()


@pytest.fixture
def package(monkeypatch, tmp_path):
    release = tmp_path / 'release'
    source = release / 'models' / installer.MODEL
    (source / '1').mkdir(parents=True)
    (source / '1/model.py').write_text('# fixture\n')
    weight = tmp_path / 'weights.gguf'
    weight.write_bytes(b'weights')
    spec = {'profile': 'fixture', 'repository_files': {'1/model.py': installer.sha(source / '1/model.py')},
            'weights': {'path': str(weight), 'sha256': installer.sha(weight)}}
    for key in installer.COMPONENTS:
        directory = tmp_path / key
        directory.mkdir()
        (directory / 'lib.so.1').write_bytes(key.encode())
        (directory / 'lib.so').symlink_to('lib.so.1')
        spec[key] = {'path': str(directory), 'files': {'lib.so.1': installer.sha(directory / 'lib.so.1')},
                     'symlinks': {'lib.so': 'lib.so.1'}}
    (release / 'release.lock.json').write_text(json.dumps(spec))
    monkeypatch.setattr(installer, 'ROOT', tmp_path)
    monkeypatch.setattr(installer, 'RELEASE', release)
    repository = tmp_path / 'models/_triton'
    repository.mkdir(parents=True)
    (repository / 'existing-model').mkdir()
    (repository / 'existing-model/untouched').write_text('preserve')
    return repository, spec


def test_install_copies_dependencies_without_touching_other_models(package):
    repository, _ = package
    assert installer.install(repository)['verified_files'] == 4
    assert installer.install(repository)['already_installed']
    assert (repository / 'existing-model/untouched').read_text() == 'preserve'
    library = repository / installer.MODEL / '1/runtime-assets/native/lib.so.1'
    library.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='pinned file'):
        installer.verify_installed(repository)


def test_incomplete_install_never_publishes_a_model(package):
    repository, spec = package
    (Path(spec['cuda']['path']) / 'lib.so.1').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='pinned file'):
        installer.install(repository)
    assert not (repository / installer.MODEL).exists()


def test_regular_profile_changes_only_weight_path_and_launch_metadata():
    root = Path('deployments/word_games_triton')
    original = json.loads((root / f'models/{installer.MODEL}/1/runtime.json').read_text())
    regular = json.loads((root / f'regular/models/{installer.MODEL}/1/runtime.json').read_text())
    args = regular['native_args'][:]
    assert args[args.index('-m')+1].startswith('/model-weights/')
    args[args.index('-m')+1] = '/model.gguf'
    assert args == original['native_args']
