"""Exercise the real OS boundary without a model call or real credentials."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from experiments.word_games import opencode_sandbox as sandbox


PROBE = '''
import errno, json, os
from pathlib import Path
workspace, outside, host_pid = __import__('sys').argv[1:]
root = Path(workspace)
assert (root/'allowed').read_text() == 'inside'
(root/'written').write_text('sandbox write')
assert not Path(outside).exists()
assert not (root/'escape').exists()
for hidden in ('.ssh', '.codex', '.claude', '.config'):
    assert not (Path.home()/hidden).exists(), hidden
for hidden in ('/data', '/opt', '/home/infra', '/proc/'+host_pid):
    assert not Path(hidden).exists(), hidden
for readonly in ('/run/word-games/config.json', '/run/word-games/config-home/new'):
    try:
        Path(readonly).write_text('should fail')
    except OSError as e:
        assert e.errno in (errno.EROFS, errno.EACCES), e
    else:
        raise AssertionError('writable config')
assert 'UNRELATED_TEST_SECRET' not in os.environ
assert 'SSH_AUTH_SOCK' not in os.environ
Path('/run/word-games/state/persistence-check').write_text('survives')
print(json.dumps({'mount_ns':str(Path('/proc/self/ns/mnt').readlink()),
 'pid_ns':str(Path('/proc/self/ns/pid').readlink()),'passed':True}))
'''


def test_real_filesystem_boundary_and_persistent_state(tmp_path, monkeypatch):
    workspace = tmp_path/'workspace'; workspace.mkdir()
    (workspace/'allowed').write_text('inside')
    outside = tmp_path/'outside'; outside.write_text('host only')
    (workspace/'escape').symlink_to(outside)
    config = tmp_path/'config.json'; config.write_text('{}')
    monkeypatch.setenv('UNRELATED_TEST_SECRET', 'canary')
    monkeypatch.setenv('SSH_AUTH_SOCK', '/host-only-agent.sock')
    monkeypatch.setattr(sandbox, 'RUNTIME', tmp_path/'runtimes')
    command, env = sandbox.launch(workspace, config, 'test-placeholder')
    result = subprocess.run(command+['/usr/bin/python3','-c',PROBE,str(workspace),str(outside),str(os.getpid())],
                            env=env, check=True, capture_output=True, text=True, timeout=30)
    proof = json.loads(result.stdout)
    assert proof['mount_ns'] != str(Path('/proc/self/ns/mnt').readlink())
    assert proof['pid_ns'] != str(Path('/proc/self/ns/pid').readlink())
    assert (workspace/'written').read_text() == 'sandbox write'
    assert outside.read_text() == 'host only'
    assert config.read_text() == '{}'
    saved = sandbox.runtime_for(workspace)/'state/persistence-check'
    assert saved.read_text() == 'survives'
    # A new sandbox invocation can see its saved state.
    resumed = subprocess.check_output(command+['/usr/bin/cat','/run/word-games/state/persistence-check'],env=env,text=True)
    assert resumed == 'survives'
    version = subprocess.check_output(command+['/run/word-games/opencode','--version'],env=env,text=True,timeout=30)
    assert version.strip() == '1.19.5'
    # Version alone does not initialize configuration; exercise that path too.
    subprocess.run(command+['/run/word-games/opencode','debug','config'],env=env,
                   capture_output=True, check=True, text=True, timeout=30)
    other = tmp_path/'other'; other.mkdir()
    command, env = sandbox.launch(other, config, 'test-placeholder')
    assert subprocess.run(command+['/usr/bin/test','-e','/run/word-games/state/persistence-check'],env=env).returncode == 1


@pytest.mark.parametrize('workspace', [Path.home(), Path('/'), Path('/home')])
def test_refuses_unbounded_workspace(workspace, tmp_path):
    config = tmp_path/'config.json'; config.write_text('{}')
    with pytest.raises(ValueError):
        sandbox.launch(workspace, config, 'placeholder')


def test_repository_workspace_cannot_reach_other_histories_or_edit_config_alias(tmp_path, monkeypatch):
    workspace = tmp_path/'repo'; workspace.mkdir()
    runtime = workspace/'.private-actor-runtime'; runtime.mkdir()
    (runtime/'old-session').write_text('other actor history')
    config = workspace/'config.json'; config.write_text('{}')
    monkeypatch.setattr(sandbox, 'RUNTIME', runtime)
    command, env = sandbox.launch(workspace, config, 'placeholder')
    code = '''
import errno, sys
from pathlib import Path
assert not Path(sys.argv[1]).exists()
assert not Path('/dev/nvidia0').exists()
try:
    Path(sys.argv[2]).write_text('overwrite via host path')
except OSError as e:
    assert e.errno in (errno.EROFS, errno.EACCES)
else:
    raise AssertionError('Writable config alias')
'''
    subprocess.run(command+['/usr/bin/python3','-c',code,str(runtime/'old-session'),str(config)],
                   env=env, check=True, capture_output=True, timeout=30)
    assert config.read_text() == '{}'
