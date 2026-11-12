"""Bubblewrap policy for the stock OpenCode harness, not an agent runtime."""
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / '.pixi/word-games-opencode-runtime'
CA = Path('/etc/docker/certs.d/10.158.218.193:5000/ca.crt')


def runtime_for(workspace):
    identity = hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:24]
    return RUNTIME / 'sandboxes' / identity


def launch(workspace, config, key, *, read_only=False):
    """Return the sandbox prefix and minimal environment. Fail if unavailable.

    The network is shared for LiteLLM. Writable host mounts are the requested
    workspace and that workspace's OpenCode history/cache only. The temporary
    sandbox root, /tmp and /dev disappear when the process exits.
    """
    workspace, config = Path(workspace).resolve(), Path(config).resolve()
    home = Path.home().resolve()
    if not workspace.is_dir() or workspace == home or workspace in home.parents:
        raise ValueError('Choose one task workspace, not home or a filesystem root')
    if workspace.is_relative_to(RUNTIME):
        raise ValueError('The internal session store cannot be a task workspace')
    if not config.is_file():
        raise ValueError('OpenCode config snapshot must exist')
    if not Path('/usr/bin/bwrap').is_file():
        raise RuntimeError('bubblewrap is required; no unsandboxed fallback')
    # The snap launcher needs host services; use its already-installed ELF.
    binary = Path('/snap/opencode/current/bin/opencode').resolve()
    if not binary.is_file():
        raise RuntimeError('Installed OpenCode binary is missing')
    runtime = runtime_for(workspace)
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    for folder in ('config', 'data', 'state', 'cache'):
        (runtime / folder).mkdir(exist_ok=True, mode=0o700)
    # OpenCode creates this directory even for --version. Precreate it so the
    # configuration mount can remain read-only for the entire child lifetime.
    (runtime / 'config/opencode').mkdir(exist_ok=True, mode=0o700)
    ignore = runtime / 'config/opencode/.gitignore'
    if not ignore.exists():
        ignore.write_text('node_modules\npackage.json\npackage-lock.json\nbun.lock\n.gitignore')
    prefix = [
        '/usr/bin/bwrap', '--unshare-all', '--share-net', '--new-session',
        '--die-with-parent', '--cap-drop', 'ALL',
        '--ro-bind', '/usr', '/usr',
        '--symlink', 'usr/bin', '/bin', '--symlink', 'usr/sbin', '/sbin',
        '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib64', '/lib64',
        '--ro-bind', '/etc', '/etc',
        '--ro-bind', '/run/systemd/resolve', '/run/systemd/resolve',
        '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
        '--ro-bind' if read_only else '--bind', str(workspace), str(workspace),
    ]
    # If the whole repository is the task workspace, hide all other actor
    # histories and mask the writable alias to this run's config snapshot.
    if RUNTIME.is_relative_to(workspace):
        prefix += ['--tmpfs', str(RUNTIME), '--remount-ro', str(RUNTIME)]
    if config.is_relative_to(workspace):
        prefix += ['--ro-bind', str(config), str(config)]
    prefix += [
        '--ro-bind', str(binary), '/run/word-games/opencode',
        '--ro-bind', str(config), '/run/word-games/config.json',
        '--ro-bind', str(CA), '/run/word-games/ca.crt',
        '--ro-bind', str(runtime / 'config'), '/run/word-games/config-home',
    ]
    for folder in ('data', 'state', 'cache'):
        prefix += ['--bind', str(runtime / folder), '/run/word-games/' + folder]
    prefix += ['--chdir', str(workspace)]
    # Never inherit subscription credentials, agent sockets or unrelated keys.
    env = {
        'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'TMPDIR': '/tmp',
        'HOME': str(home), 'USER': os.environ.get('USER', home.name),
        'XDG_CONFIG_HOME': '/run/word-games/config-home',
        'XDG_DATA_HOME': '/run/word-games/data',
        'XDG_STATE_HOME': '/run/word-games/state',
        'XDG_CACHE_HOME': '/run/word-games/cache',
        'OPENCODE_CONFIG': '/run/word-games/config.json',
        'OPENCODE_DISABLE_PROJECT_CONFIG': '1',
        'OPENCODE_DISABLE_AUTOUPDATE': '1', 'OPENCODE_DISABLE_MODELS_FETCH': '1',
        'NODE_EXTRA_CA_CERTS': '/run/word-games/ca.crt', 'LITELLM_KEY': key,
    }
    return prefix, env
