"""Reproducible @local user deployment; no sudo or infrastructure credentials.

WG_RELEASE points to the unpacked, manifest-verified release. User lingering
allows the enabled unit to survive logout and start at boot without loading Qwen.
The dedicated-account pipeline is a separate, later infrastructure adoption.
"""
import hashlib
import getpass
import json
import os
from pathlib import Path
import shlex

from pyinfra.operations import files, server, systemd

release = Path(os.environ['WG_RELEASE']).resolve()
home_dir = Path.home()
dest = home_dir/'.local/lib/word-games-bridges'
state = home_dir/'.local/share/word-games-bridges'
unit = home_dir/'.config/systemd/user/word-games-bridges.service'
for name, expected in json.loads((release/'manifest.json').read_text()).items():
    if hashlib.sha256((release/name).read_bytes()).hexdigest() != expected:
        raise ValueError('release_hash_mismatch: '+name)

files.directory(name='Private bridge application', path=str(dest), mode='700')
files.directory(name='Private durable bridge state', path=str(state), mode='700')
for directory in sorted(p for p in (release/'app').rglob('*') if p.is_dir()):
    files.directory(path=str(dest/directory.relative_to(release/'app')), mode='700')
for source in sorted(p for p in (release/'app').rglob('*') if p.is_file()):
    files.put(src=str(source), dest=str(dest/source.relative_to(release/'app')), mode='600')
files.put(src=str(release/'ca.crt'), dest=str(dest/'ca.crt'), mode='600')
files.put(src=str(release/'manifest.json'), dest=str(dest/'release-manifest.json'), mode='600')
files.put(src=str(Path(__file__).with_name('word-games-bridges.user.service')),
          dest=str(unit), mode='600')
server.shell(name='Keep user service manager across logout and boot', commands=[
    'loginctl --no-ask-password enable-linger '+shlex.quote(getpass.getuser())])
server.shell(name='Install locked CPU Pixi runtime', commands=[
    shlex.quote(str(home_dir/'.pixi/bin/pixi'))+' install --locked --manifest-path '+shlex.quote(str(dest/'pixi.toml'))])
systemd.daemon_reload(name='Reload user service definitions', user_mode=True)
systemd.service(name='Enable and start bridge user service', service='word-games-bridges',
                user_mode=True, running=True, enabled=True, restarted=True)
server.shell(name='Verify service health without model load', commands=[
    'curl -fsS --retry 5 --retry-connrefused --retry-delay 1 --max-time 10 http://10.158.218.141:18092/health'])
