"""Pyinfra deployment from an unpacked release; run with --sudo on blackbox.

The caller supplies WG_RELEASE (local unpacked release directory). No model load,
Triton/container restart, credential transfer or GPU operation occurs here.
"""
import os
from pathlib import Path

from pyinfra.operations import files, server, systemd

RELEASE = Path(os.environ['WG_RELEASE']).resolve()
DEST = '/opt/word-games-bridges'

server.group(name='Bridge service group', group='word-games-bridges', system=True)
server.user(name='Bridge service account', user='word-games-bridges', group='word-games-bridges',
            system=True, shell='/usr/sbin/nologin', create_home=False)
files.directory(name='Bridge application', path=DEST, user='root', group='root', mode='755')
for directory in sorted(p for p in (RELEASE/'app').rglob('*') if p.is_dir()):
    files.directory(path=DEST+'/'+str(directory.relative_to(RELEASE/'app')),
                    user='root', group='root', mode='755')
for source in sorted(p for p in (RELEASE/'app').rglob('*') if p.is_file()):
    files.put(src=str(source), dest=DEST+'/'+str(source.relative_to(RELEASE/'app')),
              user='root', group='root', mode='644')
files.directory(name='Bridge CA directory', path='/etc/word-games-bridges',
                user='root', group='root', mode='755')
files.put(name='Public gateway CA', src=str(RELEASE/'ca.crt'),
          dest='/etc/word-games-bridges/ca.crt', user='root', group='root', mode='644')
# Deployment credentials, if needed to reach the host, stay with the deployer.
# The existing Pixi executable on blackbox resolves this locked CPU-only runtime.
server.shell(name='Install locked Pixi runtime', commands=[
    'cd /opt/word-games-bridges && /home/rabbit/.pixi/bin/pixi install --locked'])
unit = files.put(name='Bridge systemd unit', src=str(RELEASE/'word-games-bridges.service'),
                 dest='/etc/systemd/system/word-games-bridges.service',
                 user='root', group='root', mode='644')
systemd.daemon_reload(name='Reload unit definitions')
systemd.service(name='Start managed bridge', service='word-games-bridges',
                running=True, enabled=True, restarted=True)
server.shell(name='Verify bridge readiness without loading Qwen', commands=[
    'curl -fsS --max-time 10 http://10.158.218.141:18092/health'])
