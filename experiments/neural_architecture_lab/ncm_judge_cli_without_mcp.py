#!/usr/bin/env python3
"""Disable configured MCP servers in this judge subprocess only.

The gateway's mcp_servers={} override merges with user configuration and does
not remove its server entries. Explicit enabled=false flags retain all other
Codex settings and leave the user's config file and running agents untouched.
"""
import os
from pathlib import Path
import sys
import tomllib

CLI = Path('/home/rabbit/llm_endpoint/.pixi/envs/default/lib/node_modules/@openai/codex/node_modules/'
           '@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex')
CONFIG = Path('/home/rabbit/.codex/config.toml')
SERVERS = ('central_dev', 'servant-chat')


def command(arguments, configuration):
    if not arguments or arguments[0] not in ('app-server', 'exec', 'mcp'):
        raise ValueError('only the judge gateway and read-only MCP listing are supported')
    if set(configuration.get('mcp_servers', {})) != set(SERVERS):
        raise ValueError('configured MCP population changed; inspect before launching judge')
    flags = [item for name in SERVERS for item in ('-c', f'mcp_servers.{name}.enabled=false')]
    # Later overrides win. The gateway itself passes mcp_servers={}, which
    # otherwise clears our earlier overrides before the user config is merged.
    return [str(CLI), *arguments, *flags]


def main():
    if not CLI.is_file(): raise ValueError('fixed existing Codex executable required')
    arguments = command(sys.argv[1:], tomllib.loads(CONFIG.read_text()))
    os.execv(arguments[0], arguments)


if __name__ == '__main__': main()
