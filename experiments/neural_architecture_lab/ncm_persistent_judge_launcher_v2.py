#!/usr/bin/env python3
"""Keep the judge's SQLite runtime separate from interactive Codex sessions."""
import json
import os
from pathlib import Path
import sys

from ncm_persistent_judge_launcher import command as persistent_command

SQLITE = Path('/data2/structure_net/ncm_judge_runtime_20261028_v3/sqlite')


def command(arguments):
    return [*persistent_command(arguments), '-c', 'sqlite_home='+json.dumps(str(SQLITE))]


if __name__ == '__main__':
    if not SQLITE.is_dir(): raise ValueError('the separately initialized SQLite runtime is required')
    args=command(sys.argv[1:]);os.execv(args[0],args)
