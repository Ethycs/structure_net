#!/usr/bin/env python3
"""Reject the gateway's exec fallback before it can send another model request."""
import os
from pathlib import Path
import sys

WRAPPER = Path(__file__).with_name('ncm_judge_cli_without_mcp.py').resolve()


def command(arguments):
    if not arguments or arguments[0] != 'app-server':
        raise ValueError('this qualification permits only the persistent app-server')
    return [str(WRAPPER), *arguments]


if __name__ == '__main__':
    args = command(sys.argv[1:]); os.execv(args[0], args)
