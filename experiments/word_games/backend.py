"""Access the unchanged supplied backend without exposing it to model agents."""
import importlib
import sys

from .common import PACKAGE, read, sha


def vendor():
    root = PACKAGE / "vendor"
    manifest = read(root / "provenance.json")
    for name, expected in manifest["files"].items():
        if sha(root / name) != expected:
            raise ValueError("vendored source changed: " + name)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("bridge")
    if module.__file__ != str(root / "bridge.py"):
        raise ValueError("unexpected bridge module")
    return module


def clean(engine):
    value = engine.snapshot()
    value.pop("audit")
    return value
