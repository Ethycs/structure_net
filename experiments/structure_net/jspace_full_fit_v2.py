"""Identical fit, with bounded reservation-readiness waiting after GPU release."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from . import jspace_full_capture as capture
from . import jspace_full_fit as original
from .jspace_full_lease import StableStage

DEFAULT_ROOT=capture.ROOT/"data/experiments/jspace_full_compact/20261113_fit_v2"


def prepare(root,parent):
    capture.checked(parent,"capture")
    failed=original.DEFAULT_ROOT
    return original.derive_root(root,parent,[Path(__file__),Path(original.__file__),Path(__file__).with_name("jspace_full_lease.py")],
        [parent/"capture_manifest.json",failed/"plan.json",failed/"failure.json",failed/"fit/events.jsonl"],
        capture_root=str(parent.resolve()),amendment="Same algorithm; pause up to 45 seconds for active lease memory readiness; no GPU work during wait")


def fit(root):
    with patch.object(original.capture,"Stage",StableStage):return original.fit(root)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","fit"))
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT);p.add_argument("--capture-root",type=Path,default=capture.DEFAULT_ROOT)
    a=p.parse_args();print(json.dumps(prepare(a.root,a.capture_root) if a.stage=="prepare" else fit(a.root),allow_nan=False),flush=True)
