import json
import sys

import pytest

from experiments.structure_net import ncm_source_hierarchy_supervised as supervised


def test_child_failure_is_observed_even_if_it_wrote_a_passing_result(tmp_path):
    destination = tmp_path / "child"
    code = ("import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); "
            "(p/'result.json').write_text(json.dumps({'status':'passed'})); sys.exit(7)")
    with pytest.raises(RuntimeError, match="exited 7"):
        supervised.observe_child([sys.executable, "-c", code, str(destination)], destination, tmp_path)
    receipt = json.loads((destination / "worker_terminal_receipt.json").read_text())
    assert receipt["exit_code"] == 7 and receipt["observer"] == "subprocess.wait"
    assert json.loads((tmp_path / "active_child.json").read_text())["exit_code"] == 7


def test_existing_worker_is_not_duplicated_or_overwritten(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "original"
    marker.write_bytes(b"original evidence")
    with pytest.raises(ValueError, match="must not be restarted"):
        supervised.observe_child([sys.executable, "-c", "raise AssertionError('must not launch')"], destination, tmp_path)
    assert marker.read_bytes() == b"original evidence"
    assert not (tmp_path / "active_child.json").exists()
