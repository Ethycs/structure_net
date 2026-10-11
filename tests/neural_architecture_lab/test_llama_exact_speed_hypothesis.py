from dataclasses import replace
import pytest
import torch

from neural_architecture_lab.llama_exact_speed_hypothesis import CANDIDATES, GRID, ExactSpeedEvidence, SpeedCell
from experiments.neural_architecture_lab.finalize_llama_exact_speed import read_controls
from experiments.structure_net.prepare_llama_cotenancy import sha256


def evidence():
    cells = tuple(SpeedCell(n, length, steps, True, 1.0) for n in CANDIDATES for length, steps in GRID)
    return ExactSpeedEvidence("c", "a"*64, "r", "b"*64, cells)


def test_shape_failure_vetoes_otherwise_fast_candidate():
    original = evidence()
    cells = list(original.cells)
    cells[0] = replace(cells[0], speedup=2.)
    cells[1] = replace(cells[1], exact=False, speedup=None)
    record = replace(original, cells=tuple(cells)).record()
    assert "mm" not in record["eligible_candidates"] and not record["promoted"]
    assert record["confirmation_status"] == "no_qualifying_exact_speedup"
    assert not record["confirmed"] and not record["application_accuracy_tested"]


def test_inclusive_threshold_and_complete_grid():
    original = evidence()
    cells = (replace(original.cells[0], speedup=1.1), *original.cells[1:])
    record = replace(original, cells=cells).record()
    assert record["promoted"] == [{"candidate": "mm", "length": 128, "decode_steps": 0}]
    assert record["confirmation_status"] == "exact_speedup_over_graph_supported"
    with pytest.raises(ValueError, match="complete fixed grid"):
        replace(original, cells=original.cells[:-1])
    with pytest.raises(ValueError, match="complete fixed grid"):
        replace(original, cells=(original.cells[0], *original.cells[:-1]))
    with pytest.raises(ValueError, match="only exact"):
        SpeedCell("mm", 128, 0, False, 1.2)
    with pytest.raises(ValueError, match="finite positive"):
        SpeedCell("mm", 128, 0, True, float("nan"))


def test_artifact_audit_preserves_signed_zero_and_rejects_false_claim(tmp_path):
    path = tmp_path / "controls.pt"
    values = []
    for ids in ([[1, 2]], [[3, 4]]):
        expected = (torch.tensor([0., -0.]), None)
        actual = (torch.tensor([0., 0.]), None)
        values.append({"ids": torch.tensor(ids), "expected": expected, "baseline_graph": expected, "baseline_eager": expected,
                       "candidate_eager": actual, "actual": actual, "bitwise_equal": False})
    torch.save(values, path)
    row = {"control_artifact": str(path), "control_sha256": sha256(path), "all_bitwise": False, "owned_outputs": True}
    assert not read_controls(row)
    with pytest.raises(ValueError, match="exactness gate mismatch"):
        read_controls({**row, "all_bitwise": True})
    values[1]["ids"] = values[0]["ids"]
    torch.save(values, path)
    with pytest.raises(ValueError, match="two distinct"):
        read_controls({**row, "control_sha256": sha256(path)})
