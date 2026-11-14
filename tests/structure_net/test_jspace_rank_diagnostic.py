"""CPU proofs for unchanged sketches and row-count-dependent rank diagnosis."""
import json

import numpy as np
import pytest
import torch

from experiments.structure_net import jspace_rank_diagnostic as diagnostic
from experiments.structure_net.jspace_benchmark_groebner import select_nested_anchors


def test_diagnostic_exactly_reproduces_frozen_sketch_pivots_and_spectrum():
    torch.manual_seed(807); raw = torch.randn(71, 14); scale = torch.rand(14) + .2
    options = dict(seed=345, svd_rank=6, oversample=3, power_iterations=1, chunk_rows=11)
    anchors, original = select_nested_anchors(raw, scale, max_anchors=4, **options)
    report, state = diagnostic.diagnose(raw, scale, anchors=4, progress=lambda value: None, **options)
    assert report["passes"] == 4 and report["small_shape"] == [9, 14]
    assert report["diagnostic_anchor_indices"] == anchors
    assert report["diagnostic_pivot_norms"] == original["pivot_norms"]
    assert report["singular_values"][:6] == original["singular_values"]
    assert report["captured_energy_fraction"] == original["captured_energy_fraction"]
    torch.testing.assert_close(state["standardized_row_energy"], (raw * scale).double().square().sum(1))


def test_duplicated_rows_preserve_ratios_but_inflate_frozen_rank_cutoff():
    # Full row duplication scales singular values uniformly, without changing rank.
    original = torch.diag(torch.tensor([1., 5e-5])); repeated = original.repeat(1024, 1)
    before = torch.linalg.svdvals(original); after = torch.linalg.svdvals(repeated)
    torch.testing.assert_close(before / before[0], after / after[0])
    a = diagnostic.rank_cutoffs(before, original.shape, (2, 2))
    b = diagnostic.rank_cutoffs(after, repeated.shape, (2, 2))
    assert a["frozen_original_shape"]["rank"] == 2
    assert b["frozen_original_shape"]["rank"] == 1
    assert b["actual_svd_shape"]["rank"] == 2
    assert b["frozen_original_shape"]["relative_cutoff"] == 1024 * a["frozen_original_shape"]["relative_cutoff"]


def test_case_family_energy_and_zero_rows_cover_unmodified_bank():
    raw = torch.tensor([[1., 0.], [0., 0.], [0., 2.], [0., 0.], [3., 0.], [0., 4.]])
    cases = [{"identity": "x", "family": "a", "start": 0, "stop": 2},
             {"identity": "y", "family": "a", "start": 2, "stop": 4},
             {"identity": "z", "family": "b", "start": 4, "stop": 6}]
    report, state = diagnostic.diagnose(raw, torch.tensor([2., 1.]), anchors=2, svd_rank=2, oversample=0,
                                       chunk_rows=2, case_ranges=cases, progress=lambda value: None)
    info = report["row_concentration"]
    assert info["zero_rows"] == 2 and info["families"]["a"]["zero_rows"] == 2
    assert info["families"]["a"]["energy"] == 8 and info["families"]["b"]["energy"] == 52
    assert info["top_case_energy_fraction"] == pytest.approx(52 / 60)
    assert report["total_standardized_jacobian_energy"] == 60
    assert state["raw_row_energy"].tolist() == [1, 0, 4, 0, 9, 16]


def test_diagnostic_does_not_enforce_or_change_failing_original_gate():
    raw = torch.diag(torch.tensor([1., 5e-5])).repeat(1024, 1)
    with pytest.raises(ValueError, match="Insufficient independent"):
        select_nested_anchors(raw, torch.ones(2), max_anchors=2, svd_rank=2, oversample=0, chunk_rows=256)
    report, _ = diagnostic.diagnose(raw, torch.ones(2), anchors=2, svd_rank=2, oversample=0, progress=lambda value: None)
    assert report["frozen_gate_passes"] is False
    assert len(report["diagnostic_anchor_indices"]) == 2
    assert all(x > 0 for x in report["diagnostic_pivot_norms"])


def test_case_ranges_and_cuda_guard_reject_before_unsafe_work(tmp_path):
    with pytest.raises(ValueError, match="reservation guard"):
        diagnostic.run(tmp_path, device="cuda:0")
    with pytest.raises(ValueError, match="partition"):
        diagnostic._concentration(torch.ones(4), torch.ones(4), [{"identity": "x", "family": "a", "start": 1, "stop": 4}])


def test_append_only_prepare_run_freezes_inputs_and_saves_reusable_spectrum(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch.manual_seed(912)
    capture = tmp_path / "capture"; source = capture / "block_30"; source.mkdir(parents=True)
    block = tmp_path / "block_30"; block.mkdir()
    training, jacobian = source / "train.npy", source / "jacobian.npy"
    for path, shape in ((training, (32, 32)), (jacobian, (64, 32))):
        np.save(path, torch.randn(shape).bfloat16().view(torch.int16).numpy())
    diagnostic.write(capture / "metadata.json", {"status": "completed", "calibration_indices": [0, 1],
                     "probe_groups": 2, "probes_per_group": 2})
    diagnostic.write(capture / "items.json", {"items": [
        {"identity": "a", "family": "one", "offset": 0, "stop": 8, "jacobian_row_offset": 0},
        {"identity": "b", "family": "two", "offset": 8, "stop": 16, "jacobian_row_offset": 32}]})
    interface = {"layer": 30, "teacher_fingerprint": "frozen", "capture_block": {"boundary": str(source / "boundary.npy")},
                 "jacobian": str(jacobian), "native_calibration": str(training),
                 "input_sha256": {str(path): diagnostic.sha(path) for path in (training, jacobian)}}
    diagnostic.write(block / "interface.json", interface)
    root = tmp_path / "diagnostic"; diagnostic.prepare(root, block, seed=117)
    report = diagnostic.run(root, device="cpu")
    state = torch.load(root / "spectral_state.pt", weights_only=True)
    assert report["status"] == "completed" and report["streamed_jacobian"]["rows_read"] == 4 * 64
    assert state["small"].shape == state["vh"].shape == (32, 32)
    assert state["mean"].shape == state["scale"].shape == (32,)
    plan = json.loads((root / "plan.json").read_text())
    assert report["fingerprint"] == plan["fingerprint"]
    assert diagnostic.sha(root / "spectral_state.pt") == report["spectral_state_sha256"]
    with pytest.raises(FileExistsError): diagnostic.run(root, device="cpu")
    other = tmp_path / "changed"; diagnostic.prepare(other, block, seed=117)
    with jacobian.open("ab") as stream: stream.write(b"changed")
    with pytest.raises(ValueError, match="Changed diagnostic input"):
        diagnostic.run(other, device="cpu")
