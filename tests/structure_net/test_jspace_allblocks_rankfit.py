"""The candidate changes numerical admission, not data or graph fitting."""
import json

import numpy as np
import pytest
import torch

from experiments.structure_net import jspace_allblocks_rankfit as revised
from experiments.structure_net import jspace_benchmark_groebner as frozen
from experiments.structure_net.jspace_bench_data import sha, write


def assert_identical(a, b):
    if isinstance(a, torch.Tensor): assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_identical(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_identical(x, y)
    else: assert a == b


def test_old_admitted_fit_is_identical_in_anchors_coefficients_removals_and_metrics():
    torch.manual_seed(902)
    train, dev, jacobian, weight = torch.randn(64, 12), torch.randn(24, 12), torch.randn(91, 12), torch.randn(4, 12)
    data = (train, ["a"] * 32 + ["b"] * 32, jacobian, weight, dev, ["a"] * 12 + ["b"] * 12)
    options = dict(anchor_widths=(2, 3), remove_count=3, seed=117, svd_rank=6, oversample=3,
                   power_iterations=1, chunk_rows=13, compressed_dev_activations=dev * .9 + .2)
    before = frozen.fit_benchmark_groebner(*data, **options)
    after = revised.fit_benchmark_groebner(*data, **options)
    diagnosis = after["anchor_selection"].pop("rank_admission")
    assert diagnosis["admitted"] and len(diagnosis["all_singular_values"]) == 9
    assert diagnosis["rule_id"] == "actual_sketch_svd_shape_fp32"
    assert_identical(before, after)


def test_duplicate_rows_do_not_change_new_rank_admission_or_anchor_count():
    original = torch.diag(torch.tensor([1., 5e-5])); repeated = original.repeat(1024, 1)
    settings = dict(max_anchors=2, svd_rank=2, oversample=0, power_iterations=1, chunk_rows=256)
    first, before = revised.select_nested_anchors(original, torch.ones(2), **settings)
    second, after = revised.select_nested_anchors(repeated, torch.ones(2), **settings)
    a, b = before["rank_admission"], after["rank_admission"]
    assert first == second == [0, 1]
    assert a["original_shape_gate"]["rank"] == 2 and b["original_shape_gate"]["rank"] == 1
    assert a["small_svd_gate"]["rank"] == b["small_svd_gate"]["rank"] == 2
    assert a["small_svd_gate"]["relative_cutoff"] == b["small_svd_gate"]["relative_cutoff"]
    assert a["admitted"] is b["admitted"] is True


def test_true_deficiency_is_rejected_after_diagnostics_callback():
    captured = []
    with pytest.raises(revised.RankAdmissionError, match="actual small-SVD tolerance") as error:
        revised.select_nested_anchors(torch.ones(40, 8), torch.ones(8), max_anchors=3,
            svd_rank=4, oversample=2, on_spectrum=lambda report, state: captured.append((report, state)))
    report, state = captured[0]
    assert report == error.value.diagnostics and report["admitted"] is False
    assert report["small_svd_gate"]["rank"] == 1
    assert len(report["all_singular_values"]) == 6 and state["small"].shape == (6, 8)


def interface_fixture(tmp_path, deficient=False):
    torch.manual_seed(443); directory = tmp_path / "block_30"; directory.mkdir()
    paths = {}
    values = {"native_calibration": torch.randn(16, 8), "native_development": torch.randn(8, 8),
              "compiled_development": torch.randn(8, 8),
              "jacobian": torch.ones(40, 8) if deficient else torch.randn(40, 8)}
    for name, value in values.items():
        path = directory / (name + ".npy"); np.save(path, value.bfloat16().view(torch.int16).numpy()); paths[name] = str(path)
    path = directory / "output_weight.pt"; torch.save(torch.randn(4, 8).bfloat16().float(), path); paths["output_weight"] = str(path)
    write(directory / "interface.json", {**paths, "layer": 30, "teacher_fingerprint": "unchanged",
        "calibration_families": ["a"] * 8 + ["b"] * 8, "development_families": ["a"] * 4 + ["b"] * 4,
        "input_sha256": {name: sha(name) for name in paths.values()}})
    return directory


@pytest.mark.parametrize("deficient", [False, True])
def test_artifact_interface_preserves_full_spectrum_even_on_rejection(tmp_path, deficient):
    directory = interface_fixture(tmp_path, deficient); events = []
    options = dict(anchors=3, remove_count=2, svd_rank=4, oversample=2, chunk_rows=8, progress=events.append)
    if deficient:
        with pytest.raises(revised.RankAdmissionError): revised.fit_relation(directory, **options)
        assert not (directory / "relation.pt").exists()
    else:
        result = revised.fit_relation(directory, **options)
        assert result["fixed_anchor_width"] == 3 and result["removed_channels"] == 2
        assert result["streamed_jacobian"]["rows_read"] == 160
        assert result["rank_spectrum_sha256"] == sha(directory / "rank_spectrum.pt")
    report = json.loads((directory / "rank_diagnostic.json").read_text())
    state = torch.load(directory / "rank_spectrum.pt", weights_only=True)
    assert report["diagnostics"]["admitted"] is not deficient
    assert len(report["diagnostics"]["all_singular_values"]) == 6
    assert state["vh"].shape == state["small"].shape == (6, 8)
    assert report["spectral_state_sha256"] == sha(directory / "rank_spectrum.pt")
    assert [row["pass_number"] for row in events if row["phase"] == "jacobian_pass_complete"] == [1, 2, 3, 4]
    with pytest.raises(FileExistsError): revised.fit_relation(directory, **options)
