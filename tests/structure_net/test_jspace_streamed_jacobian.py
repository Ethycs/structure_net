import math

import numpy as np
import pytest
import torch

from experiments.structure_net.jspace_benchmark_groebner import (
    fit_benchmark_groebner, select_nested_anchors,
)
from experiments.structure_net.jspace_streamed_jacobian import StreamedJacobianRows


def save_matrix(tmp_path, value, name="jacobian.npy"):
    path = tmp_path / name
    np.save(path, value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value)
    return path


def assert_metadata_close(actual, expected, tolerance=5e-6):
    # Owned chunks can have a different CPU BLAS alignment from resident views.
    # Compare numerical diagnostics at arithmetic precision, while preserving
    # exact discrete choices and the same full-row mathematical estimator.
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=tolerance, abs=tolerance * .01)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_metadata_close(actual[key], expected[key], tolerance)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_metadata_close(a, b, tolerance)
    else:
        assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_streamed_anchor_selection_matches_resident_and_reads_all_rows_four_times(tmp_path, dtype):
    torch.manual_seed(132)
    jacobian = torch.randn(129, 48, dtype=dtype)
    scale = torch.rand(48, dtype=dtype) + .3
    path = save_matrix(tmp_path, jacobian)
    options = dict(max_anchors=8, seed=981, svd_rank=16, oversample=4,
                   power_iterations=1, chunk_rows=32)
    expected, expected_info = select_nested_anchors(jacobian, scale, **options)
    with StreamedJacobianRows(path, max_rows=32) as source:
        actual, actual_info = select_nested_anchors(source, scale, **options)
        assert actual == expected
        assert_metadata_close(actual_info, expected_info, 1e-12 if dtype == torch.float64 else 5e-6)
        assert source.rows_read == 4 * len(jacobian)
        assert source.read_calls == 4 * math.ceil(len(jacobian) / 32)
        assert source.maximum_read_rows == 32
        assert source.bytes_read == 4 * jacobian.numel() * jacobian.element_size()


def test_full_unchanged_fitter_accepts_streaming_source_and_reproduces_fit(tmp_path):
    torch.manual_seed(417)
    train, dev = torch.randn(64, 48), torch.randn(32, 48)
    jacobian, output = torch.randn(129, 48), torch.randn(8, 48)
    families, dev_families = ["a"] * 32 + ["b"] * 32, ["a"] * 16 + ["b"] * 16
    options = dict(anchor_widths=(4, 8), remove_count=8, seed=192,
                   svd_rank=16, oversample=4, power_iterations=1, chunk_rows=32)
    expected = fit_benchmark_groebner(train, families, jacobian, output, dev, dev_families, **options)
    with StreamedJacobianRows(save_matrix(tmp_path, jacobian), max_rows=32) as source:
        actual = fit_benchmark_groebner(train, families, source, output, dev, dev_families, **options)
        assert actual.keys() == expected.keys()
        assert actual["selected_anchor_width"] == expected["selected_anchor_width"]
        for key in expected.keys() - {"fits"}:
            assert_metadata_close(actual[key], expected[key])
        for width, expected_fit in expected["fits"].items():
            fit = actual["fits"][width]
            assert_metadata_close(fit, expected_fit)


def test_owned_slice_does_not_mutate_source_and_survives_close(tmp_path):
    data = torch.arange(80, dtype=torch.float32).reshape(10, 8)
    path = save_matrix(tmp_path, data)
    source = StreamedJacobianRows(path, max_rows=3)
    assert source.shape == (10, 8) and source.ndim == 2 and source.dtype == torch.float32
    assert source.device == torch.device("cpu") and source.readonly and source.is_floating_point()
    chunk = source[2:5]
    source.close()
    chunk.zero_()
    assert np.array_equal(np.load(path), data.numpy())
    assert torch.equal(chunk, torch.zeros(3, 8))
    assert source.metadata()["maximum_read_rows"] == 3
    with pytest.raises(ValueError, match="closed"):
        source[:1]


def test_cuda_metadata_does_not_allocate_or_initialize_cuda(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Constructor must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    with StreamedJacobianRows(save_matrix(tmp_path, torch.ones(4, 8)), device="cuda:0") as source:
        assert source.device == torch.device("cuda:0")
        assert source.read_calls == 0


def test_invalid_sources_and_unbounded_slices_are_rejected(tmp_path):
    for array, message in [(np.ones(4, dtype=np.float32), "nonempty matrix"),
                           (np.empty((0, 4), dtype=np.float32), "nonempty matrix"),
                           (np.ones((4, 4), dtype=np.int16), "float32 or float64"),
                           (np.ones((4, 4), dtype=">f4"), "native-endian")]:
        with pytest.raises(ValueError, match=message):
            StreamedJacobianRows(save_matrix(tmp_path, array))
    source = StreamedJacobianRows(save_matrix(tmp_path, torch.ones(10, 8)), max_rows=3)
    with pytest.raises(ValueError, match="exceeds max_rows"):
        source[:]
    with pytest.raises(TypeError, match="contiguous row slices"):
        source[0]
    with pytest.raises(TypeError, match="contiguous row slices"):
        source[::2]
    source.close()


def test_nonfinite_values_are_rejected_by_existing_chunked_fitter(tmp_path):
    jacobian = torch.randn(32, 16)
    jacobian[17, 3] = float("nan")
    with StreamedJacobianRows(save_matrix(tmp_path, jacobian), max_rows=8) as source:
        with pytest.raises(ValueError, match="Jacobian contains nonfinite"):
            select_nested_anchors(source, torch.ones(16), max_anchors=4, svd_rank=8, chunk_rows=8)
