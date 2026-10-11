import math
import pytest
import torch
from structure_net.components.layers import model_major_tensor as t


def make_tt(seed):
    generator = torch.Generator().manual_seed(seed)
    ins, outs, ranks = (2, 3, 2, 2), (3, 2, 2, 3), (1, 2, 3, 2, 1)
    cores = [torch.randn(ranks[i], outs[i], ins[i], ranks[i+1], generator=generator, dtype=torch.float64) for i in range(4)]
    return {"kind": "tt4", "shape": [math.prod(outs), math.prod(ins)], "input_modes": ins, "output_modes": outs, "cores": cores}


def test_model_axis_is_independent_and_tt_contraction_matches_dense():
    torch.set_num_threads(1)
    specs = [make_tt(i) for i in range(3)]
    packed = t.pack(specs)
    x = torch.randn(3, 7, 24, dtype=torch.float64)
    actual = t.apply(x, packed)
    expected = torch.stack([x[i] @ t.reconstruct(spec).T for i, spec in enumerate(specs)])
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-11)
    separate = torch.cat([t.apply(x[i:i+1], t.pack([spec])) for i, spec in enumerate(specs)])
    torch.testing.assert_close(actual, separate, atol=1e-10, rtol=1e-11)
    packed["cores"][0][0].mul_(2)
    changed = t.apply(x, packed)
    assert not torch.equal(changed[0], actual[0])
    torch.testing.assert_close(changed[1:], actual[1:], atol=0, rtol=0)
    assert packed["cores"][0][0].data_ptr() != packed["cores"][0][1].data_ptr()


def test_decomposition_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(42)
    weight = torch.randn(64, 4) @ torch.randn(4, 32)
    spec = t.decompose(weight, t.Tensorization("test", "svd2", rank_fraction=.125))
    torch.testing.assert_close(t.reconstruct(spec), weight, atol=2e-5, rtol=2e-5)
    path = tmp_path / "factors.pt"
    torch.save(spec, path)
    restored = torch.load(path, weights_only=True)
    x = torch.randn(2, 3, 32)
    torch.testing.assert_close(t.apply(x, t.pack([restored, restored])), x @ weight.T, atol=8e-5, rtol=3e-5)


def test_deeper_factorization_matches_its_reconstructed_operator():
    torch.manual_seed(43)
    spec = t.decompose(torch.randn(64, 64), t.Tensorization("test", "tt4", rank=2))
    assert spec["kind"] == "tt4" and len(spec["cores"]) == 4
    x = torch.randn(2, 5, 64)
    torch.testing.assert_close(t.apply(x, t.pack([spec, spec])), x @ t.reconstruct(spec).T, atol=2e-5, rtol=2e-5)
    assert sum(c.numel() for c in spec["cores"]) < 64 * 64


def test_shape_validation_and_exact_mode_products():
    for size in (1024, 3072, 8192, 128256):
        assert math.prod(t.modes(size)) == size
    with pytest.raises(ValueError):
        t.Tensorization("bad", "svd2", rank_fraction=0)
    with pytest.raises(ValueError):
        t.apply(torch.randn(2, 3, 24), t.pack([make_tt(0)]))


def test_tensorized_embedding_selects_exact_rows_without_cross_model_sharing():
    specs = [make_tt(3), make_tt(4)]
    ids = torch.tensor([[1, 5, 12, 20, 1], [2, 9, 14, 3, 9]])
    actual = t.embedding(ids, t.pack(specs), chunk_tokens=2)
    expected = torch.stack([t.reconstruct(spec)[ids[i]] for i, spec in enumerate(specs)])
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-11)
