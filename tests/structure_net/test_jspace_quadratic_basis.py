"""CPU proofs for explicit Krylov/Ritz construction and honest null completion."""
import json

import pytest
import torch

from experiments.structure_net import jspace_quadratic_basis as fit


def test_full_krylov_span_recovers_known_psd_top_energy_and_eigenvectors():
    gram = torch.diag(torch.tensor([12., 9., 7., 5., 3., 2., 1., .5], dtype=torch.float64))
    basis, info = fit.block_krylov_basis(gram, 4, block_width=4, steps=1, seed=17)
    torch.testing.assert_close(basis.T @ basis, torch.eye(4, dtype=gram.dtype), rtol=0, atol=2e-14)
    assert info["krylov_dimension"] == 8 and info["expansions"][0]["new_directions"] == 4
    assert info["captured_normalized_gram_energy"] == pytest.approx(33 / 39.5, abs=1e-14)
    torch.testing.assert_close(basis @ basis.T, torch.diag(torch.tensor([1.] * 4 + [0.] * 4, dtype=gram.dtype)), atol=2e-13, rtol=0)
    assert info["ritz_residual_frobenius"] < 1e-13


def test_partial_krylov_ritz_matches_exact_constructed_span_and_improves_seed_energy():
    gram = torch.diag(torch.linspace(1., 12., 12, dtype=torch.float64)); operator = gram / gram.trace()
    seed, rank = 28, 3
    basis, info = fit.block_krylov_basis(gram, rank, block_width=rank, steps=1, seed=seed)
    _, initial = fit.block_krylov_basis(gram, rank, block_width=rank, steps=0, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    b = torch.randn(12, rank, dtype=gram.dtype, generator=generator)
    reference = torch.linalg.qr(torch.cat((b, operator @ b), dim=1), mode="reduced").Q
    torch.testing.assert_close(reference @ (reference.T @ basis), basis, atol=2e-13, rtol=0)
    values = torch.linalg.eigvalsh(reference.T @ operator @ reference)
    assert info["captured_normalized_gram_energy"] == pytest.approx(float(values[-rank:].sum()), abs=1e-14)
    assert info["captured_normalized_gram_energy"] >= initial["captured_normalized_gram_energy"]
    assert info["krylov_dimension"] == 6


def test_rank_deficiency_and_zero_operator_do_not_invent_functional_directions():
    gram = torch.diag(torch.tensor([5., 2.] + [0.] * 8, dtype=torch.float64))
    basis, info = fit.block_krylov_basis(gram, 5, block_width=5, seed=16)
    assert basis.shape == (10, 5) and info["krylov_dimension"] == 7
    assert info["retained_functional_directions"] == 2 and info["retained_numerical_null_completion"] == 3
    assert info["captured_normalized_gram_energy"] == pytest.approx(1., abs=1e-14)
    empty, zero = fit.block_krylov_basis(torch.zeros_like(gram), 5, block_width=5, seed=16)
    assert zero["retained_functional_directions"] == 0 and zero["retained_numerical_null_completion"] == 5
    assert zero["captured_normalized_gram_energy"] is None and zero["expansions"][0]["new_directions"] == 0
    torch.testing.assert_close(empty.T @ empty, torch.eye(5, dtype=gram.dtype), atol=1e-14, rtol=0)


def test_seed_determinism_and_scale_invariance():
    gram = torch.diag(torch.linspace(1., 16., 16))
    first, metadata = fit.block_krylov_basis(gram, 4, block_width=4, seed=43)
    again, same = fit.block_krylov_basis(gram, 4, block_width=4, seed=43)
    scaled, scale_info = fit.block_krylov_basis(gram * 8, 4, block_width=4, seed=43)
    assert torch.equal(first, again) and metadata == same
    torch.testing.assert_close(first, scaled, atol=0, rtol=0)
    assert scale_info["gram_trace"] == metadata["gram_trace"] * 8


def statistics():
    count = 32; moment = torch.diag(torch.tensor([5., 4., 3., 2.], dtype=torch.float64))
    return {"grams": torch.stack((torch.diag(torch.arange(1., 9., dtype=torch.float64)),) * 2),
        "mean": torch.arange(16, dtype=torch.float64).reshape(2, 8), "count": count,
        "pair_energy": torch.tensor([[[1., 4.]], [[3., 2.]]], dtype=torch.float64),
        "value_cov": moment[None, None].repeat(2, 1, 1, 1) * count,
        "value_sum": torch.tensor([[[2., 0., 0., 0.]]] * 2, dtype=torch.float64) * count}


def test_layer_fit_preserves_global_mean_pairs_and_uncentered_value_moment(tmp_path):
    stats = statistics()
    result = fit.fit_layer(stats, 0, rank=4, block_width=4, head_dim=2, seed=11)
    assert result["basis"].shape == (8, 4) and result["means"].shape == (1, 8)
    assert torch.equal(result["means"][0], stats["mean"][0])
    assert result["rope_pairs"].tolist() == [[1]]
    v = result["value_basis"][0]
    torch.testing.assert_close(v @ v.T, torch.diag(torch.tensor([1., 1., 0., 0.], dtype=v.dtype)))
    assert "NOT full-vocabulary" in result["metadata"]["jacobian_provenance"]
    assert result["metadata"]["calibration_examples"] == 2
    source = tmp_path / "statistics.pt"; torch.save(stats, source)
    events = []
    paths = fit.fit_all_layers(source, tmp_path / "output", rank=4, block_width=4, head_dim=2, seed=11, progress=events.append)
    assert set(paths) == {"0", "1"} and [event["layer"] for event in events] == [0, 1]
    saved = torch.load(paths["0"]["fit"], weights_only=True)
    assert torch.equal(saved["basis"], result["basis"])
    assert json.loads((tmp_path / "output/block_00/fit.json").read_text())["value_provenance"].startswith("uncentered")
    with pytest.raises(FileExistsError): fit.fit_all_layers(source, tmp_path / "output", rank=4, block_width=4, head_dim=2)


@pytest.mark.parametrize("gram", [torch.tensor([[1., .2], [0., 1.]]), torch.diag(torch.tensor([2., -1.])),
                                      torch.tensor([[float('nan'), 0.], [0., 1.]]), torch.eye(2).bfloat16()])
def test_invalid_operators_are_rejected(gram):
    with pytest.raises(ValueError): fit.block_krylov_basis(gram, 1, block_width=1)


def test_invalid_budget_heads_and_missing_gpu_guard_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="rank <= block_width"):
        fit.block_krylov_basis(torch.eye(4), 3, block_width=2)
    with pytest.raises(ValueError, match="head statistics"):
        fit.fit_layer(statistics(), 0, rank=4, block_width=4, head_dim=3)
    with pytest.raises(ValueError, match="reservation guard"):
        fit.fit_all_layers(tmp_path / "missing", tmp_path / "out", device="cuda:0")
