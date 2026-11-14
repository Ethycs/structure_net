from copy import deepcopy

import pytest
import torch

from experiments.structure_net.jspace_mean_preserving_basis import replace_mean_direction, _fit_from_statistics


def fixture(dtype=torch.float64):
    torch.manual_seed(609)
    q = torch.linalg.qr(torch.randn(12, 12, dtype=torch.float64)).Q.to(dtype)
    spectrum = torch.tensor([8., 7., 6., 5., 4., 3., 2., 1., .2, .1, .01, 0.], dtype=dtype)
    return q[:, :8].contiguous(), q@torch.diag(spectrum)@q.T, 3*q[:, 0]+5*q[:, -1]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mean_enters_span_only_weakest_column_changes_and_energy_is_exact(dtype):
    u, gram, mean = fixture(dtype)
    original = u.clone()
    changed, info = replace_mean_direction(u, mean, gram, resolved_functional_rank=7)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-13
    assert torch.equal(u, original)
    assert torch.equal(changed[:, :-1], original[:, :-1])
    assert info["changed"] and not info["mean_already_in_original_span"]
    torch.testing.assert_close(changed.T@changed, torch.eye(8, dtype=dtype), rtol=0, atol=tolerance)
    torch.testing.assert_close(changed@(changed.T@mean), mean, rtol=0, atol=tolerance*5)
    expected = float((changed*(gram@changed)).sum()/gram.trace())
    assert info["new_captured_gram_energy"] == pytest.approx(expected, abs=tolerance)
    assert info["captured_gram_energy_delta"] == pytest.approx(-1/float(gram.trace()), abs=tolerance)
    assert info["new_last_gram_energy"] == pytest.approx(0., abs=tolerance)
    # Every previously resolved direction is literally retained, not refitted.
    torch.testing.assert_close(changed[:, :7], original[:, :7], atol=0, rtol=0)


@pytest.mark.parametrize("which", ["fixed", "last", "zero"])
def test_mean_already_in_original_span_keeps_entire_basis(which):
    u, gram, _ = fixture()
    mean = {"fixed":u[:, 0]*3, "last":u[:, -1]*3, "zero":u[:, 0]*0}[which]
    result, info = replace_mean_direction(u, mean[None], gram, resolved_functional_rank=6)
    assert torch.equal(result, u) and not info["changed"]
    assert info["captured_gram_energy_delta"] == 0.


def test_determinism_scale_and_zero_gram_do_not_invent_jacobian_information():
    u, gram, mean = fixture()
    first, info = replace_mean_direction(u, mean, gram, resolved_functional_rank=7)
    again, same = replace_mean_direction(u, mean, gram, resolved_functional_rank=7)
    assert torch.equal(first, again) and info == same
    scaled, _ = replace_mean_direction(u, mean*8, gram*3, resolved_functional_rank=7)
    torch.testing.assert_close(first, scaled, atol=2e-15, rtol=2e-15)
    empty, meta = replace_mean_direction(u, mean, torch.zeros_like(gram), resolved_functional_rank=0)
    torch.testing.assert_close(empty, first, atol=0, rtol=0)
    assert meta["new_captured_gram_energy"] is None


def test_candidate_consumes_only_pinned_statistics_and_preserves_heads_and_mean():
    u, gram, mean = fixture()
    original = dict(basis=u, means=mean[None], rope_pairs=torch.tensor([[0, 2]]),
                    value_basis=torch.eye(4)[None], metadata={"retained_functional_directions":7})
    # No examples, labels, development or evaluation arrays are supplied. This
    # is sufficient because the correction consumes the frozen training mean.
    stats = {"mean":mean[None], "grams":gram[None]}
    saved = deepcopy(original)
    candidate = _fit_from_statistics(original, stats, 0)
    for key in ("means", "rope_pairs", "value_basis"):
        assert torch.equal(candidate[key], saved[key])
    assert torch.equal(original["basis"], saved["basis"])
    assert not torch.equal(candidate["basis"], saved["basis"])
    poisoned_mean = {**stats, "mean":stats["mean"]+1}
    with pytest.raises(ValueError, match="native training statistics"):
        _fit_from_statistics(original, poisoned_mean, 0)


@pytest.mark.parametrize("case", ["all_strong", "nonorthogonal", "nonfinite", "nonsymmetric", "dtype"])
def test_invalid_inputs_are_rejected(case):
    u, gram, mean = fixture()
    resolved = 7
    if case == "all_strong": resolved = 8
    elif case == "nonorthogonal": u[:, 0] *= 2
    elif case == "nonfinite": mean[0] = float("nan")
    elif case == "nonsymmetric": gram[0, 1] += 1
    elif case == "dtype": gram = gram.float()
    with pytest.raises(ValueError):
        replace_mean_direction(u, mean, gram, resolved_functional_rank=resolved)
