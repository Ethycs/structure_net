import math

import torch

from experiments.structure_net.tinyllm_reynolds_character_coupling import (
    _regime_name,
    character_metrics,
    fisher_rao_squared,
    residual_approximation,
    taylor_defect,
    task_effect_approximation,
)


def test_character_transform_reconstructs_real_deck_sheets() -> None:
    torch.manual_seed(3)
    for k in (2, 3):
        metrics = character_metrics(torch.randn(7, k, 2, 5))
        assert metrics["reconstruction_error"] < 1e-12
        assert math.isclose(sum(metrics["energy_fractions"].values()), 1.0)
        assert all((left + right) % k == 0 for left, right in metrics["neutral_quadratic_pairs"])


def test_neutral_quadratic_stencil_exactly_recovers_quadratic_jensen_defect() -> None:
    sheets = torch.tensor(
        [
            [[-2.0, 1.0], [0.0, 2.0], [2.0, 0.0]],
            [[1.0, -1.0], [2.0, 1.0], [3.0, 3.0]],
        ]
    )
    apply = lambda value: value + 0.4 * value.square()
    propagated, _, quadratic, cubic = taylor_defect(sheets, apply, eta=0.25)
    exact = apply(sheets).mean(1) - propagated
    assert torch.allclose(quadratic, exact, atol=1e-5)
    assert torch.allclose(cubic, torch.zeros_like(cubic), atol=1e-5)


def test_residual_and_fisher_effect_report_perfect_approximation() -> None:
    exact = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    residual = residual_approximation(exact, exact)
    assert math.isclose(residual["explained_fraction"], 1.0)
    assert math.isclose(residual["cosine"], 1.0)

    propagated = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
    actual = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
    effect = task_effect_approximation(propagated, actual, actual, floor=1e-8)
    assert not effect["degenerate"]
    assert math.isclose(effect["explained_fraction"], 1.0)
    assert torch.all(fisher_rao_squared(actual, actual) < 1e-12)


def test_four_causal_regimes_have_distinct_names() -> None:
    names = {_regime_name(left, right) for left in (False, True) for right in (False, True)}
    assert names == {
        "cover_required_after_sublayer",
        "invariant_synthesis",
        "quotient_already_closed",
        "quotient_corruption",
    }
