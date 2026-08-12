from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.structure_net import (
    tinyllm_calibrated_frontend_causal_closure as closure,
)


def _underpowered_config(**changes: object) -> closure.CausalClosureConfig:
    values = {
        "conditions": ("raw_calibrated",),
        "seeds": (7,),
        "required_seed_passes": 1,
        "device": "cpu",
        "allow_underpowered": True,
    }
    values.update(changes)
    return closure.CausalClosureConfig(**values)


def _counts(**overrides: dict[str, int]) -> dict[str, dict[str, int]]:
    values = {
        condition: {cut: 5 for cut in closure.CUTS}
        for condition in closure.PRIMARY_CONDITIONS
    }
    for condition, cuts in overrides.items():
        values[condition].update(cuts)
    return values


def test_primary_configuration_and_preregistration_are_locked() -> None:
    config = closure.CausalClosureConfig()
    assert config.conditions == closure.CONDITIONS
    assert config.seeds == closure.SEEDS
    assert config.required_seed_passes == 4
    assert config.batch_size == 256
    assert closure._sha256(closure.PREREGISTRATION_PATH) == (
        closure.PREREGISTRATION_SHA256
    )
    with pytest.raises(ValueError, match="five checkpoints are fixed"):
        closure.CausalClosureConfig(seeds=(7, 17, 29, 41))
    with pytest.raises(ValueError, match="four of five"):
        closure.CausalClosureConfig(required_seed_passes=3)


def test_source_result_manifest_and_held_out_cohorts_are_locked() -> None:
    config = _underpowered_config()
    _, _, task, _, details = closure._load_source_campaign(config)
    assert ("raw_calibrated", 7) in details
    root = Path(config.source_root)
    assert closure._source_result_manifest(root, closure.SEEDS) == (
        closure.SOURCE_RESULT_MANIFEST_SHA256
    )
    assert {
        regime: closure._dataset_hash(dataset)
        for regime, dataset in closure._datasets(task).items()
    } == closure.EXPECTED_DATASET_HASHES


def test_orbit_average_repeats_exact_two_sheet_barycenters() -> None:
    values = torch.tensor(
        [[1.0, 3.0], [2.0, 8.0], [5.0, 7.0], [4.0, 2.0]]
    )
    inverse = torch.tensor([0, 1, 0, 1])
    repeated, unique = closure.orbit_average(values, inverse)
    expected = torch.tensor([[3.0, 5.0], [3.0, 5.0]])
    assert torch.equal(unique, expected)
    assert torch.equal(repeated, expected[inverse])
    assert closure.paired_state_identity(repeated, inverse) == 0.0


def test_fiber_shuffle_preserves_marginal_and_pair_identity() -> None:
    unique = torch.tensor([[1.0], [4.0], [9.0]])
    inverse = torch.tensor([0, 0, 1, 1, 2, 2])
    shuffled = closure.shuffled_barycenters(unique, inverse, shift=1)
    assert closure.paired_state_identity(shuffled, inverse) == 0.0
    assert torch.equal(torch.sort(shuffled[:, 0]).values, torch.tensor([1.0, 1.0, 4.0, 4.0, 9.0, 9.0]))
    assert not torch.equal(shuffled, unique[inverse])
    with pytest.raises(ValueError, match="nonzero fiber rotation"):
        closure.shuffled_barycenters(unique, inverse, shift=0)


def test_task_sufficiency_uses_all_three_locked_thresholds() -> None:
    config = closure.CausalClosureConfig()
    baseline = {
        "exact_bin_accuracy": 0.95,
        "mean_circular_error_radians": 0.10,
        "mean_target_cross_entropy": 0.30,
    }
    passing = {
        "exact_bin_accuracy": 0.921,
        "mean_circular_error_radians": (
            0.10 + config.circular_error_increase_ceiling - 0.001
        ),
        "mean_target_cross_entropy": 0.399,
    }
    assert closure.task_sufficiency(passing, baseline, config)[0]
    failures = {
        "exact_bin_accuracy": 0.919,
        "mean_circular_error_radians": (
            0.10 + config.circular_error_increase_ceiling + 0.001
        ),
        "mean_target_cross_entropy": 0.401,
    }
    for metric, value in failures.items():
        failed = dict(passing)
        failed[metric] = value
        assert not closure.task_sufficiency(failed, baseline, config)[0]


def test_jensen_shannon_and_four_reynolds_regimes() -> None:
    left = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    right = torch.tensor([[0.4, 0.6], [0.9, 0.1]])
    assert closure.jensen_shannon(left, left) == pytest.approx(0.0, abs=1e-15)
    assert closure.jensen_shannon(left, right) == pytest.approx(
        closure.jensen_shannon(right, left)
    )
    assert closure.jensen_shannon(left, right) > 0.0
    assert closure._regime_name(False, False) == "cover_required_after_sublayer"
    assert closure._regime_name(False, True) == "invariant_synthesis"
    assert closure._regime_name(True, True) == "quotient_already_closed"
    assert closure._regime_name(True, False) == "quotient_corruption"


@pytest.mark.parametrize(
    ("valid", "counts", "expected"),
    [
        (False, _counts(), ("invalid", False)),
        (True, _counts(), ("frontend_causal_quotient_closed", True)),
        (
            True,
            _counts(learned_calibrated_equivariant={cut: 0 for cut in closure.CUTS}),
            ("analytic_only_frontend_closure", False),
        ),
        (
            True,
            _counts(
                analytic_calibrated={cut: 2 for cut in closure.CUTS},
                learned_calibrated_equivariant={
                    "pre_block": 0,
                    "block0_post_attention": 5,
                    "block0_post_mlp": 5,
                    "full": 5,
                },
            ),
            ("learned_frontend_requires_transformer_synthesis", False),
        ),
        (
            True,
            _counts(analytic_calibrated={cut: 2 for cut in closure.CUTS}),
            ("structured_frontend_not_causally_sufficient", False),
        ),
        (
            True,
            _counts(analytic_calibrated={"block0_post_attention": 3}),
            ("mixed_frontend_causal_closure", False),
        ),
    ],
)
def test_locked_campaign_classification(
    valid: bool,
    counts: dict[str, dict[str, int]],
    expected: tuple[str, bool],
) -> None:
    assert closure.classify_campaign(
        valid=valid,
        cut_pass_counts=counts,
        config=closure.CausalClosureConfig(),
    ) == expected


def test_campaign_reuse_requires_every_expected_cell_and_artifact(
    tmp_path: Path,
) -> None:
    config = _underpowered_config()
    result_path = tmp_path / "result.json"
    result_path.write_text("{}\n", encoding="utf-8")
    diagnostics_path = tmp_path / "diagnostics.npz"
    with diagnostics_path.open("wb") as handle:
        np.savez_compressed(handle, value=np.asarray([1]))
    implementation = "implementation"
    entry = {
        "condition": "raw_calibrated",
        "seed": 7,
        "path": str(result_path),
        "result_sha256": closure._sha256(result_path),
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_sha256": closure._sha256(diagnostics_path),
    }
    campaign = {
        "status": "completed",
        "schema_version": closure.SCHEMA_VERSION,
        "configuration": closure._json_config(config),
        "implementation_sha256": implementation,
        "results": [entry],
    }
    assert closure._campaign_reusable(campaign, config, implementation)
    campaign["results"] = []
    assert not closure._campaign_reusable(campaign, config, implementation)
    campaign["results"] = [entry]
    result_path.write_text(json.dumps({"changed": True}), encoding="utf-8")
    assert not closure._campaign_reusable(campaign, config, implementation)
