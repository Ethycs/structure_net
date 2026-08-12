from __future__ import annotations

import torch

from experiments.structure_net.tinyllm_c3_temporal_quotient_analysis import (
    CAUSAL_CUTS,
    ProbeConfig,
    build_probe_datasets,
    build_task_datasets,
    causal_orbit_analysis,
    extract_representations,
    generate_shift_dataset,
    raw_reynolds_analysis,
    representation_analysis,
    target_derangement,
    validate_split_contract,
)
from experiments.structure_net.tinyllm_c3_temporal_quotient_training import (
    _lifecycle_task,
    build_system,
)


def _probe_config() -> ProbeConfig:
    return ProbeConfig(
        train_latents=32,
        validation_latents=16,
        test_latents=24,
        steps=4,
        width=8,
        batch_size=16,
        validation_interval=2,
        patience=2,
        extraction_batch_size=32,
    )


def test_fresh_splits_are_disjoint_and_final_views_match() -> None:
    task, _ = _lifecycle_task("cpu")
    config = _probe_config()
    probes = build_probe_datasets(task, config)
    tasks = build_task_datasets(task, config.test_latents)
    contract = validate_split_contract(probes, tasks)
    assert contract["pass"] is True
    assert contract["fit_final_disjoint"] is True
    assert contract["one_sheet_all_sheet_final_match"] is True
    assert all(value == 0 for value in contract["saturation_counts"].values())
    permutation = target_derangement(probes["composition"])
    bins = probes["composition"].target_bins[0::3]
    assert torch.all(bins[permutation] != bins)
    assert torch.equal(torch.sort(permutation).values, torch.arange(len(permutation)))


def test_structured_action_is_preserved_through_every_cut() -> None:
    task, lifecycle = _lifecycle_task("cpu")
    dataset = generate_shift_dataset(
        task,
        regime="composition",
        latent_count=16,
        seed=331_003,
        all_sheets=True,
    )
    for arm in ("analytic", "learned_c3"):
        system = build_system(task, lifecycle, arm, torch.device("cpu"))
        features = extract_representations(system, dataset, torch.device("cpu"), 32)
        for cut, value in features.items():
            orbit = value.reshape(16, 3, -1)
            assert float((orbit - orbit[:, :1]).abs().max()) <= 1e-5, cut


def test_registered_probes_emit_every_primary_endpoint() -> None:
    task, lifecycle = _lifecycle_task("cpu")
    config = _probe_config()
    datasets = build_probe_datasets(task, config)
    system = build_system(task, lifecycle, "analytic", torch.device("cpu"))
    result = representation_analysis(
        system,
        datasets,
        config,
        seed=7,
        device=torch.device("cpu"),
    )
    assert set(result["cuts"]) == set(CAUSAL_CUTS)
    for cut in CAUSAL_CUTS:
        for regime in ("composition", "extrapolation"):
            semantic = result["cuts"][cut]["semantic"]["evaluations"][regime]
            deck = result["cuts"][cut]["conditional_deck"]["evaluations"][regime]
            assert set(semantic) == {"target_correlation", "target_rmse"}
            assert "balanced_accuracy" in deck
            assert "conditional_log_loss_gain" in deck


def test_causal_identity_and_structured_barycenter_are_exact() -> None:
    task, lifecycle = _lifecycle_task("cpu")
    dataset = generate_shift_dataset(
        task,
        regime="composition",
        latent_count=16,
        seed=331_003,
        all_sheets=True,
    )
    system = build_system(task, lifecycle, "analytic", torch.device("cpu"))
    result = causal_orbit_analysis(
        system,
        dataset,
        task,
        device=torch.device("cpu"),
        batch_size=24,
    )
    assert set(result["cuts"]) == set(CAUSAL_CUTS)
    for cell in result["cuts"].values():
        assert cell["maximum_orbit_state_error"] <= 1e-5
        assert cell["maximum_identity_replay_logit_error"] <= 2e-6
        assert cell["orbit_barycenter_preservation"]["pass"] is True


def test_raw_reynolds_scan_covers_every_sublayer() -> None:
    task, lifecycle = _lifecycle_task("cpu")
    dataset = generate_shift_dataset(
        task,
        regime="composition",
        latent_count=8,
        seed=331_003,
        all_sheets=True,
    )
    system = build_system(task, lifecycle, "raw", torch.device("cpu"))
    result = raw_reynolds_analysis(
        system,
        dataset,
        task,
        device=torch.device("cpu"),
        latent_limit=8,
    )
    assert result["status"] == "completed"
    assert len(result["sublayers"]) == 2 * system.model.config.n_layer
    assert all(
        cell["regime"] in {"cover_required", "synthesis", "closed", "corruption"}
        for cell in result["sublayers"]
    )
