from __future__ import annotations

from experiments.structure_net.tinyllm_calibrated_architecture_replication_preflight import (
    CONDITIONS,
    PROSPECTIVE_PRESETS,
    SEEDS,
    build_preflight,
    experiment_grid,
    model_parameter_count,
    system_parameter_count,
)
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


def test_grid_contains_every_new_matched_cell_once() -> None:
    grid = experiment_grid()
    keys = {(row["preset"], row["condition"], row["seed"]) for row in grid}
    assert len(grid) == len(PROSPECTIVE_PRESETS) * len(CONDITIONS) * len(SEEDS) == 30
    assert len(keys) == len(grid)


def test_parameter_counts_match_retained_d8_source() -> None:
    task = CircleTaskConfig(train_samples=4_096)
    assert model_parameter_count(task, "d6") == 29_956_608
    assert model_parameter_count(task, "d8") == 50_965_504
    assert model_parameter_count(task, "d10") == 81_418_240
    assert system_parameter_count(task, "d8", "raw_calibrated") == 50_970_112
    assert system_parameter_count(task, "d8", "analytic_calibrated") == 50_966_528
    assert (
        system_parameter_count(task, "d8", "learned_calibrated_equivariant")
        == 50_976_625
    )


def test_preflight_validates_anchors_protocols_and_costs() -> None:
    preflight = build_preflight()
    assert preflight["valid"] is True
    assert preflight["optimization_steps_executed"] == 0
    assert preflight["grid"]["new_cell_count"] == 30
    assert all(
        row["matched"]
        for row in preflight["protocol"]["preset_hashes_by_seed"].values()
    )
    controls = preflight["sensor_positive_controls"]
    assert all(
        row["pass"] for row in controls["analytic_canonicalizer"].values()
    )
    assert controls["learned_encoder_group_contract"]["pass"] is True
    projection = preflight["resource_projection"]
    assert 55.0 < projection["projected_new_gpu_minutes"] < 65.0
    assert 45.0 < projection["projected_new_training_probe_gpu_minutes"] < 55.0
    assert 8.0 < projection["projected_new_causal_gpu_minutes"] < 12.0
    assert 5.0 < projection["projected_new_checkpoint_gib"] < 7.0
