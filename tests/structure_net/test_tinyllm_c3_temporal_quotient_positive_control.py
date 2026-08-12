from pathlib import Path

from experiments.structure_net.tinyllm_c3_temporal_quotient_positive_control import (
    CAMPAIGN_SHA256,
    RESULT_MANIFEST_SHA256,
    build_diagnostic,
)


ROOT = Path(
    "data/experiments/tinyllm_c3_temporal_quotient/"
    "20260811_d6_preregistered"
)


def test_no_checkpoint_diagnostic_localizes_positive_control_failure() -> None:
    record = build_diagnostic(ROOT)
    assert record["classification"] == (
        "invariant_sensor_valid_trained_continuation_readout_extrapolation_unreliable"
    )
    assert record["campaign"]["sha256"] == CAMPAIGN_SHA256
    assert record["campaign"]["result_manifest_sha256"] == RESULT_MANIFEST_SHA256
    assert record["counts"] == {
        "natural_task_pass": 2,
        "representation_pass": 5,
        "causal_all_cuts_pass": 5,
        "checkpoints_loaded": 0,
        "optimizer_steps": 0,
        "trained_parameters": 0,
    }
    assert all(record["gates"].values())


def test_fixed_no_model_route_passes_both_fresh_shifts() -> None:
    record = build_diagnostic(ROOT)
    assert record["fixed_no_model_natural_task_gate"] is True
    for regime in ("composition", "extrapolation"):
        cell = record["fixed_no_model_positive_control"][regime]
        assert cell["temporal_prediction_correlation"] > 0.9999
        assert cell["temporal_prediction_rmse"] < 0.01
        metrics = cell["fixed_interval_metrics"]
        assert metrics["exact_bin_accuracy"] > 0.95
        assert metrics["posterior_mean_correlation"] > 0.9995


def test_failed_seed_endpoints_are_preserved_exactly() -> None:
    record = build_diagnostic(ROOT)
    by_seed = {cell["seed"]: cell for cell in record["checkpoint_cells"]}
    assert by_seed[7]["failed_natural_task_endpoints"] == []
    assert by_seed[53]["failed_natural_task_endpoints"] == []
    assert by_seed[17]["failed_natural_task_endpoints"] == [
        "extrapolation:posterior_mean_correlation",
        "extrapolation:exact_bin_accuracy",
        "extrapolation:target_cross_entropy",
    ]
    assert by_seed[29]["failed_natural_task_endpoints"] == [
        "extrapolation:target_cross_entropy"
    ]
    assert by_seed[41]["failed_natural_task_endpoints"] == [
        "extrapolation:exact_bin_accuracy"
    ]
