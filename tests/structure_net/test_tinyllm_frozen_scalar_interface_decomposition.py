import json
from pathlib import Path

import pytest
import torch

import experiments.structure_net.tinyllm_frozen_scalar_interface_decomposition as scalar


def test_primary_configuration_is_locked() -> None:
    config = scalar.ScalarInterfaceConfig(device="cpu")
    assert config.presets == ("d6", "d10")
    assert config.conditions == (
        "analytic_calibrated",
        "learned_calibrated_equivariant",
    )
    assert config.seeds == (7, 17, 29, 41, 53)
    assert config.scalar_grid_points == 4_097
    with pytest.raises(ValueError, match="primary scalar-interface configuration"):
        scalar.ScalarInterfaceConfig(device="cpu", scalar_grid_points=2_049)
    reduced = scalar.ScalarInterfaceConfig(
        presets=("d6",),
        conditions=("analytic_calibrated",),
        seeds=(7,),
        samples_per_regime=64,
        scalar_grid_points=129,
        batch_size=64,
        device="cpu",
        allow_underpowered=True,
    )
    assert reduced.allow_underpowered is True


def test_preregistration_and_source_manifests_are_locked() -> None:
    assert scalar._sha256(scalar.PREREGISTRATION_PATH) == (
        scalar.PREREGISTRATION_SHA256
    )
    campaign, _task, details = scalar._source_details(
        scalar.ScalarInterfaceConfig(device="cpu")
    )
    assert campaign["aggregates"]["classification"] == (
        "structured_closure_not_architecture_stable"
    )
    assert len(details) == 20
    observed_failures = {
        key: not detail["gates"]["task_adequacy_pass"]
        for key, detail in details.items()
    }
    for (preset, condition), failed_seeds in scalar.SOURCE_FAILURES.items():
        for seed in scalar.SEEDS:
            assert observed_failures[(preset, condition, seed)] == (
                seed in failed_seeds
            )


def test_primary_cohort_hashes_match_source() -> None:
    config = scalar.ScalarInterfaceConfig(device="cpu")
    campaign, task, _details = scalar._source_details(config)
    datasets = scalar._datasets(task, config)
    assert scalar._dataset_hashes(datasets) == scalar.EXPECTED_DATASET_HASHES
    assert campaign["dataset_hashes"] == scalar.EXPECTED_DATASET_HASHES


def test_task_metrics_and_floor_gate_are_exact() -> None:
    target = torch.eye(4)
    posterior = torch.eye(4) * 0.9 + 0.1 / 4
    metrics = scalar.task_metrics(posterior, target)
    assert metrics["exact_bin_accuracy"] == 1.0
    assert metrics["mean_circular_error_radians"] == 0.0
    assert scalar.task_adequacy_pass(metrics, 1.0)
    assert not scalar.task_adequacy_pass(
        {**metrics, "exact_bin_accuracy": 0.74}, 0.75
    )


@pytest.mark.parametrize(
    ("exact", "oracle", "expected"),
    [
        (True, True, "sensor_scalar_estimation_failure"),
        (False, True, "scalar_coordinate_or_boundary_failure"),
        (False, False, "continuation_or_answer_row_failure"),
        (True, False, "invalid_oracle_resolution"),
    ],
)
def test_failed_cell_classification_is_exhaustive(
    exact: bool, oracle: bool, expected: str
) -> None:
    assert scalar.classify_failed_cell(exact, oracle) == expected


def _aggregate_detail(
    preset: str,
    condition: str,
    seed: int,
    *,
    source_failed: bool,
    exact: bool,
    oracle: bool,
    negative: bool = False,
    shuffled: bool = False,
) -> dict:
    return {
        "preset": preset,
        "condition": condition,
        "seed": seed,
        "classification": (
            scalar.classify_failed_cell(exact, oracle)
            if source_failed
            else "source_passing_positive_control"
        ),
        "gates": {
            "source_failed": source_failed,
            "exact_cosine_both_shifts": exact,
            "oracle_both_shifts": oracle,
            "negative_cosine_both_shifts": negative,
            "shuffled_cosine_both_shifts": shuffled,
            "validity": True,
        },
    }


def _synthetic_population(*, exact_repairs: bool, oracle_repairs: bool) -> list[dict]:
    details = []
    for preset in scalar.PRESETS:
        for condition in scalar.CONDITIONS:
            failed = set(scalar.SOURCE_FAILURES[(preset, condition)])
            for seed in scalar.SEEDS:
                source_failed = seed in failed
                details.append(
                    _aggregate_detail(
                        preset,
                        condition,
                        seed,
                        source_failed=source_failed,
                        exact=exact_repairs if source_failed else True,
                        oracle=oracle_repairs if source_failed else True,
                    )
                )
    return details


def test_population_classification_separates_nested_hypotheses() -> None:
    config = scalar.ScalarInterfaceConfig(device="cpu")
    exact = scalar.aggregate_details(
        _synthetic_population(exact_repairs=True, oracle_repairs=True), config
    )
    assert exact["valid"] is True
    assert exact["gates"]["uniformly_upstream_of_scalar_embedding"] is True
    assert exact["classification"] == (
        "source_failure_uniformly_upstream_of_scalar_embedding"
    )

    coordinate = scalar.aggregate_details(
        _synthetic_population(exact_repairs=False, oracle_repairs=True), config
    )
    assert coordinate["gates"][
        "one_dimensional_interface_expressively_sufficient"
    ] is True
    assert coordinate["classification"] == (
        "scalar_coordinate_failure_with_sufficient_continuation"
    )

    continuation = scalar.aggregate_details(
        _synthetic_population(exact_repairs=False, oracle_repairs=False), config
    )
    assert continuation["classification"] == "continuation_capacity_failure_present"


def test_population_specificity_control_invalidates_campaign() -> None:
    config = scalar.ScalarInterfaceConfig(device="cpu")
    details = _synthetic_population(exact_repairs=False, oracle_repairs=True)
    for detail in [item for item in details if item["gates"]["source_failed"]][:2]:
        detail["gates"]["negative_cosine_both_shifts"] = True
    aggregate = scalar.aggregate_details(details, config)
    assert aggregate["valid"] is False
    assert aggregate["classification"] == "invalid"


def test_scalar_injection_replays_actual_frozen_system() -> None:
    config = scalar.ScalarInterfaceConfig(
        presets=("d6",),
        conditions=("analytic_calibrated",),
        seeds=(7,),
        samples_per_regime=16,
        scalar_grid_points=17,
        batch_size=16,
        device="cpu",
        allow_underpowered=True,
    )
    _campaign, task, details = scalar._source_details(config)
    dataset = scalar._datasets(task, config)["composition"]
    system = scalar._load_system(
        details[("d6", "analytic_calibrated", 7)],
        task,
        "d6",
        "analytic_calibrated",
        torch.device("cpu"),
    )
    natural = scalar._natural_scalar(
        system, dataset, task, config, torch.device("cpu")
    )
    direct = scalar._direct_posteriors(
        system, dataset, task, config, torch.device("cpu")
    )
    injected = scalar._scalar_posteriors(
        system,
        dataset.paired.circle.input_ids,
        natural,
        task,
        config,
        torch.device("cpu"),
    )
    assert float((direct - injected).abs().max()) == 0.0
    assert not any(parameter.requires_grad for parameter in system.parameters())


def test_existing_result_requires_matching_diagnostics_hash(tmp_path: Path) -> None:
    config = scalar.ScalarInterfaceConfig(
        presets=("d6",),
        conditions=("analytic_calibrated",),
        seeds=(7,),
        samples_per_regime=16,
        scalar_grid_points=17,
        batch_size=16,
        device="cpu",
        allow_underpowered=True,
    )
    _campaign, _task, details = scalar._source_details(config)
    source = details[("d6", "analytic_calibrated", 7)]
    datasets = {"composition": "a", "extrapolation": "b"}
    implementation = "implementation"
    fingerprint = scalar._fingerprint(
        config,
        "d6",
        "analytic_calibrated",
        7,
        implementation,
        source,
        datasets,
    )
    directory = tmp_path / "runs/d6/analytic_calibrated/seed_7"
    directory.mkdir(parents=True)
    diagnostics = directory / "diagnostics.npz"
    diagnostics.write_bytes(b"sealed")
    result = {
        "schema_version": scalar.SCHEMA_VERSION,
        "status": "completed",
        "scientific_fingerprint": fingerprint,
        "implementation_sha256": implementation,
        "gates": {"validity": True},
        "artifacts": {
            "diagnostics": str(diagnostics),
            "diagnostics_sha256": scalar._sha256(diagnostics),
        },
    }
    (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    assert (
        scalar._existing_detail(
            tmp_path,
            config,
            "d6",
            "analytic_calibrated",
            7,
            implementation,
            source,
            datasets,
        )
        is not None
    )
    diagnostics.write_bytes(b"changed")
    assert (
        scalar._existing_detail(
            tmp_path,
            config,
            "d6",
            "analytic_calibrated",
            7,
            implementation,
            source,
            datasets,
        )
        is None
    )


def test_runner_contains_no_training_path() -> None:
    source = Path(scalar.__file__).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "requires_grad_(False)" in source
