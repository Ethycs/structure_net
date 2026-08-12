from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.structure_net.tinyllm_calibrated_architecture_replication import (
    CONDITIONS,
    PRIMARY_CONDITIONS,
    PRIMARY_PRESETS,
    PREREGISTRATION_SHA256,
    SEEDS,
    SCHEMA_VERSION,
    ArchitectureReplicationConfig,
    _existing_detail,
    _experiments,
    _fingerprint,
    _implementation_sources,
    _source_config,
    aggregate_details,
    representation_endpoint_pass,
)
from experiments.structure_net.tinyllm_calibrated_architecture_replication_preflight import (
    build_preflight,
)
from experiments.structure_net.tinyllm_calibrated_frontend_causal_closure import (
    EXPECTED_DATASET_HASHES,
)
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from neural_architecture_lab.core import Experiment


def _detail(
    preset: str,
    condition: str,
    seed: int,
    *,
    joint: bool,
    valid: bool = True,
    shuffle: bool = False,
) -> dict:
    return {
        "preset": preset,
        "condition": condition,
        "seed": seed,
        "gates": {
            "validity": valid,
            "representation_pass": joint,
            "task_adequacy_pass": joint,
            "causal_all_cuts_pass": joint,
            "joint_seed_pass": joint,
        },
        "causal": {"shuffle_seed_gates": {"pre_block": shuffle}},
    }


def test_primary_configuration_is_locked() -> None:
    ArchitectureReplicationConfig()
    with pytest.raises(ValueError, match="primary architecture-replication"):
        ArchitectureReplicationConfig(training_steps=599)
    lifecycle = ArchitectureReplicationConfig(
        presets=("d10",),
        conditions=("raw_calibrated",),
        seeds=(7,),
        training_steps=2,
        train_samples=32,
        batch_size=8,
        probe_steps=20,
        probe_train_samples=64,
        probe_validation_samples=32,
        probe_test_samples=32,
        required_seed_passes=1,
        allow_underpowered=True,
    )
    assert lifecycle.allow_underpowered is True


def test_preregistration_and_implementation_sources_are_frozen() -> None:
    sources = _implementation_sources()
    assert sources["preregistration"] == PREREGISTRATION_SHA256
    assert len(sources) == 5


def test_representation_gate_includes_conditional_log_loss() -> None:
    config = ArchitectureReplicationConfig()
    passing = {
        "cosine_pearson": 0.91,
        "balanced_accuracy": 0.54,
        "conditional_log_loss_gain_over_cosine_only": 0.019,
    }
    assert representation_endpoint_pass(passing, config)
    failing = dict(passing)
    failing["conditional_log_loss_gain_over_cosine_only"] = 0.021
    assert not representation_endpoint_pass(failing, config)


def test_source_adapter_changes_preset_without_changing_protocol() -> None:
    config = ArchitectureReplicationConfig()
    d6 = _source_config(config, "d6")
    d10 = _source_config(config, "d10")
    assert d6.preset == "d6"
    assert d10.preset == "d10"
    assert d6.training_steps == d10.training_steps == 600
    assert d6.train_samples == d10.train_samples == 4_096
    assert d6.batch_size == d10.batch_size == 64
    assert d6.allow_underpowered and d10.allow_underpowered


def test_experiment_grid_is_30_fresh_cells_with_locked_floors(tmp_path: Path) -> None:
    config = ArchitectureReplicationConfig()
    task = CircleTaskConfig(train_samples=config.train_samples)
    record = build_preflight()
    experiments = _experiments(
        config,
        task,
        tmp_path,
        "implementation",
        record,
        "preflight",
        EXPECTED_DATASET_HASHES,
    )
    assert len(experiments) == 30
    keys = {
        (
            item.parameters["preset"],
            item.parameters["condition"],
            item.seed,
        )
        for item in experiments
    }
    assert len(keys) == 30
    assert all(
        len(item.parameters["task_accuracy_floors"]) == 2
        for item in experiments
        if item.parameters["condition"] in PRIMARY_CONDITIONS
    )
    assert all(
        item.parameters["task_accuracy_floors"] == {}
        for item in experiments
        if item.parameters["condition"] == "raw_calibrated"
    )


def test_aggregate_requires_both_structured_arms_in_both_presets() -> None:
    config = ArchitectureReplicationConfig()
    details = []
    for preset in PRIMARY_PRESETS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                details.append(
                    _detail(
                        preset,
                        condition,
                        seed,
                        joint=condition in PRIMARY_CONDITIONS,
                    )
                )
    aggregate = aggregate_details(details, config)
    assert aggregate["valid"] is True
    assert aggregate["primary_hypothesis_pass"] is True
    assert aggregate["classification"] == "structured_family_replication_with_specificity"

    for row in details:
        if (
            row["preset"] == "d10"
            and row["condition"] == "learned_calibrated_equivariant"
            and row["seed"] in (7, 17)
        ):
            row["gates"]["joint_seed_pass"] = False
    aggregate = aggregate_details(details, config)
    assert aggregate["primary_hypothesis_pass"] is False
    assert aggregate["classification"] == "analytic_closure_stable_learned_family_dependent"


def test_semantic_shuffle_population_control_invalidates_campaign() -> None:
    config = ArchitectureReplicationConfig()
    details = []
    for preset in PRIMARY_PRESETS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                details.append(
                    _detail(
                        preset,
                        condition,
                        seed,
                        joint=condition in PRIMARY_CONDITIONS,
                        shuffle=(
                            preset == "d6"
                            and condition == "analytic_calibrated"
                            and seed in (7, 17)
                        ),
                    )
                )
    aggregate = aggregate_details(details, config)
    assert aggregate["valid"] is False
    assert aggregate["classification"] == "invalid"


def test_existing_detail_requires_fingerprint_and_artifact_hashes(
    tmp_path: Path,
) -> None:
    params = {
        "configuration": {"config": "value"},
        "task_config": {"task": "value"},
        "preset": "d6",
        "condition": "raw_calibrated",
        "seed": 7,
        "implementation_sha256": "implementation",
        "preflight_sha256": "preflight",
        "identifiability_contract": {"passed": True},
        "dataset_hashes": dict(EXPECTED_DATASET_HASHES),
        "task_accuracy_floors": {},
        "output_dir": str(tmp_path),
    }
    experiment = Experiment(
        id="cell",
        hypothesis_id="hypothesis",
        name="cell",
        parameters=params,
        seed=7,
    )
    directory = tmp_path / "runs" / "d6" / "raw_calibrated" / "seed_7"
    directory.mkdir(parents=True)
    artifacts = {}
    for name, content in (
        ("checkpoint", b"model"),
        ("frontend_checkpoint", b"frontend"),
        ("diagnostics", b"diagnostics"),
    ):
        path = directory / f"{name}.bin"
        path.write_bytes(content)
        artifacts[name] = str(path)
        artifacts[f"{name}_sha256"] = hashlib.sha256(content).hexdigest()
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "scientific_fingerprint": _fingerprint(experiment),
        "implementation_sha256": "implementation",
        "artifacts": artifacts,
    }
    result_path = directory / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert _existing_detail(experiment, tmp_path) == result
    Path(artifacts["diagnostics"]).write_bytes(b"changed")
    assert _existing_detail(experiment, tmp_path) is None
