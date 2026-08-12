from __future__ import annotations

from pathlib import Path

import torch

from experiments.structure_net.tinyllm_c3_temporal_quotient_training import (
    ARMS,
    STAGE0_REGISTRATION_SHA256,
    C3TaskConfig,
    LifecycleConfig,
    _arm_initialization_contract,
    _feature_contract,
    _lifecycle_task,
    _source_hashes,
    _state_digest,
    build_system,
    exact_resume_lifecycle,
    generate_evaluation_dataset,
    generate_training_dataset,
    learned_registered_state_contract,
    parameter_counts,
    protocol_material,
    validate_training_protocol,
)


def test_registered_training_support_is_paired_and_compositional() -> None:
    task, config = _lifecycle_task("cpu")
    dataset, pair_batches, data_hash, batch_hash = protocol_material(task, config)
    contract = validate_training_protocol(dataset)
    assert contract["pass"] is True
    assert contract["maximum_pair_target_error"] == 0.0
    assert contract["maximum_pair_calibration_error"] == 0.0
    assert contract["all_pairs_use_distinct_decks"] is True
    assert contract["canonical_action_token_error_count"] == 0
    assert dataset.saturation_count == 0
    assert pair_batches.shape == (2, 4)
    assert len(data_hash) == len(batch_hash) == 64

    latent_calibration = dataset.calibration[0::2]
    changed = torch.stack(
        (
            (latent_calibration[:, 0] != 1.2),
            (latent_calibration[:, 1] != 0.0),
            (latent_calibration[:, 2] != 0.0),
        ),
        dim=-1,
    )
    assert bool(torch.all(changed.sum(-1) <= 1))

    composition = generate_evaluation_dataset(task, "composition", 96)
    combined = torch.stack(
        (
            (composition.calibration[:, 0] != 1.2),
            (composition.calibration[:, 1] != 0.0),
            (composition.calibration[:, 2] != 0.0),
        ),
        dim=-1,
    )
    assert bool(torch.any(combined.sum(-1) == 3))


def test_exact_c3_function_class_and_raw_sheet_visibility() -> None:
    task, config = _lifecycle_task("cpu")
    dataset = generate_evaluation_dataset(task, "composition", 96)
    for arm in ARMS:
        system = build_system(task, config, arm, torch.device("cpu"))
        contract = _feature_contract(system, dataset)
        assert contract["pass"] is True
        if arm == "raw":
            assert contract["maximum_nonidentity_change"] > 1e-3
        else:
            assert contract["maximum_nonidentity_change"] <= 1e-5

    registered = learned_registered_state_contract(
        task, config, dataset, torch.device("cpu")
    )
    assert registered["pass"] is True
    assert registered["initial"]["pass"] is True
    assert registered["deterministic_perturbed"]["pass"] is True


def test_matched_initialization_and_closed_parameter_accounting() -> None:
    task, config = _lifecycle_task("cpu")
    records = {}
    counts = {}
    for arm in ARMS:
        system = build_system(task, config, arm, torch.device("cpu"))
        values = parameter_counts(system)
        assert values["total"] == (
            values["tinyllm"]
            + values["sequence_injection"]
            + values["learned_encoder"]
        )
        records[arm] = {
            "initial_tinyllm_sha256": _state_digest(system.model),
            "initial_injection_sha256": _state_digest(system.sequence_embedding),
        }
        counts[arm] = values
    contract = _arm_initialization_contract(records)
    assert contract["pass"] is True
    assert counts["analytic"]["tinyllm"] == counts["learned_c3"]["tinyllm"]
    assert (
        counts["analytic"]["sequence_injection"]
        == counts["learned_c3"]["sequence_injection"]
    )
    assert counts["analytic"]["learned_encoder"] == 0
    assert counts["learned_c3"]["learned_encoder"] > 0


def test_two_step_checkpoint_resume_is_tensor_exact(tmp_path: Path) -> None:
    task, config = _lifecycle_task("cpu")
    sources = _source_hashes()
    result = exact_resume_lifecycle(
        task,
        config,
        "learned_c3",
        torch.device("cpu"),
        tmp_path,
        sources,
    )
    assert result["pass"] is True
    assert all(result["exact_resume"].values())
    assert result["finite_history"] is True
    assert result["nonzero_parameter_change"] is True
    assert all(
        value == 0.0 for value in result["maximum_posterior_resume_error"].values()
    )
    assert Path(result["resume_checkpoint"]["path"]).is_file()


def test_stage0_sources_and_modes_are_frozen() -> None:
    sources = _source_hashes()
    assert sources["stage0_registration"] == STAGE0_REGISTRATION_SHA256
    assert len(sources) == 5
    cpu_task, cpu = _lifecycle_task("cpu")
    cuda_task, cuda = _lifecycle_task("cuda")
    assert (cpu.preset, cpu.steps, cpu.split_step) == ("tiny", 2, 1)
    assert (cuda.preset, cuda.steps, cuda.split_step) == ("d6", 64, 32)
    assert cpu_task.vocab_size == 256
    assert cuda_task.vocab_size == 50_257


def test_lifecycle_config_rejects_unpaired_or_invalid_resume() -> None:
    C3TaskConfig()
    try:
        LifecycleConfig(
            preset="tiny",
            steps=2,
            split_step=2,
            train_samples=64,
            batch_size=8,
            evaluation_samples=16,
        )
    except ValueError as error:
        assert "resume split" in str(error)
    else:
        raise AssertionError("invalid resume split was accepted")
