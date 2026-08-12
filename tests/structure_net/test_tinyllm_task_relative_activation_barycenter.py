from __future__ import annotations

import torch

import experiments.structure_net.tinyllm_task_relative_activation_barycenter as study
from experiments.structure_net.tinyllm_predictive_circle import _tinyllm_config
from structure_net.components.models import TinyLLMModel


def _small_config(**changes: object) -> study.TaskBarycenterConfig:
    values: dict[str, object] = {
        "presets": ("d6",),
        "regimes": ("training_support",),
        "fiber_pairs": 8,
        "batch_size": 16,
        "device": "cpu",
        "allow_underpowered": True,
    }
    values.update(changes)
    return study.TaskBarycenterConfig(**values)


def test_primary_configuration_and_preregistration_are_locked() -> None:
    config = study.TaskBarycenterConfig()
    assert config.presets == ("d6", "d8")
    assert config.regimes == ("training_support", "outside_range")
    assert config.fiber_pairs == 512
    assert config.accuracy_loss_ceiling == 0.03
    assert config.posterior_js_ceiling == 0.02
    assert study._sha256(study.PREREGISTRATION_PATH) == study.PREREGISTRATION_SHA256


def test_locked_sources_and_checkpoint_metadata_validate() -> None:
    task, checkpoints, hashes = study._validate_sources(_small_config())
    assert task.phase_bins == 16
    assert set(checkpoints) == {
        ("d6", "cosine_interval"),
        ("d6", "phase_circle"),
    }
    assert hashes["source_task_result"] == study.SOURCE_TASK_SHA256


def test_exact_fibers_hold_nuisance_and_change_only_task_sheet() -> None:
    task = study.CircleTaskConfig()
    for regime in study.REGIMES:
        cohort = study.generate_exact_task_fibers(
            task, pair_count=32, seed=991, regime=regime
        )
        assert cohort.contract["pass"] is True
        assert cohort.contract["cosine_target_maximum_pair_difference"] == 0.0
        assert cohort.contract["shared_nuisance_byte_identical"] is True
        assert cohort.contract["shared_prequantization_noise_byte_identical"] is True
        assert cohort.contract["serialized_sheets_differ_every_fiber"] is True
        assert cohort.contract["minimum_semantic_control_cosine_change"] >= 0.50
        assert torch.equal(cohort.semantic_partner, torch.arange(32) ^ 1)


def test_barycenter_and_semantic_reassignment_are_exact() -> None:
    values = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
    repeated, unique = study._fiber_barycenters(values)
    assert torch.equal(repeated[0], repeated[1])
    assert torch.equal(repeated[2], repeated[3])
    reassigned = study._semantic_reassignment(unique)
    assert torch.equal(reassigned[0], unique[1])
    assert torch.equal(reassigned[1], unique[1])
    assert torch.equal(reassigned[2], unique[0])
    assert torch.equal(reassigned[3], unique[0])


def test_task_sufficiency_gate_is_simultaneous() -> None:
    task = study.CircleTaskConfig()
    cohort = study.generate_exact_task_fibers(
        task, pair_count=8, seed=733, regime="training_support"
    )
    baseline = cohort.cosine_targets.double()
    passing = study._candidate_record(
        baseline,
        baseline,
        cohort.cosine_targets,
        "cosine_interval",
        task,
        cohort,
        _small_config(),
    )
    assert passing["gates"]["task_sufficient"] is True
    uniform = torch.full_like(baseline, 1.0 / task.phase_bins)
    failing = study._candidate_record(
        uniform,
        baseline,
        cohort.cosine_targets,
        "cosine_interval",
        task,
        cohort,
        _small_config(),
    )
    assert failing["gates"]["task_sufficient"] is False


def test_mature_front_requires_all_later_cuts_and_both_regimes() -> None:
    cuts = ("a", "b", "c")
    task_result = {"regimes": {}}
    for regime in study.REGIMES:
        task_result["regimes"][regime] = {
            "cuts": {
                cut: {
                    "correct_barycenter": {
                        "gates": {"task_sufficient": cut in {"b", "c"}}
                    }
                }
                for cut in cuts
            }
        }
    assert study.mature_front(task_result, cuts) == "b"
    task_result["regimes"]["outside_range"]["cuts"]["c"][
        "correct_barycenter"
    ]["gates"]["task_sufficient"] = False
    assert study.mature_front(task_result, cuts) is None


def test_specificity_requires_failure_in_every_regime() -> None:
    task_result = {
        "regimes": {
            "training_support": {
                "cuts": {
                    "x": {
                        "correct_barycenter": {
                            "gates": {"task_sufficient": False}
                        }
                    }
                }
            },
            "outside_range": {
                "cuts": {
                    "x": {
                        "correct_barycenter": {
                            "gates": {"task_sufficient": True}
                        }
                    }
                }
            },
        }
    }
    assert study._all_regimes_fail(
        task_result, "x", "correct_barycenter"
    ) is False
    task_result["regimes"]["outside_range"]["cuts"]["x"][
        "correct_barycenter"
    ]["gates"]["task_sufficient"] = False
    assert study._all_regimes_fail(
        task_result, "x", "correct_barycenter"
    ) is True


def test_tiny_model_continuation_replays_every_cut() -> None:
    task = study.CircleTaskConfig()
    model = TinyLLMModel(_tinyllm_config("tiny", task, seed=17)).eval()
    cohort = study.generate_exact_task_fibers(
        task, pair_count=2, seed=881, regime="training_support"
    )
    inputs = cohort.input_ids
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long)
    captured, baseline = study._capture_batch(model, inputs, answer_ids)
    for cut, state in captured.items():
        replay = study.continue_from_cut(model, cut, state, answer_ids)
        assert torch.max(torch.abs(replay - baseline)).item() <= 2e-6


def test_final_cut_full_and_query_only_barycenters_are_output_identical() -> None:
    task = study.CircleTaskConfig()
    model = TinyLLMModel(_tinyllm_config("tiny", task, seed=19)).eval()
    cohort = study.generate_exact_task_fibers(
        task, pair_count=2, seed=887, regime="training_support"
    )
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long)
    captured, baseline = study._capture_batch(model, cohort.input_ids, answer_ids)
    final_cut = study.cut_names(model.config.n_layer)[-1]
    state = captured[final_cut]
    full_barycenter, _ = study._fiber_barycenters(state)

    query_only = state.clone()
    query_only[:, -1, :] = full_barycenter[:, -1, :]
    context_only = state.clone()
    context_only[:, :-1, :] = full_barycenter[:, :-1, :]

    full_posterior = study.continue_from_cut(
        model, final_cut, full_barycenter, answer_ids
    )
    query_posterior = study.continue_from_cut(
        model, final_cut, query_only, answer_ids
    )
    context_posterior = study.continue_from_cut(
        model, final_cut, context_only, answer_ids
    )

    assert torch.equal(full_posterior, query_posterior)
    assert torch.equal(context_posterior, baseline)
