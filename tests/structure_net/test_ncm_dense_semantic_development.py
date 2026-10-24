from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.structure_net import ncm_dense_semantic_development as run
from experiments.neural_architecture_lab import finalize_ncm_dense_semantic_development as audit


def test_inference_pilot_covers_short_and_longest_long_without_quality_selection():
    cases = {run.reference.COHORTS[0]: [{"id": "s0", "source_ids": [1]}, {"id": "s1", "source_ids": [1, 2]}],
        run.reference.COHORTS[1]: [{"id": "l0", "source_ids": [1, 2]}, {"id": "l1", "source_ids": [1, 2, 3]}]}
    assert [r["id"] for r in run.inference_probes(cases)] == ["s0", "l1"]


def test_judge_pilot_includes_four_distinct_predeclared_cases_and_accepts_negatives():
    rows = [{"id": f"{split}-{i}", "split": split, "source_tokens": n, "text": "candidate"}
        for split in run.reference.COHORTS for i, n in enumerate((100, 90, 120))]
    expected = [rows[i]["id"] for i in (0, 2, 3, 5)]
    assert [r["id"] for r in run.judge_probes(rows)] == expected
    rows[2]["text"] = ""
    assert [r["id"] for r in run.judge_probes(rows)] == expected
    assert run.judge_pilot_valid({"cases": 4, "valid": 4, "passed": 0})
    assert not run.judge_pilot_valid({"cases": 4, "valid": 3, "passed": 3})


def test_source_prediction_api_rejects_every_label_channel_before_model_access():
    for field in ("word_ids", "concept_ids", "target_text", "teacher_latent"):
        with pytest.raises(ValueError, match="cannot receive labels"):
            run.predict(None, None, {field: "label"}, {}, None)


def test_constructor_parity_binds_raw_words_latents_and_actual_codes():
    output = {"id": "a", "raw_word_ids": [7, 50256] + [0] * 190,
        "coarse_codes": [[0] * n for n in (16, 24, 40, 56, 80, 104, 128, 160)], "latent_sha256": "a" * 64}
    expected = {"a": {"raw_word_ids": output["raw_word_ids"], "actual_coarse_codes": [0] * 608, "latent_sha256": "a" * 64}}
    run.parity_check([output], expected)
    changed = deepcopy(output)
    changed["raw_word_ids"][0] += 1
    with pytest.raises(ValueError, match="registered TRAIN"):
        run.parity_check([changed], expected)


def test_joint_quality_is_not_the_smaller_marginal_pass_count():
    secondary = [{"id": str(i), "valid": True, "passed": i < 61} for i in range(64)]
    primary = [{"id": str(i), "valid": True, "passed": i not in (58, 59, 60)} for i in range(64)]
    result = audit.joint_counts(secondary, primary)
    assert sum(r["passed"] for r in secondary) == sum(r["passed"] for r in primary) == 61
    assert result["passed"] == 58 and result["valid"] == 64
    assert result["task_accuracy"] == 58 / 64
    assert audit.joint_counts(secondary, None) is None
    primary[0]["id"] = "different"
    with pytest.raises(ValueError, match="same ordered"):
        audit.joint_counts(secondary, primary)


def test_invalid_joint_decisions_keep_the_fixed_denominator():
    secondary = [{"id": str(i), "valid": True, "passed": True} for i in range(64)]
    primary = deepcopy(secondary)
    primary[0].update(valid=False, passed=None)
    result = audit.joint_counts(secondary, primary)
    assert result["cases"] == 64 and result["passed"] == 63 and result["invalid"] == 1
    assert result["task_accuracy"] is None
    assert result["wilson_95"] is None
    assert result["all_case_pass_upper_bound"] == 1


def test_raw_output_replay_preserves_global_eos_and_complete_source_frame():
    source = {"id": "a", "source_id": "s", "group_id": "g", "source_sha256": "hash", "source": "source",
        "spans": [], "split": run.reference.COHORTS[0], "source_ids": [1, 2], "source_frame": 256}
    value = {**source, "text": "w7", "raw_word_ids": [7, 50256] + [0] * 190,
        "coarse_codes": [[0] * n for n in (16, 24, 40, 56, 80, 104, 128, 160)],
        "global_EOS_position": 1, "latent_sha256": "a" * 64, "source_tokens": 2,
        "source_truncated": False, "fixed_generator_passes": 9, "AR_fallback": False,
        "oracle_target_codes": False, "output_repair": False}
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: " ".join("w" + str(x) for x in ids))
    audit.replay_outputs([value], [source], tokenizer)
    changed = deepcopy(value)
    changed["raw_word_ids"][0] = 9
    with pytest.raises(ValueError, match="raw complete"):
        audit.replay_outputs([changed], [source], tokenizer)
    value["output_repair"] = True
    with pytest.raises(ValueError, match="source/order/decoding"):
        audit.replay_outputs([value], [source], tokenizer)
