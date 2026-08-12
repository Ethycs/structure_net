"""Build and store the TinyLLM sensor-noise dose-localization result."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-noise-law-dose-localization.v1"
HYPOTHESIS_ID = "tinyllm-noise-law-dose-localization-v1"
EVIDENCE_ROLE = (
    "preregistered_post_outcome_corrective_frozen_dose_localization"
)
CLASSIFICATION = "asymmetric_law_breaks_within_isotropic_window"
CAMPAIGN_SHA256 = (
    "9b05823ebdb88bd828f27699da596dc5e7dcf0c4af5e13f1664fa70e5111f9bd"
)
IMPLEMENTATION_SHA256 = (
    "bab495e0f3985c8358d90344fc3cf02986b6e138adaeb9fa01c1d38c482187c2"
)
RUNNER_SHA256 = (
    "39a72dd535f96f13bae644c74096b298b85fb8587d980211dc489ed463aeb725"
)
PREREGISTRATION_SHA256 = (
    "79913c913c7f6f41714400fd4337224f039be0466541ec0a7f26736c599b7a4a"
)
RESULT_MANIFEST_SHA256 = (
    "976545c812e428ea4b020ca46a88643cb741a6ad5c7797389e9a5e6ca81f7562"
)
SELECTED_ARRAYS_SHA256 = (
    "740c5c30f01c482fa799db1865a11c069ad3b59f474879a59f1906b94f4130f3"
)
SELECTED_CONTRACT_SHA256 = (
    "89ddaa6d726d89767e34bc2efb4ef75af8cbe098528da0f214473d01a51ac1f5"
)
SOURCE_CAMPAIGN_SHA256 = (
    "868ad0ffee546f157e701790c34a83f20bfb3116e78b2f8c5bc34dd7bfe660d7"
)
SOURCE_NOISE_FILE_SHA256 = (
    "d3771eac8e29f7940df7feaedebe74a5a78fb273cda2e70928c9be9e37ff3ba6"
)
SOURCE_NOISE_CONTENT_SHA256 = (
    "93df61bc76ed073ea241c9450e7ec3523e7a98b5ac06e58d7e920a5df07d70aa"
)
SOURCE_RUNNER_SHA256 = (
    "7bed49c064e8a2148268d2a4ab3a42ec70847a15d83c7297cff3d9dccc7970d2"
)
SOURCE_DVC_ROOT = "19f1fbbe86b6b9235eb211a88bb32aa2.dir"
SOURCE_LAKEFS_COMMIT = (
    "f3c895cdf8d5f25e8ae6a87b3f694d0bbacb24cdd14d4736d0c7dfa41399c130"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
LAWS = ("isotropic", "lab_anisotropic", "lab_biased")
CUTS = ("pre_block", "full")
DOSE_KEYS = (
    "0.000",
    "0.125",
    "0.250",
    "0.375",
    "0.500",
    "0.625",
    "0.750",
    "1.000",
)
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_noise_law_dose_localization.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-noise-law-dose-localization-preregistration.md"
)
SOURCE_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_noise_law_observed_twirl/"
    "20260810_d10_preregistered/campaign_results.json"
)
SOURCE_NOISE_FILE_PATH = Path(
    "data/experiments/tinyllm_noise_law_observed_twirl/"
    "20260810_d10_preregistered/noise_law_arrays.npz"
)
SOURCE_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_noise_law_observed_twirl.py"
)
DATASET_HASHES = {
    "composition": "b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6",
    "extrapolation": "f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=128)
def _require_source_digest(
    path: Path, expected: str, size: int, modified_ns: int
) -> None:
    del size, modified_ns
    if _sha256(path) != expected:
        raise ValueError(f"source artifact changed: {path}")


def _require_source(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ValueError(f"source artifact changed: {path}")
    stat = path.stat()
    _require_source_digest(path, expected, stat.st_size, stat.st_mtime_ns)


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, Iterable):
        return all(_finite(item) for item in value)
    return True


def _expected_configuration() -> dict[str, Any]:
    return {
        "accuracy_loss_ceiling": 0.03,
        "allow_underpowered": False,
        "analytic_feature_tolerance": 1e-6,
        "batch_size": 256,
        "calibrated_source_root": (
            "data/experiments/tinyllm_calibrated_frontend_causal/"
            "20260806_d8_preregistered"
        ),
        "circular_error_increase_ceiling": math.pi / 16.0,
        "conditions": list(CONDITIONS),
        "cross_entropy_increase_ceiling": 0.1,
        "device": "cuda:2",
        "dose_multipliers": [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1],
        "laws": list(LAWS),
        "maximum_control_seed_passes": 1,
        "natural_accuracy_loss_ceiling": 0.05,
        "natural_circular_error_increase_ceiling": math.pi / 16.0,
        "natural_cross_entropy_increase_ceiling": 0.1,
        "replay_tolerance": 2e-6,
        "required_seed_passes": 4,
        "sample_limit": None,
        "seeds": list(SEEDS),
        "source_closure_root": (
            "data/experiments/tinyllm_calibrated_frontend_causal_closure/"
            "20260810_d15_preregistered"
        ),
        "source_noise_root": (
            "data/experiments/tinyllm_noise_law_observed_twirl/"
            "20260810_d10_preregistered"
        ),
        "source_noise_sigma": 0.05,
        "source_observed_root": (
            "data/experiments/tinyllm_observed_deck_twirl/"
            "20260810_d10_preregistered"
        ),
        "source_replay_tolerance": 2e-6,
        "zero_replay_tolerance": 2e-6,
    }


def _expected_summary() -> dict[str, int]:
    return {
        "requested_stage1": 10,
        "completed_stage1": 10,
        "reused_stage1": 0,
        "requested_stage2": 10,
        "completed_stage2": 10,
        "reused_stage2": 0,
        "failed": 0,
        "excluded": 0,
        "trained_models": 0,
        "trained_frontends": 0,
        "trained_task_heads": 0,
        "fitted_noise_models": 0,
        "fitted_actions": 0,
        "fitted_observers": 0,
        "fitted_probes": 0,
    }


def _expected_stage2_counts() -> dict[str, dict[str, int]]:
    return {
        "analytic_calibrated": {
            "isotropic": 5,
            "lab_anisotropic": 4,
            "lab_biased": 1,
        },
        "learned_calibrated_equivariant": {
            "isotropic": 5,
            "lab_anisotropic": 5,
            "lab_biased": 3,
        },
    }


def _campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    aggregates = campaign.get("aggregates", {})
    stage1 = campaign.get("stage1", {})
    stage2 = campaign.get("stage2", {})
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or campaign.get("schema_version") != EXPERIMENT_SCHEMA
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != EVIDENCE_ROLE
        or campaign.get("configuration") != _expected_configuration()
        or campaign.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or campaign.get("source_digests")
        != {
            "runner": RUNNER_SHA256,
            "preregistration": PREREGISTRATION_SHA256,
            "source_noise_runner": SOURCE_RUNNER_SHA256,
        }
        or campaign.get("source_campaign_sha256") != SOURCE_CAMPAIGN_SHA256
        or campaign.get("source_noise_file_sha256")
        != SOURCE_NOISE_FILE_SHA256
        or campaign.get("source_noise_content_sha256")
        != SOURCE_NOISE_CONTENT_SHA256
        or campaign.get("source_dvc_root") != SOURCE_DVC_ROOT
        or campaign.get("source_lakefs_commit") != SOURCE_LAKEFS_COMMIT
        or campaign.get("dataset_hashes") != DATASET_HASHES
        or campaign.get("summary") != _expected_summary()
        or campaign.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or campaign.get("selected_arrays_sha256") != SELECTED_ARRAYS_SHA256
        or campaign.get("selected_noise_law_contract_sha256")
        != SELECTED_CONTRACT_SHA256
        or campaign.get("selected_noise_law_contract", {}).get("pass") is not True
        or aggregates
        != {
            "classification": CLASSIFICATION,
            "integrity_valid": True,
            "maximum_control_seed_passes": 1,
            "primary_evaluable": True,
            "primary_hypothesis_pass": False,
            "required_seed_passes": 4,
            "selected_multiplier": 0.625,
            "selected_noise_sigma": 0.03125,
            "valid": True,
            "zero_dose_control_pass": True,
        }
        or stage1.get("selected_multiplier") != 0.625
        or stage1.get("selected_noise_sigma") != 0.03125
        or stage1.get("zero_dose_control_pass") is not True
        or stage1.get("integrity_valid") is not True
        or stage1.get("selection_uses_asymmetric_outcomes") is not False
        or set(stage1.get("doses", {})) != set(DOSE_KEYS)
        or stage2.get("integrity_valid") is not True
        or stage2.get("controls_pass") is not True
        or stage2.get("isotropic_joint_population_pass") is not True
        or stage2.get("all_laws_joint_population_pass") is not False
        or len(campaign.get("stage1_results", [])) != 10
        or len(campaign.get("stage2_results", [])) != 10
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid noise-law dose-localization campaign {path}")
    if (
        stage1["doses"]["0.625"]["arms"]
        != {
            "analytic_calibrated": {"natural_utility_pass_count": 5},
            "learned_calibrated_equivariant": {
                "natural_utility_pass_count": 5
            },
        }
        or stage1["doses"]["0.750"]["arms"]
        != {
            "analytic_calibrated": {"natural_utility_pass_count": 2},
            "learned_calibrated_equivariant": {
                "natural_utility_pass_count": 5
            },
        }
        or not all(
            stage1["doses"][key]["prefix_valid"]
            for key in ("0.125", "0.250", "0.375", "0.500", "0.625")
        )
        or stage1["doses"]["0.750"]["prefix_valid"]
        or stage1["doses"]["1.000"]["prefix_valid"]
    ):
        raise ValueError(f"invalid dose selection in {path}")
    expected_counts = _expected_stage2_counts()
    for condition in CONDITIONS:
        for law in LAWS:
            record = stage2["arms"][condition]["laws"][law]
            expected = expected_counts[condition][law]
            if (
                record.get("joint_pass_count") != expected
                or record.get("natural_utility_pass_count") != expected
                or record.get("control_pass_count") != 0
                or record.get("action_pass_counts")
                != {cut: 5 for cut in CUTS}
                or record.get("twirl_pass_counts")
                != {cut: 5 for cut in CUTS}
            ):
                raise ValueError(f"invalid selected-law population in {path}")
    for source_path, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (SOURCE_CAMPAIGN_PATH, SOURCE_CAMPAIGN_SHA256),
        (SOURCE_NOISE_FILE_PATH, SOURCE_NOISE_FILE_SHA256),
        (SOURCE_RUNNER_PATH, SOURCE_RUNNER_SHA256),
    ):
        _require_source(source_path, expected)
    return campaign


def _entry_details(
    campaign: Mapping[str, Any], stage: str
) -> dict[tuple[str, int], dict[str, Any]]:
    key = "stage1_results" if stage == "isotropic_localization" else "stage2_results"
    expected_cells = {
        (condition, seed) for condition in CONDITIONS for seed in SEEDS
    }
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in campaign[key]:
        path = Path(entry["path"])
        diagnostics_path = Path(entry["diagnostics_path"])
        detail = json.loads(path.read_text(encoding="utf-8"))
        cell = (str(detail.get("condition")), int(detail.get("seed", -1)))
        if (
            _sha256(path) != entry.get("result_sha256")
            or _sha256(diagnostics_path) != entry.get("diagnostics_sha256")
            or detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("stage") != stage
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("configuration") != _expected_configuration()
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or detail.get("dataset_hashes") != DATASET_HASHES
            or detail.get("gates", {}).get("validity") is not True
            or detail.get("gates", {}).get("state_unchanged") is not True
            or detail.get("gates", {}).get("finite") is not True
            or detail.get("artifacts", {}).get("diagnostics_sha256")
            != entry.get("diagnostics_sha256")
            or not _finite(detail)
        ):
            raise ValueError(f"invalid {stage} result {path}")
        provenance = detail.get("provenance", {})
        for provenance_key in (
            "model_checkpoint",
            "frontend_checkpoint",
            "source_result",
            "observed_predecessor_result",
            "observed_predecessor_diagnostics",
        ):
            _require_source(
                Path(provenance[provenance_key]),
                provenance[f"{provenance_key}_sha256"],
            )
        detail["_result_path"] = str(path)
        detail["_result_sha256"] = entry["result_sha256"]
        detail["_diagnostics_path"] = str(diagnostics_path)
        detail["_diagnostics_sha256"] = entry["diagnostics_sha256"]
        output[cell] = detail
    if set(output) != expected_cells:
        raise ValueError(f"{stage} population changed")
    return output


def _details(results_path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(results_path)
    stage1 = _entry_details(campaign, "isotropic_localization")
    stage2 = _entry_details(campaign, "selected_law_comparison")
    output = []
    for cell in sorted(stage1):
        localization = stage1[cell]
        comparison = stage2[cell]
        if (
            localization["provenance"] != comparison["provenance"]
            or set(localization.get("natural_seed_gates", {})) != set(DOSE_KEYS)
            or localization["natural_seed_gates"]["0.625"] is not True
            or localization["gates"].get("zero_dose_replay") is not True
            or any(
                localization["regimes"][regime]["doses"]["0.000"][
                    "zero_replay_maximum_absolute_posterior_error"
                ]
                != 0.0
                for regime in REGIMES
            )
            or comparison.get("selected_multiplier") != 0.625
            or comparison.get("selected_noise_sigma") != 0.03125
            or comparison.get("selected_arrays_sha256")
            != SELECTED_ARRAYS_SHA256
            or comparison.get("selected_noise_law_contract_sha256")
            != SELECTED_CONTRACT_SHA256
            or comparison.get("law_seed_gates")
            != comparison.get("natural_seed_gates")
            or any(comparison.get("control_seed_gates", {}).values())
            or comparison["gates"].get("source_clean_replay") is not True
            or comparison["gates"].get("cut_replay") is not True
            or comparison["gates"].get("analytic_feature_invariance") is not True
        ):
            raise ValueError(f"invalid paired dose result {cell}")
        for regime in REGIMES:
            for law in LAWS:
                law_detail = comparison["regimes"][regime]["laws"][law]
                if (
                    law_detail.get("maximum_replay_error") != 0.0
                    or any(
                        law_detail["cuts"][cut][variant]["task_gate"]
                        is not expected_gate
                        for cut in CUTS
                        for variant, expected_gate in (
                            ("correct_action", True),
                            ("correct_twirl", True),
                            ("orthogonal_action", False),
                            ("orthogonal_twirl", False),
                        )
                    )
                    or (
                        cell[0] == "analytic_calibrated"
                        and law_detail[
                            "correct_action_feature_maximum_absolute_difference"
                        ]
                        > 1e-6
                    )
                ):
                    raise ValueError(
                        f"invalid selected-dose mechanics {cell}/{regime}/{law}"
                    )
        output.append(
            {
                "condition": cell[0],
                "seed": cell[1],
                "stage1": localization,
                "stage2": comparison,
            }
        )
    return output


def _maximum(values: Iterable[float]) -> float:
    return max(float(value) for value in values)


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, float]:
    localization = detail["stage1"]
    comparison = detail["stage2"]
    metrics: dict[str, float] = {
        "validity": 1.0,
        "selected_multiplier": 0.625,
        "selected_noise_sigma": 0.03125,
        "zero_dose_replay_pass": 1.0,
        "selected_isotropic_natural_pass": float(
            localization["natural_seed_gates"]["0.625"]
        ),
        "next_isotropic_natural_pass": float(
            localization["natural_seed_gates"]["0.750"]
        ),
    }
    for law in LAWS:
        metrics[f"{law}_joint_seed_pass"] = float(
            comparison["law_seed_gates"][law]
        )
        metrics[f"{law}_natural_seed_pass"] = float(
            comparison["natural_seed_gates"][law]
        )
        metrics[f"{law}_control_seed_pass"] = float(
            comparison["control_seed_gates"][law]
        )
        for regime in REGIMES:
            record = comparison["regimes"][regime]["laws"][law]
            metrics[f"{regime}_{law}_natural_accuracy_loss"] = float(
                record["natural_utility"]["accuracy_loss"]
            )
            metrics[f"{regime}_{law}_natural_circular_increase"] = float(
                record["natural_utility"]["circular_error_increase"]
            )
            metrics[f"{regime}_{law}_natural_cross_entropy_increase"] = float(
                record["natural_utility"]["cross_entropy_increase"]
            )
        metrics[f"{law}_maximum_action_accuracy_loss"] = _maximum(
            comparison["regimes"][regime]["laws"][law]["cuts"][cut][
                "correct_action"
            ]["task_sufficiency"]["accuracy_loss"]
            for regime in REGIMES
            for cut in CUTS
        )
        metrics[f"{law}_maximum_twirl_accuracy_loss"] = _maximum(
            comparison["regimes"][regime]["laws"][law]["cuts"][cut][
                "correct_twirl"
            ]["task_sufficiency"]["accuracy_loss"]
            for regime in REGIMES
            for cut in CUTS
        )
        metrics[f"{law}_maximum_action_posterior_js"] = _maximum(
            comparison["regimes"][regime]["laws"][law]["cuts"][cut][
                "correct_action"
            ]["posterior_js_from_noisy_identity"]
            for regime in REGIMES
            for cut in CUTS
        )
        metrics[f"{law}_maximum_twirl_posterior_js"] = _maximum(
            comparison["regimes"][regime]["laws"][law]["cuts"][cut][
                "correct_twirl"
            ]["posterior_js_from_noisy_identity"]
            for regime in REGIMES
            for cut in CUTS
        )
    return metrics


def build_noise_law_dose_localization_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-10_tinyllm-noise-law-dose-localization.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    summaries = [
        {
            "experiment_id": detail["stage2"]["experiment_id"],
            "condition": detail["condition"],
            "seed": detail["seed"],
            "validity": True,
            "metrics": _detail_summary(detail),
        }
        for detail in details
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM sensor-noise law closure inside a valid dose window",
            "category": "mechanistic_interpretability",
            "description": (
                "A two-stage frozen intervention first localizes a common "
                "isotropic utility-valid dose, then compares isotropic, "
                "lab-anisotropic, and lab-biased error exactly once at that dose."
            ),
            "question": (
                "Do both calibrated structured systems preserve natural utility "
                "and observed C2 closure under every matched-energy additive law "
                "inside a common isotropic utility window?"
            ),
            "prediction": (
                "All six arm/law populations pass natural utility and correct "
                "action/twirl gates in at least four of five seeds at the largest "
                "common prefix-valid isotropic dose."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_lab_biased_law_failure_at_selected_dose"
            ),
            "evidence_count": 10,
            "direct_experiment_count": 10,
            "direct_quality_experiment_count": 10,
            "integrity_valid_experiment_count": 10,
            "fixed_checkpoint_seed_count": 5,
            "stored_system_count": 10,
            "evidence_role": EVIDENCE_ROLE,
            "power_profile": (
                "two_arm_five_seed_two_stage_corrective_frozen_law_comparison"
            ),
            "tested_scope": (
                "ten retained d8/N3 structured systems, eight nested multipliers "
                "of one frozen error-draw pair, two shifts, and one all-law "
                "comparison at selected sigma=0.03125"
            ),
            "subclaims": {
                "common_isotropic_utility_window": (
                    "supported_selected_sigma_0.03125_five_of_five_both_arms"
                ),
                "zero_mean_anisotropic_utility": (
                    "supported_analytic_four_of_five_learned_five_of_five"
                ),
                "lab_biased_utility": (
                    "rejected_analytic_one_of_five_learned_three_of_five"
                ),
                "conditional_correct_action_sufficiency": (
                    "supported_all_thirty_system_law_cells"
                ),
                "conditional_correct_twirl_sufficiency": (
                    "supported_all_thirty_system_law_cells"
                ),
                "orthogonal_action_specificity": (
                    "supported_zero_of_thirty_system_law_cells"
                ),
                "measurement_robustness_equals_group_closure": "rejected",
                "same_scope_retraining": "not_licensed",
            },
            "tags": [
                "tinyllm",
                "sensor-noise",
                "dose-localization",
                "biased-noise",
                "anisotropic-noise",
                "observed-group-action",
                "reynolds-twirl",
                "frozen-checkpoint",
                "preregistered-corrective",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "A common isotropic utility-valid dose was selected without "
                "asymmetric outcomes, but the lab-biased law passed only one of "
                "five analytic and three of five learned systems. Zero-mean "
                "anisotropy and all conditional action/twirl gates passed."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "high_confidence_valid_corrective_negative_with_bias_specific_failure"
            ),
            "num_direct_experiments": 10,
            "num_direct_quality_experiments": 10,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "stage1": campaign["stage1"],
                "stage2": campaign["stage2"],
                "system_summaries": summaries,
            },
            "key_insights": [
                "The largest common prefix-valid isotropic dose is sigma=0.03125; the next registered dose fails the analytic population gate.",
                "Zero-mean anisotropy passes both arms, so reflection asymmetry alone is not the failing property.",
                "A persistent lab-frame mean fails natural utility in both arms, driven by composition exact-bin accuracy.",
                "Every correct action and twirl remains task-sufficient relative to noisy identity, while every target-changing control fails.",
                "Measurement robustness and functional group closure remain empirically distinct."
            ],
            "suggested_hypotheses": [
                "The deterministic mean component alone explains the biased-law composition failures.",
                "Reversing the lab-frame mean moves failures across phase bins, identifying a directional sensor/readout calibration defect.",
                "If mean-only does not fail, a nonlinear interaction between persistent bias and centered noise is responsible."
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"system_summaries": summaries},
        "source_artifacts": [
            str(results_path),
            str(report_path),
            str(PREREGISTRATION_PATH),
            str(SOURCE_CAMPAIGN_PATH),
            str(SOURCE_NOISE_FILE_PATH),
        ],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "selected_arrays_sha256": SELECTED_ARRAYS_SHA256,
            "selected_contract_sha256": SELECTED_CONTRACT_SHA256,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_noise_file_sha256": SOURCE_NOISE_FILE_SHA256,
            "source_noise_content_sha256": SOURCE_NOISE_CONTENT_SHA256,
            "source_dvc_root": SOURCE_DVC_ROOT,
            "source_lakefs_commit": SOURCE_LAKEFS_COMMIT,
        },
    }


def build_noise_law_dose_localization_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    campaign = _campaign(results_path)
    completed = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        comparison = detail["stage2"]
        output.append(
            ExperimentResult(
                experiment_id=comparison["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=_detail_summary(detail),
                primary_metric=0.0,
                model_architecture=[8, 512],
                model_parameters=50_964_992,
                training_time=float(
                    detail["stage1"]["analysis_seconds"]
                    + comparison["analysis_seconds"]
                ),
                model_checkpoint=comparison["provenance"]["model_checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}; seed: {detail['seed']}.",
                    "The common isotropic dose was selected without asymmetric-law outcomes.",
                    "All integrity, correct-action, correct-twirl, and specificity controls passed.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[
                    "The lab-biased law missed the population utility gate despite intact conditional C2 closure."
                ],
                timestamp=completed,
            )
        )
    return output


def store_noise_law_dose_localization_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_noise_law_dose_localization_meta_hypothesis(results_path)
    experiments = build_noise_law_dose_localization_experiment_results(
        record, results_path
    )
    storage: dict[str, Any] = {
        "aggregate_path": str(output_path),
        "experiment_ids": [item.experiment_id for item in experiments],
    }
    if chromadb_path is not None:
        from structure_net.logging.standardized_logging import (
            LoggingConfig,
            StandardizedLogger,
        )

        root = chromadb_path.parent
        logger = StandardizedLogger(
            LoggingConfig(
                project_name="structure_net_meta_hypotheses",
                queue_dir=str(root / "experiment_queue"),
                sent_dir=str(root / "experiment_sent"),
                rejected_dir=str(root / "experiment_rejected"),
                enable_wandb=False,
                auto_upload=False,
                enable_chromadb=True,
                chromadb_path=str(chromadb_path),
            )
        )
        logger.log_hypothesis(record["hypothesis"])
        storage["result_hashes"] = [
            logger.log_experiment_result(item) for item in experiments
        ]
        storage["chromadb_path"] = str(chromadb_path)
        hypothesis = logger.hypotheses_collection.get(
            ids=[HYPOTHESIS_ID], include=["metadatas"]
        )
        results = logger.experiments_collection.get(
            ids=storage["result_hashes"], include=["metadatas"]
        )
        if (
            hypothesis.get("ids") != [HYPOTHESIS_ID]
            or len(results.get("ids", [])) != len(experiments)
            or {item.get("hypothesis_id") for item in results.get("metadatas", [])}
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("noise-law dose-localization ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": len(results["ids"]),
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


__all__ = [
    "build_noise_law_dose_localization_experiment_results",
    "build_noise_law_dose_localization_meta_hypothesis",
    "store_noise_law_dose_localization_meta_hypothesis",
]
