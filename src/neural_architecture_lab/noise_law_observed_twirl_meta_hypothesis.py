"""Build and store the TinyLLM observed-twirl noise-law result."""

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
EXPERIMENT_SCHEMA = "nal.tinyllm-noise-law-observed-twirl.v1"
HYPOTHESIS_ID = "tinyllm-noise-law-observed-twirl-v1"
EVIDENCE_ROLE = "preregistered_frozen_sensor_noise_law_intervention"
CLASSIFICATION = "invalid_isotropic_positive_control"
CAMPAIGN_SHA256 = (
    "868ad0ffee546f157e701790c34a83f20bfb3116e78b2f8c5bc34dd7bfe660d7"
)
NOISE_ARRAYS_FILE_SHA256 = (
    "d3771eac8e29f7940df7feaedebe74a5a78fb273cda2e70928c9be9e37ff3ba6"
)
NOISE_ARRAYS_SHA256 = (
    "93df61bc76ed073ea241c9450e7ec3523e7a98b5ac06e58d7e920a5df07d70aa"
)
NOISE_CONTRACT_SHA256 = (
    "29d6a8f7efc966e60d28257fcdb4133084a423ef3ffbf907c663b009856f7ee9"
)
RESULT_MANIFEST_SHA256 = (
    "7246968593214d5a91b9283e856472cf351b2e921d6712402f9fc128bb457d4d"
)
IMPLEMENTATION_SHA256 = (
    "d4a7e172b0cb9ed5da9a4508c812211882075fcb75db540a17ac6912a8330d6a"
)
RUNNER_SHA256 = (
    "7bed49c064e8a2148268d2a4ab3a42ec70847a15d83c7297cff3d9dccc7970d2"
)
PREREGISTRATION_SHA256 = (
    "8ff50bc47cb7c6223dbf234044a3d18fefd91b15c9d67578d674bb33029be26b"
)
SOURCE_OBSERVED_CAMPAIGN_SHA256 = (
    "79c3e27374d8b6f4611552595de5852ace940204bda825e64cf80eff6ab2050d"
)
SOURCE_OBSERVED_IMPLEMENTATION_SHA256 = (
    "c970fe8801524f5248a9314e821b6783127596d05a2f206325ed85deb42f9629"
)
SOURCE_OBSERVED_RESULT_MANIFEST_SHA256 = (
    "b91af38162fbf45e29348fbdf583cb676660d68cf22e5a795b438fd8cd015db3"
)
SOURCE_CLOSURE_CAMPAIGN_SHA256 = (
    "1457fdcce224157c5d8b19c11317b3d3c47cc7d8575097c78e76ba8798431b14"
)
SOURCE_CLOSURE_RESULT_MANIFEST_SHA256 = (
    "baed34a16dca206536b2e9cd221fd9f7556f4c063f85ee857352522e770844f4"
)
SOURCE_OBSERVED_RUNNER_SHA256 = (
    "6468a7af23cd1ae11cdc6cae3cf553252c2cb19352ed89e44c053a1ade60d213"
)
SOURCE_CLOSURE_RUNNER_SHA256 = (
    "0169a1653695ebe7e55b7a3f49b12401f73439f5a784c2ddd5584a4b685761c6"
)
SOURCE_CALIBRATED_RUNNER_SHA256 = (
    "73e152b9b6c7ab51ad08e6a80786c32a71566f9d4fb410ff2cf3df53761faf77"
)
CONDITIONS = ("analytic_calibrated", "learned_calibrated_equivariant")
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
LAWS = ("isotropic", "lab_anisotropic", "lab_biased")
CUTS = ("pre_block", "full")
RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_noise_law_observed_twirl.py"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-noise-law-observed-twirl-preregistration.md"
)
SOURCE_OBSERVED_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_observed_deck_twirl/"
    "20260810_d10_preregistered/campaign_results.json"
)
SOURCE_CLOSURE_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_calibrated_frontend_causal_closure/"
    "20260810_d15_preregistered/campaign_results.json"
)
SOURCE_OBSERVED_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_observed_deck_twirl.py"
)
SOURCE_CLOSURE_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_calibrated_frontend_causal_closure.py"
)
SOURCE_CALIBRATED_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_calibrated_frontend_causal.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=64)
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
        "anisotropic_covariance_defect_floor": 0.1,
        "batch_size": 256,
        "biased_mean_defect_floor": 0.05,
        "calibrated_source_root": (
            "data/experiments/tinyllm_calibrated_frontend_causal/"
            "20260806_d8_preregistered"
        ),
        "circular_error_increase_ceiling": math.pi / 16.0,
        "conditions": list(CONDITIONS),
        "cross_entropy_increase_ceiling": 0.1,
        "device": "cuda:1",
        "laws": list(LAWS),
        "maximum_control_seed_passes": 1,
        "natural_accuracy_loss_ceiling": 0.05,
        "natural_circular_error_increase_ceiling": math.pi / 16.0,
        "natural_cross_entropy_increase_ceiling": 0.1,
        "noise_energy_relative_tolerance": 0.05,
        "noise_sigma": 0.05,
        "replay_tolerance": 2e-6,
        "required_seed_passes": 4,
        "sample_limit": None,
        "seeds": list(SEEDS),
        "source_closure_root": (
            "data/experiments/tinyllm_calibrated_frontend_causal_closure/"
            "20260810_d15_preregistered"
        ),
        "source_observed_root": (
            "data/experiments/tinyllm_observed_deck_twirl/"
            "20260810_d10_preregistered"
        ),
        "source_replay_tolerance": 2e-6,
    }


def _expected_aggregates() -> dict[str, Any]:
    joint = {
        "analytic_calibrated": {
            "isotropic": 0,
            "lab_anisotropic": 0,
            "lab_biased": 1,
        },
        "learned_calibrated_equivariant": {
            "isotropic": 4,
            "lab_anisotropic": 3,
            "lab_biased": 0,
        },
    }
    arms = {}
    for condition in CONDITIONS:
        arms[condition] = {
            "laws": {
                law: {
                    "joint_pass_count": joint[condition][law],
                    "natural_utility_pass_count": joint[condition][law],
                    "control_pass_count": 0,
                    "action_pass_counts": {cut: 5 for cut in CUTS},
                    "twirl_pass_counts": {cut: 5 for cut in CUTS},
                }
                for law in LAWS
            }
        }
    return {
        "classification": CLASSIFICATION,
        "primary_hypothesis_pass": False,
        "valid": False,
        "integrity_valid": True,
        "isotropic_positive_control": False,
        "analytic_positive_control": False,
        "controls_pass": True,
        "any_natural_failure": True,
        "required_seed_passes": 4,
        "maximum_control_seed_passes": 1,
        "arms": arms,
    }


def _campaign(path: Path) -> dict[str, Any]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    expected_summary = {
        "requested": 10,
        "scheduled": 10,
        "completed": 10,
        "failed": 0,
        "excluded": 0,
        "retries": 0,
        "reused": 0,
        "trained_models": 0,
        "trained_frontends": 0,
        "trained_task_heads": 0,
        "fitted_probes": 0,
        "fitted_observers": 0,
        "fitted_noise_models": 0,
        "fitted_action_parameters": 0,
    }
    expected_source_digests = {
        "runner": RUNNER_SHA256,
        "preregistration": PREREGISTRATION_SHA256,
        "observed_deck": SOURCE_OBSERVED_RUNNER_SHA256,
        "causal_closure": SOURCE_CLOSURE_RUNNER_SHA256,
        "calibrated_frontend": SOURCE_CALIBRATED_RUNNER_SHA256,
    }
    provenance = campaign.get("provenance", {})
    artifacts = campaign.get("artifacts", {})
    noise_path = Path(artifacts.get("noise_law_arrays", ""))
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or campaign.get("schema_version") != EXPERIMENT_SCHEMA
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != EVIDENCE_ROLE
        or campaign.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or campaign.get("source_digests") != expected_source_digests
        or campaign.get("configuration") != _expected_configuration()
        or campaign.get("summary") != expected_summary
        or campaign.get("aggregates") != _expected_aggregates()
        or campaign.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or campaign.get("noise_arrays_sha256") != NOISE_ARRAYS_SHA256
        or campaign.get("noise_law_contract_sha256")
        != NOISE_CONTRACT_SHA256
        or campaign.get("noise_law_contract", {}).get("pass") is not True
        or artifacts.get("noise_law_arrays_file_sha256")
        != NOISE_ARRAYS_FILE_SHA256
        or not noise_path.is_file()
        or _sha256(noise_path) != NOISE_ARRAYS_FILE_SHA256
        or provenance.get("source_observed_campaign_sha256")
        != SOURCE_OBSERVED_CAMPAIGN_SHA256
        or provenance.get("source_observed_implementation_sha256")
        != SOURCE_OBSERVED_IMPLEMENTATION_SHA256
        or provenance.get("source_observed_result_manifest_sha256")
        != SOURCE_OBSERVED_RESULT_MANIFEST_SHA256
        or provenance.get("source_closure_campaign_sha256")
        != SOURCE_CLOSURE_CAMPAIGN_SHA256
        or provenance.get("source_closure_result_manifest_sha256")
        != SOURCE_CLOSURE_RESULT_MANIFEST_SHA256
        or provenance.get("preregistration_sha256")
        != PREREGISTRATION_SHA256
        or len(campaign.get("results", [])) != 10
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid noise-law observed-twirl campaign {path}")
    contract = campaign["noise_law_contract"]["regimes"]
    for regime in REGIMES:
        if (
            contract[regime]["isotropic"][
                "maximum_normalized_covariance_reflection_defect"
            ]
            > 1e-12
            or contract[regime]["lab_anisotropic"][
                "median_normalized_covariance_reflection_defect"
            ]
            < 0.10
            or contract[regime]["lab_biased"][
                "median_normalized_mean_reflection_defect"
            ]
            < 0.05
            or max(
                contract[regime][law]["energy_relative_error"] for law in LAWS
            )
            > 0.05
        ):
            raise ValueError(f"invalid noise-law contract in {path}")
    for source, expected in (
        (RUNNER_PATH, RUNNER_SHA256),
        (PREREGISTRATION_PATH, PREREGISTRATION_SHA256),
        (SOURCE_OBSERVED_CAMPAIGN_PATH, SOURCE_OBSERVED_CAMPAIGN_SHA256),
        (SOURCE_CLOSURE_CAMPAIGN_PATH, SOURCE_CLOSURE_CAMPAIGN_SHA256),
        (SOURCE_OBSERVED_RUNNER_PATH, SOURCE_OBSERVED_RUNNER_SHA256),
        (SOURCE_CLOSURE_RUNNER_PATH, SOURCE_CLOSURE_RUNNER_SHA256),
        (SOURCE_CALIBRATED_RUNNER_PATH, SOURCE_CALIBRATED_RUNNER_SHA256),
    ):
        _require_source(source, expected)
    return campaign


def _details(results_path: Path) -> list[dict[str, Any]]:
    campaign = _campaign(results_path)
    output = []
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    observed_cells = set()
    for entry in campaign["results"]:
        path = Path(entry["path"])
        diagnostics = Path(entry["diagnostics_path"])
        detail = json.loads(path.read_text(encoding="utf-8"))
        cell = (detail.get("condition"), int(detail.get("seed", -1)))
        observed_cells.add(cell)
        provenance = detail.get("provenance", {})
        if (
            _sha256(path) != entry.get("result_sha256")
            or _sha256(diagnostics) != entry.get("diagnostics_sha256")
            or detail.get("schema_version") != EXPERIMENT_SCHEMA
            or detail.get("hypothesis_id") != HYPOTHESIS_ID
            or detail.get("status") != "completed"
            or detail.get("evidence_role") != EVIDENCE_ROLE
            or detail.get("implementation_sha256") != IMPLEMENTATION_SHA256
            or detail.get("configuration") != _expected_configuration()
            or detail.get("dataset_hashes")
            != {
                "composition": "b025623e23f534ca3670f49f1bafc1c3979d1ca834aec2223f9c3166be8f52a6",
                "extrapolation": "f2f1e8c2c06b38fbecf856f0679e9861d9fab3e1c2a41358e55043f4d309a214",
            }
            or detail.get("noise_arrays_sha256") != NOISE_ARRAYS_SHA256
            or detail.get("noise_law_contract_sha256")
            != NOISE_CONTRACT_SHA256
            or detail.get("gates")
            != {
                "source_clean_replay": True,
                "cut_replay": True,
                "analytic_feature_invariance": True,
                "state_unchanged": True,
                "finite": True,
                "validity": True,
            }
            or set(detail.get("regimes", {})) != set(REGIMES)
            or set(detail.get("law_seed_gates", {})) != set(LAWS)
            or detail.get("law_seed_gates") != detail.get("natural_seed_gates")
            or any(detail.get("control_seed_gates", {}).values())
            or detail.get("artifacts", {}).get("diagnostics_sha256")
            != entry.get("diagnostics_sha256")
            or provenance.get("observed_predecessor_result_sha256") is None
            or provenance.get("observed_predecessor_diagnostics_sha256") is None
            or not _finite(detail)
        ):
            raise ValueError(f"invalid noise-law result {path}")
        for regime in REGIMES:
            regime_detail = detail["regimes"][regime]
            if (
                regime_detail.get("source_clean_replay_maximum_absolute_error")
                != 0.0
                or set(regime_detail.get("laws", {})) != set(LAWS)
            ):
                raise ValueError(f"invalid noise-law regime {regime} in {path}")
            for law in LAWS:
                law_detail = regime_detail["laws"][law]
                if (
                    law_detail.get("maximum_replay_error") != 0.0
                    or set(law_detail.get("cuts", {})) != set(CUTS)
                    or any(
                        law_detail["cuts"][cut][variant]["task_gate"] is not expected_gate
                        for cut in CUTS
                        for variant, expected_gate in (
                            ("correct_action", True),
                            ("correct_twirl", True),
                            ("orthogonal_action", False),
                            ("orthogonal_twirl", False),
                        )
                    )
                    or (
                        detail["condition"] == "analytic_calibrated"
                        and law_detail[
                            "correct_action_feature_maximum_absolute_difference"
                        ]
                        > 1e-6
                    )
                ):
                    raise ValueError(
                        f"invalid noise-law mechanics {regime}/{law} in {path}"
                    )
        for key in (
            "model_checkpoint",
            "frontend_checkpoint",
            "source_result",
            "observed_predecessor_result",
            "observed_predecessor_diagnostics",
        ):
            digest_key = f"{key}_sha256"
            _require_source(Path(provenance[key]), provenance[digest_key])
        detail["_result_path"] = str(path)
        detail["_result_sha256"] = entry["result_sha256"]
        detail["_diagnostics_path"] = str(diagnostics)
        detail["_diagnostics_sha256"] = entry["diagnostics_sha256"]
        output.append(detail)
    if observed_cells != expected:
        raise ValueError("noise-law result population changed")
    return output


def _maximum(values: Iterable[float]) -> float:
    return max(float(value) for value in values)


def _detail_summary(detail: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "validity": float(detail["gates"]["validity"]),
        "campaign_primary_valid": 0.0,
    }
    for law in LAWS:
        metrics[f"{law}_joint_seed_pass"] = float(detail["law_seed_gates"][law])
        metrics[f"{law}_natural_seed_pass"] = float(
            detail["natural_seed_gates"][law]
        )
        metrics[f"{law}_control_seed_pass"] = float(
            detail["control_seed_gates"][law]
        )
        for regime in REGIMES:
            record = detail["regimes"][regime]["laws"][law]
            metrics[f"{regime}_{law}_natural_accuracy_loss"] = float(
                record["natural_utility"]["accuracy_loss"]
            )
        metrics[f"{law}_maximum_action_accuracy_loss"] = _maximum(
            detail["regimes"][regime]["laws"][law]["cuts"][cut][
                "correct_action"
            ]["task_sufficiency"]["accuracy_loss"]
            for regime in REGIMES
            for cut in CUTS
        )
        metrics[f"{law}_maximum_twirl_accuracy_loss"] = _maximum(
            detail["regimes"][regime]["laws"][law]["cuts"][cut][
                "correct_twirl"
            ]["task_sufficiency"]["accuracy_loss"]
            for regime in REGIMES
            for cut in CUTS
        )
        metrics[f"{law}_maximum_action_posterior_js"] = _maximum(
            detail["regimes"][regime]["laws"][law]["cuts"][cut][
                "correct_action"
            ]["posterior_js_from_noisy_identity"]
            for regime in REGIMES
            for cut in CUTS
        )
        metrics[f"{law}_maximum_twirl_posterior_js"] = _maximum(
            detail["regimes"][regime]["laws"][law]["cuts"][cut][
                "correct_twirl"
            ]["posterior_js_from_noisy_identity"]
            for regime in REGIMES
            for cut in CUTS
        )
        metrics[f"{law}_maximum_feature_action_error"] = _maximum(
            detail["regimes"][regime]["laws"][law][
                "correct_action_feature_maximum_absolute_difference"
            ]
            for regime in REGIMES
        )
    return metrics


def build_noise_law_observed_twirl_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/2026-08-10_tinyllm-noise-law-observed-twirl.md"
    ),
) -> dict[str, Any]:
    campaign = _campaign(results_path)
    details = _details(results_path)
    summaries = [
        {
            "experiment_id": detail["experiment_id"],
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
            "name": "TinyLLM observed-twirl robustness to asymmetric noise laws",
            "category": "mechanistic_interpretability",
            "description": (
                "A frozen matched-energy intervention tests observed C2 action "
                "and twirl closure under isotropic, lab-anisotropic, and "
                "lab-biased decoded planar error."
            ),
            "question": (
                "Does calibrated observed quotient closure remain usable when "
                "the sensor-noise law is not reflection-invariant?"
            ),
            "prediction": (
                "Both structured arms preserve natural utility and correct "
                "action/twirl sufficiency in four of five seeds under every law."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "not_confirmed_invalid_isotropic_positive_control"
            ),
            "evidence_count": 10,
            "direct_experiment_count": 10,
            "direct_quality_experiment_count": 0,
            "integrity_valid_experiment_count": 10,
            "fixed_checkpoint_seed_count": 5,
            "stored_system_count": 10,
            "evidence_role": EVIDENCE_ROLE,
            "power_profile": (
                "two_arm_five_seed_three_law_two_shift_frozen_intervention_invalid_primary"
            ),
            "tested_scope": (
                "ten retained d8/N3 structured systems, one added sigma=0.05 "
                "decoded planar dose, three matched-energy laws, composition "
                "and extrapolation, pre-block and full cuts"
            ),
            "subclaims": {
                "asymmetric_noise_law_quotient_robustness": (
                    "not_tested_invalid_isotropic_positive_control"
                ),
                "analytic_isotropic_natural_utility": "failed_zero_of_five",
                "learned_isotropic_natural_utility": "passed_four_of_five",
                "learned_anisotropic_natural_utility": "failed_three_of_five",
                "learned_biased_natural_utility": "failed_zero_of_five",
                "conditional_correct_action_sufficiency": (
                    "supported_all_thirty_system_law_cells"
                ),
                "conditional_correct_twirl_sufficiency": (
                    "supported_all_thirty_system_law_cells"
                ),
                "orthogonal_action_specificity": (
                    "supported_zero_of_thirty_system_law_cells"
                ),
                "analytic_feature_action_invariance": "supported_all_laws_shifts",
                "same_scope_retraining": "not_licensed",
            },
            "tags": [
                "tinyllm",
                "observed-group-action",
                "noise-law",
                "anisotropic-noise",
                "biased-noise",
                "reynolds-twirl",
                "frozen-checkpoint",
                "preregistered-invalid",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "The analytic isotropic natural-utility positive control passes "
                "zero of five seeds, so asymmetric-law effects cannot be isolated "
                "at the registered dose despite universal conditional action and "
                "twirl sufficiency."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "high_confidence_invalid_primary_with_narrow_conditional_mechanics"
            ),
            "num_direct_experiments": 10,
            "num_direct_quality_experiments": 0,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "noise_law_contract": campaign["noise_law_contract"],
                "system_summaries": summaries,
            },
            "key_insights": [
                "The fixed sigma=0.05 dose breaks analytic natural task utility even under isotropic error, invalidating the shape comparison.",
                "Conditional on a noisy identity, every correct action and twirl remains task-sufficient across all systems, laws, cuts, and shifts.",
                "Every matched target-changing orthogonal control fails, so the narrow action result is specific.",
                "The learned temporal encoder is more naturally noise-robust than the endpoint-only analytic canonicalizer, but that was not the registered primary claim.",
                "Measurement robustness and functional group equivariance are empirically distinct properties."
            ],
            "suggested_hypotheses": [
                "A nested lower-dose window can isolate law shape while preserving both isotropic positive controls.",
                "The learned temporal encoder gains measurement robustness by averaging history rather than by stronger deck equivariance.",
                "Exact scalar symmetrization can preserve action closure but cannot by itself repair noisy natural utility."
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"system_summaries": summaries},
        "source_artifacts": [
            str(results_path),
            str(Path(campaign["artifacts"]["noise_law_arrays"])),
            str(report_path),
            str(PREREGISTRATION_PATH),
        ],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "noise_arrays_file_sha256": NOISE_ARRAYS_FILE_SHA256,
            "noise_arrays_sha256": NOISE_ARRAYS_SHA256,
            "noise_contract_sha256": NOISE_CONTRACT_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "source_observed_campaign_sha256": SOURCE_OBSERVED_CAMPAIGN_SHA256,
            "source_observed_result_manifest_sha256": (
                SOURCE_OBSERVED_RESULT_MANIFEST_SHA256
            ),
            "source_closure_campaign_sha256": SOURCE_CLOSURE_CAMPAIGN_SHA256,
            "source_closure_result_manifest_sha256": (
                SOURCE_CLOSURE_RESULT_MANIFEST_SHA256
            ),
        },
    }


def build_noise_law_observed_twirl_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    campaign = _campaign(results_path)
    completed = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    output = []
    for detail in _details(results_path):
        metrics = _detail_summary(detail)
        output.append(
            ExperimentResult(
                experiment_id=detail["experiment_id"],
                hypothesis_id=HYPOTHESIS_ID,
                metrics=metrics,
                primary_metric=0.0,
                model_architecture=[8, 512],
                model_parameters=50_964_992,
                training_time=float(detail["analysis_seconds"]),
                model_checkpoint=detail["provenance"]["model_checkpoint"],
                observations=[
                    f"Condition: {detail['condition']}; seed: {detail['seed']}.",
                    "The system and action remained frozen and every integrity gate passed.",
                    "Correct action and twirl passed all laws, cuts, and shifts conditional on noisy identity.",
                    f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
                ],
                anomalies=[
                    "The campaign-level isotropic analytic natural-utility positive control failed, invalidating the primary law-shape comparison."
                ],
                timestamp=completed,
            )
        )
    return output


def store_noise_law_observed_twirl_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_noise_law_observed_twirl_meta_hypothesis(results_path)
    experiments = build_noise_law_observed_twirl_experiment_results(
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
            raise RuntimeError("noise-law observed-twirl ChromaDB read-back failed")
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
    "build_noise_law_observed_twirl_experiment_results",
    "build_noise_law_observed_twirl_meta_hypothesis",
    "store_noise_law_observed_twirl_meta_hypothesis",
]
