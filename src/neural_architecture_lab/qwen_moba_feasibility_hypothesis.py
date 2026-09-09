"""Conservative registration of public-Qwen MoBA feasibility evidence."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from .core import Hypothesis, HypothesisCategory

HYPOTHESIS_ID = "qwen3-8b-prefix6-moba-feasibility-v1"


def run_registered(config):
    from experiments.structure_net.qwen_moba_shared_assessment import run
    return run(Path(config["output"]))


@dataclass(frozen=True)
class QwenMoBAEvidence:
    classification: str
    completed_cells: int
    final_joint_passes: int
    final_cells: int
    campaign_path: str
    campaign_sha256: str
    report_path: str
    report_sha256: str

    def __post_init__(self):
        if self.classification not in {"no_shared_gpu_candidate_passed", "shared_gpu_prefix_feasibility_supported", "shared_gpu_prefix_feasibility_not_confirmed"}:
            raise ValueError("requires a completed exploratory decision")
        if not 0 <= self.final_joint_passes <= self.final_cells <= 10 or self.completed_cells < self.final_cells:
            raise ValueError("invalid evidence counts")
        if self.classification == "shared_gpu_prefix_feasibility_supported" and self.final_joint_passes != 10:
            raise ValueError("shared-GPU support requires all ten held-out joint gates")
        if self.classification == "no_shared_gpu_candidate_passed" and self.final_cells:
            raise ValueError("no selection must leave final evaluation sealed")


def build_hypothesis(evidence):
    return Hypothesis(
        id=HYPOTHESIS_ID, name="Qwen3-8B prefix MoBA readout and runtime feasibility",
        description="Official MoBA on a frozen public Qwen3-8B six-layer prefix; exploratory proxy and shared-GPU measurements.",
        category=HypothesisCategory.ARCHITECTURE,
        question="Can replacing attention preserve sixth-layer readouts while reducing measured prefix runtime?",
        prediction="A selected sparse configuration preserves defined representation/change proxies and reduces runtime on fresh windows.",
        test_function=run_registered, parameter_space={"window_seed": [7, 17, 29, 41, 53]},
        control_parameters={"classification": evidence.classification, "confirmed": False,
                            "uncontended_speed_confirmed": False, "application_accuracy_tested": False,
                            "campaign": evidence.campaign_path, "campaign_sha256": evidence.campaign_sha256,
                            "report": evidence.report_path, "report_sha256": evidence.report_sha256,
                            "completed_cells": evidence.completed_cells, "final_cells": evidence.final_cells,
                            "final_joint_passes": evidence.final_joint_passes,
                            "trained_parameters": 0, "full_transformer_blocks_executed": 5,
                            "attention_only_final_block": 6},
        success_metrics={"held_out_joint_passes": 10.0}, created_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
        tested=True, tags=["Qwen3-8B", "MoBA", "frozen-weights", "exploratory", "shared-GPU", evidence.classification],
        references=[evidence.campaign_path, evidence.report_path, "https://arxiv.org/pdf/2502.13189"],
    )


def store_evidence(evidence, logger, registry_path):
    from experiments.neural_architecture_lab.audit_qwen_moba_feasibility import sha256
    if sha256(evidence.campaign_path) != evidence.campaign_sha256 or sha256(evidence.report_path) != evidence.report_sha256:
        raise ValueError("evidence or report hash changed")
    hypothesis = build_hypothesis(evidence)
    record = {"id": hypothesis.id, "name": hypothesis.name, "description": hypothesis.description,
              "question": hypothesis.question, "prediction": hypothesis.prediction,
              "category": hypothesis.category.value, "created_at": hypothesis.created_at.isoformat(),
              "tested": True, "confirmed": False, "status": "completed_exploratory",
              "confirmation_status": evidence.classification, "evidence_count": evidence.completed_cells,
              "direct_experiment_count": evidence.completed_cells,
              "tested_scope": "one_frozen_public_prefix_generic_text_shared_gpu",
              **hypothesis.control_parameters}
    expected = {"tested": True, "confirmed": False, "confirmation_status": evidence.classification,
                "evidence_count": evidence.completed_cells, "direct_experiment_count": evidence.completed_cells}
    registry = {"schema": "nal.qwen-moba-feasibility-registry.v1", "hypotheses": [record]}
    registry_path = Path(registry_path)
    if registry_path.exists() and json.loads(registry_path.read_text()) != registry:
        raise ValueError("preserve previous different registry; use an explicit amendment")
    logger.log_hypothesis(record)
    collection = logger.hypotheses_collection
    if collection is None:
        raise RuntimeError("searchable registry unavailable")
    found = collection.get(ids=[HYPOTHESIS_ID], include=["metadatas"])
    if found["ids"] != [HYPOTHESIS_ID] or any(found["metadatas"][0].get(k) != v for k, v in expected.items()):
        raise RuntimeError("exact registry readback failed")
    semantic = collection.query(query_texts=["Qwen3-8B MoBA attention prefix frozen weights runtime feasibility"], n_results=10)
    if HYPOTHESIS_ID not in semantic["ids"][0]:
        raise RuntimeError("semantic registry readback failed")
    if not registry_path.exists():
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if json.loads(registry_path.read_text()) != registry:
        raise RuntimeError("JSON registry readback failed")
    return {"exact_readback_verified": True, "semantic_readback_verified": True,
            "registry": str(registry_path), "hypothesis_id": HYPOTHESIS_ID, **expected}
