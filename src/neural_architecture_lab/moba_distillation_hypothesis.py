"""Conservative typed registration for checkpoint-conditional MoBA conversion."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .core import Hypothesis, HypothesisCategory

HYPOTHESIS_ID = "tinyllm-moba-distillation-v1"


def run_registered(config):
    from experiments.structure_net.tinyllm_moba_distillation import StudyConfig, run_campaign
    return run_campaign(StudyConfig(), Path(config["output"]))


@dataclass(frozen=True)
class MoBADistillationEvidence:
    classification: str
    completed_cells: int
    joint_passes: int
    campaign_path: str
    campaign_sha256: str
    report_path: str
    report_sha256: str
    direct_experiment_ids: tuple[str, ...]

    def __post_init__(self):
        allowed = {"routing_distillation_supported", "behavioral_gain_without_full_joint_gate",
                   "distillation_not_confirmed", "no_material_conversion_gap", "invalid_full_budget_conversion"}
        if self.classification not in allowed:
            raise ValueError("only completed preregistered classifications can be registered")
        if self.completed_cells != len(self.direct_experiment_ids) or len(set(self.direct_experiment_ids)) != self.completed_cells:
            raise ValueError("direct evidence count or identity mismatch")
        if not 0 <= self.joint_passes <= 5:
            raise ValueError("invalid seed gate count")
        if self.classification == "routing_distillation_supported" and (self.joint_passes < 4 or self.completed_cells != 40):
            raise ValueError("confirmation requires the full joint gate and complete campaign")

    @property
    def confirmed(self):
        return self.classification == "routing_distillation_supported"


def build_hypothesis(evidence: MoBADistillationEvidence) -> Hypothesis:
    return Hypothesis(
        id=HYPOTHESIS_ID, name="TinyLLM teacher-distilled MoBA conversion",
        description="Matched small-budget conversion of one pretrained native-256 TinyLLM to causal block routing.",
        category=HypothesisCategory.ARCHITECTURE,
        question="Does routing and behavioral distillation preserve a frozen teacher better than LM-only adaptation?",
        prediction="Mass-based combined distillation passes teacher KL, routing recall and NLL jointly in at least four of five adaptation streams.",
        test_function=run_registered, parameter_space={"adaptation_seed": [7, 17, 29, 41, 53]},
        control_parameters={"classification": evidence.classification, "confirmed": evidence.confirmed,
                            "campaign": evidence.campaign_path, "campaign_sha256": evidence.campaign_sha256,
                            "report": evidence.report_path, "report_sha256": evidence.report_sha256,
                            "pretrained_checkpoints": 1, "adaptation_updates": 128,
                            "adaptation_tokens_per_arm": 32768,
                            "direct_experiment_ids": list(evidence.direct_experiment_ids),
                            "seed_semantics": "paired adaptation minibatch schedules, not pretrained initialization"},
        success_metrics={"joint_seed_passes": 4.0}, created_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
        tested=True, tags=["TinyLLM", "MoBA", "distillation", "completed", evidence.classification],
        references=[evidence.campaign_path, evidence.report_path, "https://github.com/MoonshotAI/MoBA"],
    )


def store_evidence(evidence: MoBADistillationEvidence, logger, registry_path: Path):
    """Persist through the canonical logger, then verify exact metadata and JSON."""
    from experiments.structure_net.tinyllm_moba_distillation import sha256, write_json
    import json
    if sha256(evidence.campaign_path) != evidence.campaign_sha256 or sha256(evidence.report_path) != evidence.report_sha256:
        raise RuntimeError("evidence/report changed before registration")
    hypothesis = build_hypothesis(evidence)
    record = {"id": hypothesis.id, "name": hypothesis.name, "description": hypothesis.description,
              "category": hypothesis.category.value, "question": hypothesis.question,
              "prediction": hypothesis.prediction, "created_at": hypothesis.created_at.isoformat(),
              "tested": True, "confirmed": evidence.confirmed, "status": "completed",
              "confirmation_status": evidence.classification, "classification": evidence.classification,
              "tested_scope": "one_pretrained_checkpoint_five_adaptation_streams",
              "evidence_count": evidence.completed_cells, "direct_experiment_count": evidence.completed_cells,
              **hypothesis.control_parameters}
    expected = {"tested": True, "confirmed": evidence.confirmed, "confirmation_status": evidence.classification,
                "evidence_count": evidence.completed_cells, "direct_experiment_count": evidence.completed_cells}
    registry = {"schema": "nal.completed-moba-distillation-registry.v1", "status": "completed",
                "hypotheses": [record], "storage": {"readback": {"verified": True, "hypothesis_id": HYPOTHESIS_ID, **expected}}}
    if registry_path.exists() and json.loads(registry_path.read_text()) != registry:
        raise RuntimeError("preserve existing different registry; use a dated amendment")
    logger.log_hypothesis(record)
    if logger.hypotheses_collection is None:
        raise RuntimeError("searchable hypothesis collection unavailable")
    readback = logger.hypotheses_collection.get(ids=[HYPOTHESIS_ID], include=["metadatas"])
    metadata = readback.get("metadatas", [])
    if readback.get("ids") != [HYPOTHESIS_ID] or not metadata or any(metadata[0].get(k) != v for k, v in expected.items()):
        raise RuntimeError("exact hypothesis metadata read-back failed")
    if not registry_path.exists():
        write_json(registry_path, registry)
    if json.loads(registry_path.read_text()) != registry:
        raise RuntimeError("JSON registry read-back failed")
    return registry
