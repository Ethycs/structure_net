"""Conservative evidence for exact residual correction of a complete Llama."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LosslessResidualEvidence:
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    classification: str
    matrices: int
    final_examples: int
    all_weights_bitwise: bool
    all_outputs_bitwise: bool

    def __post_init__(self):
        if self.matrices != 197 or self.final_examples not in (0, 10):
            raise ValueError("requires all 28 blocks, embedding/head, and the fixed final split")
        if self.classification not in {"lossless_preserved_no_efficiency_gain", "lossless_efficiency_supported", "lossless_accuracy_assay_invalid", "lossless_not_preserved"}:
            raise ValueError("unknown classification")
        if self.classification.startswith("lossless_preserved") or self.classification == "lossless_efficiency_supported":
            if not self.all_weights_bitwise or not self.all_outputs_bitwise or self.final_examples != 10:
                raise ValueError("lossless quality claims require every exactness gate and final evaluation")
        if any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in (self.campaign_sha256,self.report_sha256)):
            raise ValueError("SHA-256 hashes required")

    def record(self):
        return {"id":"llama32-3b-lossless-residual-cotenancy-v1", "name":"Llama lossless tensorization with exact residual correction",
                "description":"Complete 28-layer Llama: bitwise weight reconstruction, exact residual-network output correction and paired quality/timing. Dense correction storage and work are explicitly included; no smaller exact correction network is assumed.",
                "question":"Can exact residual correction preserve Llama behavior while improving physical-model co-tenancy efficiency?",
                "category":"architecture", "tested":True,"confirmed":False,"confirmation_status":self.classification,
                "status":"completed_exploratory_lossless_residual_study", "campaign":self.campaign,"campaign_sha256":self.campaign_sha256,
                "tested_scope":"complete_model_bitwise_residual_controls_and_synthetic_validation_with_conditional_final_test", "direct_experiment_count":7,
                "report":self.report,"report_sha256":self.report_sha256,"matrices":self.matrices,"final_examples":self.final_examples,
                "all_weights_bitwise":self.all_weights_bitwise,"all_outputs_bitwise":self.all_outputs_bitwise,
                "trained_parameters":0,"application_accuracy_tested":False}
