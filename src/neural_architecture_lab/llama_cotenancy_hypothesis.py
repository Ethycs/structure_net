"""Conservative record for the fixed complete-model tensor packing assay."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LlamaCotenancyEvidence:
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    classification: str
    tested_configurations: int
    promoted_configurations: int

    def __post_init__(self):
        if self.tested_configurations != 4 or not 0 <= self.promoted_configurations <= 4:
            raise ValueError("all four fixed tensorizations are required")
        if self.classification not in {"quality_assay_invalid", "tensorization_quality_failed", "packing_execution_invalid",
                                       "quality_retained_no_practical_win", "practical_cotenancy_supported"}:
            raise ValueError("unknown completed classification")
        if self.classification == "practical_cotenancy_supported" and not self.promoted_configurations:
            raise ValueError("support requires a promoted candidate")
        for digest in (self.campaign_sha256, self.report_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("evidence requires SHA-256 digests")

    def record(self):
        return {"id": "llama32-3b-tensorized-cotenancy-v1", "name": "Llama 3.2 3B complete tensorized model co-tenancy",
                "description": "Complete independent physical replicas with model-major SVD or four-core TT factors. Packing parity, compression quality, resident memory and complete-model latency are assessed separately against the deployed Q4_K_M artifact.",
                "question": "Can post-hoc tensorization and dense batched model-major contractions improve co-tenancy while preserving exact-source summary quality?",
                "category": "architecture", "tested": True, "confirmed": False,
                "confirmation_status": self.classification, "status": "completed_exploratory_feasibility_study",
                "tested_configurations": self.tested_configurations, "promoted_configurations": self.promoted_configurations,
                "campaign": self.campaign, "campaign_sha256": self.campaign_sha256,
                "report": self.report, "report_sha256": self.report_sha256,
                "trained_parameters": 0, "application_accuracy_tested": False,
                "scope": "one_checkpoint_one_gpu_synthetic_extractive_summaries_posthoc_compression"}
