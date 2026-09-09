"""Typed systems evidence; no promotion to application-accuracy confirmation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class IdleLatencyEvidence:
    campaign_path: str
    campaign_sha256: str
    report_path: str
    report_sha256: str
    completed_cells: int

    def __post_init__(self):
        if self.completed_cells != 10:
            raise ValueError("requires all ten fixed latency cells")
        if any(len(value) != 64 for value in (self.campaign_sha256, self.report_sha256)):
            raise ValueError("requires content hashes")

    def record(self):
        return {"id": "qwen3-8b-prefix6-moba-idle-latency-v1",
                "name": "Qwen3-8B MoBA idle-GPU latency diagnostic",
                "description": "Fixed-setting idle RTX 3060 measurements isolate attention, MLP and projection costs. Corrects the earlier claim that GPU contention caused the approximately two-second prefix latency.",
                "question": "Does removing head and MLP chunking explain the slow prefix latency?",
                "prediction": "Execution scheduling accounts for only part of the prefix cost; unchanged MLPs and projections remain.",
                "category": "architecture", "tested": True, "confirmed": False,
                "confirmation_status": "idle_gpu_latency_diagnostic", "status": "completed_systems_diagnostic",
                "tested_scope": "one_reused_validation_window_two_lengths_five_execution_variants",
                "evidence_count": self.completed_cells, "direct_experiment_count": self.completed_cells,
                "campaign": self.campaign_path, "campaign_sha256": self.campaign_sha256,
                "report": self.report_path, "report_sha256": self.report_sha256,
                "application_accuracy_tested": False, "trained_parameters": 0}
