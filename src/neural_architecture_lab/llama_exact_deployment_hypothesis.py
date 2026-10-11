"""Keep dispatch speedups distinct from tensor-factor or two-model packing claims."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExactDeploymentEvidence:
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    all_bitwise: bool
    speedups: tuple
    parameter_bytes: int

    def __post_init__(self):
        if len(self.speedups) != 3 or any(not math.isfinite(s) or s <= 0 for s in self.speedups):
            raise ValueError("three finite positive workload speedups required")
        if self.parameter_bytes != 6425499648:
            raise ValueError("unchanged complete original model parameters required")
        for value in (self.campaign_sha256, self.report_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("SHA-256 digests required")

    def record(self):
        passed = self.all_bitwise and any(s >= 1.10 for s in self.speedups)
        classification = "exact_execution_speedup_supported_no_two_for_one" if passed else "exact_execution_speedup_not_supported"
        return {"id": "llama32-3b-lossless-deployment-efficiency-v1",
            "name": "Llama exact CUDA graph deployment efficiency",
            "description": "Complete unchanged 28-layer FP16 model, eager versus CUDA graph replay with owned outputs. Three fixed workloads, ten alternating paired samples and bitwise controls. Dispatch optimization only; no weight-compression or two-independent-model packing claim.",
            "category": "architecture", "tested": True, "confirmed": False,
            "confirmation_status": classification, "status": "completed_exploratory_exact_execution_study",
            "campaign": self.campaign, "campaign_sha256": self.campaign_sha256,
            "report": self.report, "report_sha256": self.report_sha256,
            "direct_experiment_count": 3, "all_bitwise": self.all_bitwise,
            "faster_single_model_supported": passed, "two_for_one_supported": False,
            "weight_compression_used": False, "inference_arithmetic_changed": False,
            "best_speedup": max(self.speedups), "parameter_bytes": self.parameter_bytes,
            "trained_parameters": 0, "application_accuracy_tested": False}
