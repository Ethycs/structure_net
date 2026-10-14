"""Conservative evidence contract for a single-model neighbourhood experiment."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NeighbourhoodEvidence:
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    completed_groups: int
    development_passed: int
    development_total: int
    parameter_ratio: float
    heldout_tested: bool = False
    heldout_quality_gate: bool = False
    speed_target_met: bool = False

    def __post_init__(self):
        for h in (self.campaign_sha256, self.report_sha256):
            if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                raise ValueError("SHA-256 required")
        if self.completed_groups != 14 or self.development_total != 8:
            raise ValueError("complete prespecified training and development evidence required")
        if not 0 <= self.development_passed <= self.development_total:
            raise ValueError("invalid quality count")
        if not math.isfinite(self.parameter_ratio) or not 0 < self.parameter_ratio < 1:
            raise ValueError("actual parameter compression required")
        if self.heldout_tested and self.development_passed != self.development_total:
            raise ValueError("held-out stage cannot pass the development stop rule")
        if self.heldout_quality_gate and not self.heldout_tested:
            raise ValueError("unmeasured quality cannot pass")

    def record(self):
        if self.development_passed < self.development_total:
            classification = "tensorized_student_development_quality_failed"
        elif not self.heldout_tested:
            classification = "local_recovery_supported_final_quality_unmeasured"
        elif not self.heldout_quality_gate:
            classification = "tensorized_student_heldout_quality_failed"
        elif not self.speed_target_met:
            classification = "proxy_quality_retained_twofold_speed_target_failed"
        else:
            classification = "proxy_quality_and_twofold_speed_supported"
        return {"id": "llama32-3b-neighbourhood-tensorization-v1",
                "name": "Llama 3B tensorized neighbourhood distillation",
                "description": "Rank-1024 activation-scaled MLP factors initialized from original Llama 3.2 3B, trained in 14 independent two-layer neighbourhoods and assembled for bounded summarization. Local hidden-state recovery, actual compression, complete-summary quality, and native Q4 deployment performance are distinct endpoints.",
                "category": "architecture", "tested": True, "confirmed": False,
                "status": "completed_exploratory", "confirmation_status": classification,
                "campaign": self.campaign, "campaign_sha256": self.campaign_sha256,
                "report": self.report, "report_sha256": self.report_sha256,
                "completed_neighbourhoods": self.completed_groups,
                "direct_experiment_count": 106,
                "development_quality_passed": self.development_passed,
                "development_quality_total": self.development_total,
                "parameter_ratio": self.parameter_ratio, "heldout_tested": self.heldout_tested,
                "heldout_quality_gate": self.heldout_quality_gate, "speed_target_met": self.speed_target_met,
                "application_accuracy_tested": False, "independent_training_seeds": 1,
                "cross_request_cache_reuse": False, "exact_dense_correction": False}
