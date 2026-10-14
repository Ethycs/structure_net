"""Conservative promotion for measured summarization throughput and task quality."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SummaryThroughputEvidence:
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    speedup: float
    latency_speedup: float
    q4_speedup: float
    quality_passed: int
    quality_total: int
    baseline_quality_passed: int
    source_review_completed: bool
    direct_experiment_count: int

    def __post_init__(self):
        for value in (self.campaign_sha256, self.report_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("SHA-256 required")
        if any(not math.isfinite(v) or v <= 0 for v in (self.speedup, self.latency_speedup, self.q4_speedup)):
            raise ValueError("finite positive ratios required")
        if not 0 <= self.quality_passed <= self.quality_total or self.quality_total < 1:
            raise ValueError("invalid quality counts")
        if not 0 <= self.baseline_quality_passed <= self.quality_total or self.direct_experiment_count < 1:
            raise ValueError("invalid baseline or cell counts")

    def record(self):
        quality = (self.source_review_completed and self.quality_passed / self.quality_total >= .95
                   and self.quality_passed >= self.baseline_quality_passed)
        speed = self.speedup >= 2 and self.latency_speedup > 1
        classification = ("twofold_fp16_speedup_with_proxy_quality_supported" if speed and quality else
                          "speed_target_met_quality_not_established" if speed else "twofold_speed_target_not_met")
        return {"id": "llama32-3b-summary-throughput-v1", "name": "Llama 3B bounded summary throughput",
                "description": "Matched full Llama 3.2 3B fresh-input summary generation on RTX 3060. Quantized weights and target-verified prompt lookup compared with optimized FP16 and existing Q4 controls. Held-out synthetic final-decision quality is source-reviewed; production application quality remains unmeasured.",
                "category": "architecture", "tested": True, "confirmed": False,
                "status": "completed_exploratory", "confirmation_status": classification,
                "campaign": self.campaign, "campaign_sha256": self.campaign_sha256,
                "report": self.report, "report_sha256": self.report_sha256,
                "speedup_over_fp16": self.speedup, "speedup_over_q4": self.q4_speedup,
                "latency_speedup_over_fp16": self.latency_speedup,
                "proxy_quality_passed": self.quality_passed, "proxy_quality_total": self.quality_total,
                "source_review_completed": self.source_review_completed, "quality_gate": quality,
                "direct_experiment_count": self.direct_experiment_count,
                "application_accuracy_tested": False, "bitwise_losslessness_required": False,
                "cross_request_cache_reuse": False, "trained_parameters": 0}
