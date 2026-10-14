"""Conservative evidence contract for the paired continuation study."""
from dataclasses import dataclass

HYPOTHESIS_ID = "llama32-3b-noise-tensor-continuation-v1"


def heldout_gate(candidate, dense, q4, total=20):
    if total != 20 or any(type(n) is not int or not 0 <= n <= total for n in (candidate, dense, q4)):
        raise ValueError("complete final-test counts required")
    return candidate >= 19 and candidate >= dense and candidate >= q4


@dataclass(frozen=True)
class ContinuationEvidence:
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    clean_passed: int
    noise_passed: int
    direct_count: int
    heldout_tested: bool = False
    heldout_quality_gate: bool = False

    def __post_init__(self):
        for value in (self.campaign_sha256, self.report_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("SHA-256 required")
        if not 0 <= self.clean_passed <= 8 or not 0 <= self.noise_passed <= 8 or self.direct_count < 56:
            raise ValueError("complete paired experiment counts required")
        if self.heldout_tested and max(self.clean_passed, self.noise_passed) != 8:
            raise ValueError("heldout cannot bypass development gate")
        if self.heldout_quality_gate and not self.heldout_tested:
            raise ValueError("unmeasured final quality cannot pass")

    def record(self):
        best = max(self.clean_passed, self.noise_passed)
        if best < 8:
            classification = "continued_tensor_student_development_quality_failed"
        elif not self.heldout_tested:
            classification = "development_recovered_final_quality_unmeasured"
        elif not self.heldout_quality_gate:
            classification = "continued_tensor_student_heldout_quality_failed"
        else:
            classification = "proxy_quality_recovered_deployment_speed_unproven"
        return {"id": HYPOTHESIS_ID, "name": "Noisy tensor continuation with global Llama recovery",
                "description": "Paired clean and Gaussian-noise weight-only continuation of Llama 3B rank-1024 factors, response-balanced local fitting and assembled block-coordinate output distillation. Measures complete summaries and native Q4 speed; one paired seed, adaptive synthetic development data.",
                "category": "architecture", "tested": True, "confirmed": False,
                "status": "completed_exploratory", "confirmation_status": classification,
                "campaign": self.campaign, "campaign_sha256": self.campaign_sha256,
                "report": self.report, "report_sha256": self.report_sha256,
                "direct_experiment_count": self.direct_count, "clean_development_passed": self.clean_passed,
                "noise_development_passed": self.noise_passed, "development_total": 8,
                "noise_benefit_observed": self.noise_passed > self.clean_passed,
                "heldout_tested": self.heldout_tested, "heldout_quality_gate": self.heldout_quality_gate,
                "application_accuracy_tested": False, "speed_target_met": False,
                "independent_training_seeds": 1, "cross_request_cache_reuse": False}
