"""Fixed task quality and complete-request speed gates for runtime changes."""
from dataclasses import dataclass
import math

HYPOTHESIS_ID = "llama32-3b-tensor-runtime-v1"


def classification(candidate, reference, native, old_ratio, native_ratio, native_tps_ratio):
    if any(type(n) is not int or not 0 <= n <= 20 for n in (candidate, reference, native)):
        raise ValueError("twenty complete final quality decisions required")
    if any(not math.isfinite(v) or v <= 0 for v in (old_ratio, native_ratio, native_tps_ratio)):
        raise ValueError("positive measured speed ratios required")
    if candidate < 19 or candidate < max(reference, native):
        return "runtime_final_quality_failed"
    if min(native_ratio, native_tps_ratio) >= 2:
        return "proxy_quality_and_double_native_speed_supported"
    if old_ratio >= 2:
        return "prototype_speed_recovered_native_target_unmet"
    return "runtime_speed_target_unmet"


@dataclass(frozen=True)
class RuntimeEvidence:
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    classification: str
    direct_count: int

    def record(self):
        if self.classification not in {"runtime_development_quality_failed", "runtime_final_quality_failed",
                "proxy_quality_and_double_native_speed_supported", "prototype_speed_recovered_native_target_unmet", "runtime_speed_target_unmet"}:
            raise ValueError("known conservative classification required")
        if self.direct_count < 96:
            raise ValueError("complete development evidence required")
        if any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in (self.campaign_sha256, self.report_sha256)):
            raise ValueError("SHA256 required")
        return {"id":HYPOTHESIS_ID, "name":"Frozen noisy tensor student GPU replay and verified prompt lookup",
                "description":"GPU graph replay and within-request prompt lookup for a fixed Llama 3B tensorized summarizer; fresh synthetic source reviews and five technical timing repetitions versus native Q4 lookup. No new training or cross-request evidence cache reuse.",
                "category":"architecture", "tested":True, "confirmed":False, "status":"completed_exploratory",
                "confirmation_status":self.classification, "direct_experiment_count":self.direct_count,
                "campaign":self.campaign, "campaign_sha256":self.campaign_sha256,
                "report":self.report, "report_sha256":self.report_sha256,
                "application_accuracy_tested":False, "independent_training_seeds":1,
                "cross_request_cache_reuse":False}
