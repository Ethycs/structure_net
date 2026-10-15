"""Quality-first speed gates for a fixed packed-weight kernel optimization."""
from dataclasses import dataclass
import math

HYPOTHESIS_ID="llama32-3b-q4-kernel-runtime-v1"
PAYLOAD_BYTES=3_441_776_640


def classification(development,quality=None,old_quality=None,payload=None,latency_ratio=None,tps_ratio=None):
    if set(development)!={"packed","staged","transient"} or any(type(n) is not int or not 0<=n<=8 for n in development.values()):
        raise ValueError("three complete development decisions required")
    if max(development.values())<8:return "development_quality_failed"
    if any(type(n) is not int or not 0<=n<=20 for n in (quality,old_quality)):
        raise ValueError("complete final quality decisions required")
    if payload!=PAYLOAD_BYTES:raise ValueError("unchanged persistent payload required")
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in (latency_ratio,tps_ratio)):
        raise ValueError("positive finite measured speed ratios required")
    if quality<19 or quality<old_quality:return "final_quality_failed"
    return "kernel_speed_improved" if min(latency_ratio,tps_ratio)>=1.05 else "speed_target_unmet"


@dataclass(frozen=True)
class KernelEvidence:
    campaign:str
    campaign_sha256:str
    report:str
    report_sha256:str
    classification:str
    direct_count:int

    def record(self):
        if self.classification not in {"development_quality_failed","final_quality_failed","kernel_speed_improved","speed_target_unmet"} or self.direct_count<72:
            raise ValueError("complete fixed development evidence required")
        if any(len(v)!=64 or any(c not in "0123456789abcdef" for c in v) for v in (self.campaign_sha256,self.report_sha256)):
            raise ValueError("SHA256 required")
        return {"id":HYPOTHESIS_ID,"name":"Frozen Llama FP16-core/Q4-tail kernel optimization",
            "description":"Packed Q4 kernel layouts, fused epilogues, short FP16 tensor partials with FP32 outer accumulation and transient prefill expansion for an unchanged neighborhood-distilled Llama 3B checkpoint. Source-reviewed summaries and paired full-request timing versus old packed and FP16 runtimes.",
            "category":"architecture","tested":True,"confirmed":False,"status":"completed_exploratory",
            "confirmation_status":self.classification,"direct_experiment_count":self.direct_count,
            "campaign":self.campaign,"campaign_sha256":self.campaign_sha256,"report":self.report,"report_sha256":self.report_sha256,
            "training_updates":0,"independent_training_seeds":1,"application_accuracy_tested":False,
            "cross_request_cache_reuse":False,"persistent_dense_tail_cached":False}
