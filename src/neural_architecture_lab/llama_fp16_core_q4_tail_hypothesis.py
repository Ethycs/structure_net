"""Conservative gates for a trained FP16 core and uniform-Q4 residual."""
from dataclasses import dataclass
import math

HYPOTHESIS_ID="llama32-3b-fp16-core-q4-tail-v1"
BASELINE_BYTES=4_135_409_664


def classification(development,quality=None,reference_quality=None,payload=None,latency_ratio=None,tps_ratio=None):
    if set(development)!={64,128} or any(type(n) is not int or not 0<=n<=8 for n in development.values()):
        raise ValueError("two complete eight-request development decisions required")
    if max(development.values())<8:return "development_quality_failed"
    if any(type(n) is not int or not 0<=n<=20 for n in (quality,reference_quality)):
        raise ValueError("twenty final source decisions required")
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in (payload,latency_ratio,tps_ratio)):
        raise ValueError("positive finite measured size and speed required")
    if quality<19 or quality<reference_quality:return "final_quality_failed"
    if payload>BASELINE_BYTES*.9:return "no_useful_compression"
    if min(latency_ratio,tps_ratio)>=1.05:return "smaller_and_faster_proxy_quality_retained"
    return "smaller_speed_unmet"


@dataclass(frozen=True)
class CoreQ4Evidence:
    campaign:str
    campaign_sha256:str
    report:str
    report_sha256:str
    classification:str
    direct_count:int

    def record(self):
        if self.classification not in {"development_quality_failed","final_quality_failed","no_useful_compression",
                "smaller_and_faster_proxy_quality_retained","smaller_speed_unmet"} or self.direct_count<48:
            raise ValueError("complete fixed evidence and known classification required")
        if any(len(v)!=64 or any(c not in "0123456789abcdef" for c in v) for v in (self.campaign_sha256,self.report_sha256)):
            raise ValueError("SHA256 required")
        return {"id":HYPOTHESIS_ID,"name":"Trained Llama FP16 tensor core with Q4 residual",
            "description":"Exploratory mixed precision compression of a noise-trained neighborhood-distilled Llama 3B student: FP16 rank64/128 factor core plus packed signed Q4 residual, CUDA graph verification and current-prompt lookup. Fixed source-reviewed summary quality, full payload size and complete-request timing gates.",
            "category":"architecture","tested":True,"confirmed":False,"status":"completed_exploratory",
            "confirmation_status":self.classification,"direct_experiment_count":self.direct_count,
            "campaign":self.campaign,"campaign_sha256":self.campaign_sha256,"report":self.report,"report_sha256":self.report_sha256,
            "application_accuracy_tested":False,"independent_training_seeds":1,"training_updates":0,
            "random_inference_noise":False,"dense_tail_cached":False,"cross_request_cache_reuse":False}
