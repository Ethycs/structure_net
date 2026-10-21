"""Prospective quality-first decisions for the bounded Llama-to-NCM pilot."""
from dataclasses import dataclass
import math

HYPOTHESIS_ID="llama32-3b-ncm-distillation-v1"


def classification(training_labels,codec_passes=None,teacher_passes=None,student_passes=None,
                   latency_ratio=None,tps_ratio=None):
    if type(training_labels) is not int or not 0<=training_labels<=1024:raise ValueError("complete training-label count required")
    if training_labels<.95*1024:return "training_label_capacity_or_control_failed"
    if type(codec_passes) is not int or not 0<=codec_passes<=64:raise ValueError("development codec count required")
    if codec_passes<.95*64:return "codec_fidelity_failed"
    if any(type(x) is not int or not 0<=x<=100 for x in (teacher_passes,student_passes)):
        raise ValueError("complete final source-review counts required")
    if teacher_passes<95:return "teacher_control_failed"
    if student_passes<95 or student_passes<teacher_passes:return "distilled_quality_failed"
    if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in (latency_ratio,tps_ratio)):
        raise ValueError("positive finite full-request speed ratios required")
    return "task_specific_ncm_prototype_meets_speed_and_quality" if min(latency_ratio,tps_ratio)>=2 else "distilled_quality_retained_speed_target_unmet"


@dataclass(frozen=True)
class NcmEvidence:
    classification: str
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    timing_cells: int

    def record(self):
        allowed={"training_label_capacity_or_control_failed","codec_fidelity_failed","teacher_control_failed",
            "distilled_quality_failed","distilled_quality_retained_speed_target_unmet","task_specific_ncm_prototype_meets_speed_and_quality"}
        if self.classification not in allowed or self.timing_cells not in (0,1200):raise ValueError("complete stopped or final pilot required")
        if any(len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in (self.campaign_sha256,self.report_sha256)):
            raise ValueError("SHA256 identities required")
        return {"id":HYPOTHESIS_ID,"name":"Llama-to-NCM short-summary distillation pilot",
            "description":"Sequence distillation of original Llama-3.2-3B-Instruct summaries into the official pretrained Next-Concept Model, with frozen multiscale codebooks, conditional decoder repair, all-level code cross-entropy, source-fidelity gates and isolated matched-input timing.",
            "category":"architecture","tested":True,"confirmed":False,"status":"completed_exploratory",
            "confirmation_status":self.classification,"direct_experiment_count":self.timing_cells,
            "campaign":self.campaign,"campaign_sha256":self.campaign_sha256,"report":self.report,"report_sha256":self.report_sha256,
            "independent_adaptation_seeds":1,"underpowered":True,"application_accuracy_tested":False,
            "full_application_context_tested":False,"source_prefix_gpt2_tokens":512,"output_canvas_gpt2_tokens":48}


def graph_classification(parity,teacher_quality,student_quality,latency_ratio,tps_ratio):
    if type(parity) is not int or not 0<=parity<=300:raise ValueError("complete graph parity count required")
    if parity<300:return "output_parity_failed"
    if any(type(x) is not int or not 0<=x<=100 for x in (teacher_quality,student_quality)):raise ValueError("source quality counts required")
    if teacher_quality<95 or student_quality<95 or student_quality<teacher_quality:return "inherited_model_quality_failed"
    if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in (latency_ratio,tps_ratio)):raise ValueError("finite speed ratios required")
    return "frozen_ncm_graph_prototype_meets_speed_and_quality" if min(latency_ratio,tps_ratio)>=2 else "speed_target_unmet"


@dataclass(frozen=True)
class NcmGraphEvidence:
    classification: str
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    parent_campaign: str
    parent_campaign_sha256: str

    def record(self):
        if self.classification not in {"output_parity_failed","inherited_model_quality_failed","speed_target_unmet","frozen_ncm_graph_prototype_meets_speed_and_quality"}:
            raise ValueError("known complete graph classification required")
        if any(len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in (self.campaign_sha256,self.report_sha256,self.parent_campaign_sha256)):
            raise ValueError("SHA256 identities required")
        return {"id":"llama32-3b-ncm-graph-runtime-v1","name":"Frozen distilled NCM GPU graph runtime",
            "description":"Full nine-level NCM generation and text decoder replayed in one CUDA graph with unchanged weights and native sampling. Exact eager-output parity and common-token full-request speed versus explicitly reused original and factorized Llama controls.",
            "category":"architecture","tested":True,"confirmed":False,"status":"completed_exploratory",
            "confirmation_status":self.classification,"direct_experiment_count":300,"reused_control_timing_cells":900,
            "training_updates":0,"underpowered":True,"campaign":self.campaign,"campaign_sha256":self.campaign_sha256,
            "report":self.report,"report_sha256":self.report_sha256,"parent_campaign":self.parent_campaign,
            "parent_campaign_sha256":self.parent_campaign_sha256,"application_accuracy_tested":False,
            "full_application_context_tested":False,"used_runtime_validation_sources":True}
