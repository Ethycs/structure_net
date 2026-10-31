"""Measured exploratory Qwen38 task success; no population accuracy claim."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Qwen38Evidence:
    classification: str
    successes: int
    cases: int
    rate: float
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    semantic_supported: int = 0

    def __post_init__(self):
        if self.cases not in (0,12) or not 0<=self.successes<=self.cases:
            raise ValueError('invalid twelve-case result')
        if not 0<=self.semantic_supported<=self.cases: raise ValueError('invalid manual review count')
        if not math.isfinite(self.rate) or self.rate<=0: raise ValueError('positive measured rate required')
        expected=('metaphor_gate_failed' if self.cases==0 else 'higher_joint_task_success' if self.successes>5 else
                  'unchanged_joint_task_success' if self.successes==5 else 'lower_joint_task_success')
        if self.classification not in (expected,'disclosure_failure'): raise ValueError('classification disagrees')
        for h in (self.campaign_sha256,self.report_sha256):
            if len(h)!=64 or any(c not in '0123456789abcdef' for c in h): raise ValueError('SHA256 required')

    def record(self):
        return {'id':'word-games-qwen38-moe-v1','name':'Empero Qwen3.8-35B-A3B distillation for WORD_GAMES',
                'description':'Community Empero Qwen3.8-35B-A3B distillation Q4_K_M with native llama.cpp b11960 on RTX 3060 plus 26 CPU MoE layers, Flash Attention and Q8 KV, thinking disabled, no draft or lookup. Frozen 36-request replay versus Qwen2.5-14B and twelve requested live bridge cases. One deterministic pass, no independent semantic judge or training.',
                'project':'WORD_GAMES','category':'inference','tested':True,'confirmed':False,
                'confirmation_status':self.classification,'joint_successes':self.successes,'cases_completed':self.cases,
                'cases_requested':12,'output_tokens_per_second':self.rate,'replay_requests':36,
                'campaign':self.campaign,'campaign_sha256':self.campaign_sha256,
                'report':self.report,'report_sha256':self.report_sha256,'trained_parameters':0,'cpu_expert_layers':26,'weight_quant':'Q4_K_M',
                'semantic_review_method':'manual_assistant_review' if self.cases else 'not_reached',
                'semantic_supported_summaries':self.semantic_supported}
