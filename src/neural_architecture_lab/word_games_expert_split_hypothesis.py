"""Conservative tested records for hardware expert partition experiments."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExpertSplitEvidence:
    hypothesis_id: str
    classification: str
    description: str
    baseline_tok_s: float
    candidate_tok_s: float
    full_passes: int
    completed_requests: int
    service_restored: bool
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str

    def __post_init__(self):
        if self.hypothesis_id not in ('word-games-hybrid-experts-v1','word-games-three-gpu-experts-v1'):
            raise ValueError('unknown hypothesis')
        if self.classification not in ('negative_speed_pilot','full_speedup_quality_unestablished',
                                       'full_no_speedup_quality_unestablished','implementation_failed'):
            raise ValueError('unknown classification')
        if any(not math.isfinite(v) or v<=0 for v in (self.baseline_tok_s,self.candidate_tok_s)):
            raise ValueError('positive finite measured rates required')
        if self.classification.startswith('full_') and self.full_passes<2:
            raise ValueError('matched full passes required')
        if self.classification=='full_speedup_quality_unestablished' and self.candidate_tok_s<=self.baseline_tok_s:
            raise ValueError('speed gain absent')
        if self.completed_requests<1 or self.full_passes<0:raise ValueError('invalid counts')
        if self.hypothesis_id.endswith('three-gpu-experts-v1') and not self.service_restored:
            raise ValueError('service restoration required')
        for value in (self.campaign_sha256,self.report_sha256):
            if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('SHA256 required')

    def record(self):
        return {'id':self.hypothesis_id,'name':'WORD_GAMES '+self.hypothesis_id,
                'description':self.description,'project':'WORD_GAMES','category':'inference',
                'tested':True,'confirmed':False,'confirmation_status':self.classification,
                'baseline_tok_s':self.baseline_tok_s,'candidate_tok_s':self.candidate_tok_s,
                'throughput_ratio':self.candidate_tok_s/self.baseline_tok_s,
                'full_passes':self.full_passes,'completed_requests':self.completed_requests,
                'service_restored':self.service_restored,'trained_parameters':0,
                'independent_semantic_judge':False,'accuracy_noninferiority_established':False,
                'campaign':self.campaign,'campaign_sha256':self.campaign_sha256,
                'report':self.report,'report_sha256':self.report_sha256}
