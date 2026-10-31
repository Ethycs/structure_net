"""Conservative finite-suite evidence for MoE placement tuning."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MoEPlacementEvidence:
    classification: str
    best_condition: str
    speedup: float
    parsed_equal: int
    full_passes: int
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str

    def __post_init__(self):
        if self.classification not in ('speedup_with_replay_parity','no_speedup_with_replay_parity'):
            raise ValueError('unknown classification')
        if not math.isfinite(self.speedup) or self.speedup<=0:raise ValueError('invalid speed ratio')
        if not 0<=self.parsed_equal<=36 or self.full_passes<3:raise ValueError('invalid replay count')
        if self.classification=='speedup_with_replay_parity' and (self.parsed_equal!=36 or self.speedup<1/.95):
            raise ValueError('speed/parity gate not satisfied')
        for h in (self.campaign_sha256,self.report_sha256):
            if len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('SHA256 required')

    def record(self):
        return {'id':'word-games-moe-placement-v1','name':'WORD_GAMES exact-weight MoE placement and CPU tuning',
                'description':'Exploratory Empero 35B Q4_K_M native b11960 hardware placement, CPU thread/affinity and exact expert caching on RTX 3060 plus RTX 2060 SUPER. Fixed 36-request replay and repeated beginning/ending controls. Same weights, Flash Attention and Q8 KV. Parsed output parity is not task accuracy; existing selector and unsupported approval failures remain.',
                'project':'WORD_GAMES','category':'inference','tested':True,'confirmed':False,
                'confirmation_status':self.classification,'best_condition':self.best_condition,
                'conservative_speedup':self.speedup,'parsed_equal_requests':self.parsed_equal,
                'requests_per_pass':36,'full_passes':self.full_passes,'trained_parameters':0,
                'independent_semantic_judge':False,'quality_measure':'finite_frozen_replay_parity',
                'campaign':self.campaign,'campaign_sha256':self.campaign_sha256,
                'report':self.report,'report_sha256':self.report_sha256}
