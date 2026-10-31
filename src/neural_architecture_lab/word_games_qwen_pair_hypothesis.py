"""Conservative evidence record for the Qwen WORD_GAMES model switch."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class QwenPairEvidence:
    classification: str
    speed_ratio: float
    baseline_successes: int
    draft_successes: int
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    draft_cases: int = 12

    def __post_init__(self):
        if self.classification not in ('draft_metaphor_gate_failed','disclosure_failure','draft_quality_regressed','draft_pilot_slower','observed_pilot_gain_no_suite_regression'):
            raise ValueError('unknown classification')
        if not math.isfinite(self.speed_ratio) or self.speed_ratio <= 0:
            raise ValueError('positive measured pilot ratio required')
        if any(not 0 <= v <= 12 for v in (self.baseline_successes,self.draft_successes)):
            raise ValueError('twelve-case denominator required')
        if (self.classification == 'draft_metaphor_gate_failed') != (self.draft_cases == 0):
            raise ValueError('failed selector is not a completed twelve-case evaluation')
        if self.draft_cases not in (0,12) or self.draft_successes > self.draft_cases:
            raise ValueError('invalid completed case count')
        if self.classification == 'observed_pilot_gain_no_suite_regression' and (
                self.speed_ratio <= 1 or self.draft_successes < self.baseline_successes):
            raise ValueError('gain classification contradicts measurements')
        if self.classification == 'draft_pilot_slower' and self.speed_ratio > 1:
            raise ValueError('slower classification contradicts measurements')
        for h in (self.campaign_sha256,self.report_sha256):
            if len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
                raise ValueError('SHA256 required')

    def record(self):
        return {'id': 'word-games-qwen25-pair-v1', 'name': 'WORD_GAMES Qwen2.5 14B target / 3B draft',
                'description': 'Official Qwen2.5-Instruct Q4_K_M GGUF pair; shared token IDs, native draft-simple K=3 on RTX 3060 plus RTX 2060 SUPER. Three paired pilot requests and twelve requested live structural bridge cases per arm, subject to the metaphor selection gate. Authorized-only receiver view; no independent semantic judge or training.',
                'project': 'WORD_GAMES', 'category': 'inference', 'tested': True, 'confirmed': False,
                'confirmation_status': self.classification, 'pilot_speed_ratio': self.speed_ratio,
                'baseline_joint_successes': self.baseline_successes, 'draft_joint_successes': self.draft_successes,
                'quality_cases_requested_per_arm': 12, 'baseline_cases_completed':12,
                'draft_cases_completed':self.draft_cases, 'paired_pilot_requests': 3,
                'campaign': self.campaign, 'campaign_sha256': self.campaign_sha256,
                'report': self.report, 'report_sha256': self.report_sha256, 'trained_parameters': 0}
