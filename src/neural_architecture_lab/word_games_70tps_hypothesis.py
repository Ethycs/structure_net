"""Typed speed evidence with an explicit boundary around semantic quality."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SeventyTpsEvidence:
    best_arm: str
    baseline_tok_s: float
    candidate_tok_s: float
    receiver_correct: int
    receiver_cases: int
    analysis: str
    analysis_sha256: str
    report: str
    report_sha256: str
    valid_selectors: int = 2

    def __post_init__(self):
        if not self.best_arm or any(not math.isfinite(v) or v <= 0 for v in (self.baseline_tok_s, self.candidate_tok_s)):
            raise ValueError('named arm and finite positive throughput required')
        if self.receiver_cases != 24 or not 0 <= self.receiver_correct <= self.receiver_cases:
            raise ValueError('two full receiver passes required')
        if not 0 <= self.valid_selectors <= 2:
            raise ValueError('two selector comparisons required')
        for value in (self.analysis_sha256, self.report_sha256):
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('SHA256 required')

    def record(self):
        reached = self.candidate_tok_s >= 70
        checks_pass = self.receiver_correct == 24 and self.valid_selectors == 2
        classification = 'exploratory_below_70tps_quality_unestablished'
        if reached:
            classification = ('exploratory_70tps_reached_quality_unestablished' if checks_pass
                              else 'exploratory_70tps_reached_existing_checks_failed')
        return {'id': 'word-games-70tps-v1', 'name': 'WORD_GAMES target 70 tokens per second',
            'description': 'Unchanged-weight whole-layer GPU placement, microbatch tuning, prompt checkpoint '
                           'elimination and verified native MTP on a 3060 and two 2060 cards. '
                           'Repeated frozen WORD_GAMES requests; semantic population accuracy unestablished.',
            'project': 'WORD_GAMES', 'category': 'inference', 'tested': True, 'confirmed': False,
            'confirmation_status': classification,
            'best_measured_arm': self.best_arm, 'target_tok_s': 70, 'speed_target_reached': reached,
            'baseline_tok_s': self.baseline_tok_s, 'candidate_tok_s': self.candidate_tok_s,
            'throughput_ratio': self.candidate_tok_s / self.baseline_tok_s,
            'fixed_receiver_correct': self.receiver_correct, 'fixed_receiver_comparisons': self.receiver_cases,
            'fixed_selector_valid': self.valid_selectors, 'fixed_selector_comparisons': 2,
            'existing_checks_pass': checks_pass, 'qualified_70tps_reached': reached and checks_pass,
            'population_accuracy_established': False, 'independent_semantic_judge': False,
            'production_deployed': False, 'trained_parameters': 0, 'weights_changed': False,
            'analysis': self.analysis, 'analysis_sha256': self.analysis_sha256,
            'report': self.report, 'report_sha256': self.report_sha256}
