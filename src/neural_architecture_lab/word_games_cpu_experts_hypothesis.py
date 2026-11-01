"""Conservative fixed-suite evidence for removing CPU expert computation."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CpuExpertEvidence:
    best_arm: str
    baseline_tok_s: float
    candidate_tok_s: float
    cpu_per_token_reduction: float
    receiver_correct: int
    receiver_cases: int
    analysis: str
    analysis_sha256: str
    report: str
    report_sha256: str

    def __post_init__(self):
        if self.best_arm not in ('packed38', 'gpu40', 'gpu40_cpu1', 'q2_3060'):
            raise ValueError('unknown arm')
        if any(not math.isfinite(v) or v <= 0 for v in (self.baseline_tok_s, self.candidate_tok_s)):
            raise ValueError('finite positive rates required')
        if not math.isfinite(self.cpu_per_token_reduction) or self.cpu_per_token_reduction > 1:
            raise ValueError('invalid CPU reduction')
        if not 0 <= self.receiver_correct <= self.receiver_cases or self.receiver_cases != 24:
            raise ValueError('two full fixed-suite receiver passes required')
        for value in (self.analysis_sha256, self.report_sha256):
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('SHA256 required')

    def record(self):
        return {
            'id': 'word-games-cpu-expert-elimination-v1', 'name': 'WORD_GAMES CPU expert elimination',
            'description': 'Move the final two CPU expert banks onto GPUs; compare unchanged weights, '
                           'one CPU compute thread, and selective Q2_K compression onto RTX 3060.',
            'project': 'WORD_GAMES', 'category': 'inference', 'tested': True, 'confirmed': False,
            'confirmation_status': 'exploratory_speed_and_cpu_cost_measured_quality_unestablished',
            'best_measured_arm': self.best_arm, 'baseline_tok_s': self.baseline_tok_s,
            'candidate_tok_s': self.candidate_tok_s,
            'throughput_ratio': self.candidate_tok_s / self.baseline_tok_s,
            'cpu_per_token_reduction': self.cpu_per_token_reduction,
            'fixed_receiver_correct': self.receiver_correct, 'fixed_receiver_comparisons': self.receiver_cases,
            'population_accuracy_established': False, 'independent_semantic_judge': False,
            'production_deployed': False, 'trained_parameters': 0,
            'analysis': self.analysis, 'analysis_sha256': self.analysis_sha256,
            'report': self.report, 'report_sha256': self.report_sha256,
        }
