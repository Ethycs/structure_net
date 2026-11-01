"""Conservative typed record for public instruction sequence-state caching."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SchemaPrefillEvidence:
    baseline_tok_s: float
    cached_tok_s: float
    matching_outputs: int
    existing_checks_pass: bool
    classification: str
    analysis: str
    analysis_sha256: str
    report: str
    report_sha256: str

    def __post_init__(self):
        if not all(math.isfinite(n) and n > 0 for n in (self.baseline_tok_s, self.cached_tok_s)):
            raise ValueError('finite positive throughput required')
        if not 0 <= self.matching_outputs <= 72:
            raise ValueError('72 paired output comparisons required')
        for value in (self.analysis_sha256, self.report_sha256):
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('SHA256 required')

    def record(self):
        return {'id': 'word-games-schema-prefill-v1', 'name': 'WORD_GAMES public schema prefill',
            'description': 'Restore fixed public sender and receiver prefixes, including recurrent '
                'and KV sequence state, before every request. Matched ABBA Triton whole-request '
                'timings, unchanged weights, independent semantic accuracy unestablished.',
            'project': 'WORD_GAMES', 'category': 'inference', 'tested': True, 'confirmed': False,
            'confirmation_status': self.classification, 'baseline_tok_s': self.baseline_tok_s,
            'cached_tok_s': self.cached_tok_s, 'speed_ratio': self.cached_tok_s / self.baseline_tok_s,
            'target_70_reached': self.cached_tok_s >= 70, 'matching_paired_outputs': self.matching_outputs,
            'paired_comparisons': 72, 'existing_checks_pass': self.existing_checks_pass,
            'population_accuracy_established': False, 'weights_changed': False,
            'analysis': self.analysis, 'analysis_sha256': self.analysis_sha256,
            'report': self.report, 'report_sha256': self.report_sha256}
