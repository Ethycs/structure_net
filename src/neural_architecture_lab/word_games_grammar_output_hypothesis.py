"""Conservative records for the fixed-suite grammar download experiments."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GrammarOutputEvidence:
    hypothesis_id: str
    control_tok_s: float
    candidate_tok_s: float
    control_raw_bytes: int
    candidate_raw_bytes: int
    faster_pairs: int
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str

    def __post_init__(self):
        if self.hypothesis_id not in (
            'word-games-grammar-on-demand-v1', 'word-games-grammar-on-demand-v2'
        ):
            raise ValueError('unknown hypothesis')
        if any(not math.isfinite(v) or v <= 0 for v in (self.control_tok_s, self.candidate_tok_s)):
            raise ValueError('positive finite throughput required')
        if not 0 <= self.candidate_raw_bytes < self.control_raw_bytes:
            raise ValueError('measured transfer reduction required')
        if self.faster_pairs not in (0, 1, 2):
            raise ValueError('two paired full passes required')
        for value in (self.campaign_sha256, self.report_sha256):
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('SHA256 required')

    def record(self):
        classification = ('exact_outputs_less_transfer_faster_in_both_pairs'
                          if self.faster_pairs == 2 else
                          'exact_outputs_less_transfer_no_consistent_speedup')
        return {
            'id': self.hypothesis_id, 'name': 'WORD_GAMES grammar logits on demand',
            'description': 'Greedy grammar sampling with deferred full-vocabulary GPU readback; '
                           'matched fixed 36-request suite, two reversed-order full pairs.',
            'project': 'WORD_GAMES', 'category': 'inference',
            'tested': True, 'confirmed': False, 'confirmation_status': classification,
            'control_tok_s': self.control_tok_s, 'candidate_tok_s': self.candidate_tok_s,
            'throughput_ratio': self.candidate_tok_s / self.control_tok_s,
            'control_raw_bytes': self.control_raw_bytes,
            'candidate_raw_bytes': self.candidate_raw_bytes,
            'raw_bytes_reduction': 1 - self.candidate_raw_bytes / self.control_raw_bytes,
            'faster_pairs': self.faster_pairs, 'full_pairs': 2, 'paired_request_comparisons': 72,
            'exact_replay_outputs': True, 'service_restored': True,
            'population_accuracy_established': False, 'production_deployed': False,
            'trained_parameters': 0, 'campaign': self.campaign,
            'campaign_sha256': self.campaign_sha256, 'report': self.report,
            'report_sha256': self.report_sha256,
        }


def summarize(campaign):
    """Reject incomplete/invalid runs before deriving any successful record."""
    if campaign['error'] is not None or not campaign['service_restored']:
        raise ValueError('completed run and service restoration required')
    groups = {arm: [] for arm in ('control', 'candidate')}
    faster = 0
    for replay in ('full_1', 'full_2'):
        parity = campaign['parity'][replay]
        if not parity['equal'] or parity['requests'] != 36 or parity['mismatches']:
            raise ValueError('exact full-suite equality required')
        pair = {}
        for arm in groups:
            matches = [r for r in campaign['results'] if r['replay'] == replay and r['arm'] == arm]
            if len(matches) != 1:
                raise ValueError('exactly one result per arm and pair required')
            r = matches[0]
            t = r['totals']
            if (not r['completed'] or r['error'] is not None or t['calls'] != 36
                    or t['json_objects'] != 36 or t['truncated'] != 0):
                raise ValueError('incomplete or invalid request results')
            if not math.isfinite(t['request_seconds']) or t['request_seconds'] <= 0 or t['output_tokens'] <= 0:
                raise ValueError('invalid throughput measurements')
            pair[arm] = r
            groups[arm].append(r)
        c, p = pair['control'], pair['candidate']
        if c['totals']['output_tokens'] != p['totals']['output_tokens']:
            raise ValueError('output count mismatch')
        if not 0 <= p['downloads']['raw_bytes'] < c['downloads']['raw_bytes']:
            raise ValueError('each pair must reduce raw downloads')
        faster += p['totals']['request_seconds'] < c['totals']['request_seconds']
    result = {'faster_pairs': faster}
    for arm, cells in groups.items():
        result[arm + '_tok_s'] = (sum(r['totals']['output_tokens'] for r in cells)
                                 / sum(r['totals']['request_seconds'] for r in cells))
        result[arm + '_raw_bytes'] = sum(r['downloads']['raw_bytes'] for r in cells)
    return result
