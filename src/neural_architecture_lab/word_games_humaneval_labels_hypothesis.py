"""Conservative record for a paired 32-problem public coding sample."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HumanEvalLabelsEvidence:
    native: int
    relabeled: int
    both: int

    def __post_init__(self):
        if any(type(x) is not int or not 0 <= x <= 32 for x in (self.native, self.relabeled, self.both)):
            raise ValueError('Invalid task counts')
        if not max(0, self.native + self.relabeled - 32) <= self.both <= min(self.native, self.relabeled):
            raise ValueError('Impossible paired counts')

    def record(self):
        return {'id': 'word-games-humaneval-labels-v1',
            'name': 'WORD_GAMES paired HumanEval interface relabeling',
            'description': f'Terra interprets and Qwen executes 32 preselected public HumanEval tasks. Direct {self.native}/32, Qwen interface-relabeled {self.relabeled}/32; {self.both} both pass. Official tests and normal actor completion required. One sample per arm, tool-assisted controller pipeline. No general coding accuracy, privacy or bare-model leaderboard claim.',
            'project': 'WORD_GAMES', 'category': 'information_flow',
            'tested': True, 'confirmed': False,
            'confirmation_status': 'sampled_public_coding_evaluated',
            'tested_scope': '32 of 164 HumanEval functions selected with seed 731113; one generation per arm; interface labels only',
            'direct_experiment_count': 32, 'native_correct': self.native,
            'relabeled_correct': self.relabeled, 'both_correct': self.both,
            'direct_only': self.native - self.both, 'relabeled_only': self.relabeled - self.both,
            'general_accuracy_established': False, 'information_secrecy_tested': False,
            'production_deployed': False}
