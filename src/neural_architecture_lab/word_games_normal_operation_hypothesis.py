"""Measured ordinary functionality with separate independent/workflow counts."""
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalOperationEvidence:
    independent_correct: int
    workflow_correct: int
    workflow_completed: int
    followup_correct: int
    analysis: str
    analysis_sha256: str
    report: str
    report_sha256: str

    def __post_init__(self):
        if not (0 <= self.independent_correct <= 4 and 0 <= self.workflow_correct <= self.workflow_completed <= 7
                and 0 <= self.followup_correct <= 3):
            raise ValueError('invalid separate case counts')
        for value in (self.analysis_sha256, self.report_sha256):
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('SHA256 required')

    def record(self):
        passed = self.independent_correct == 4 and self.workflow_correct == 7 and self.followup_correct == 3
        return {'id': 'word-games-normal-operation-v1', 'name': 'WORD_GAMES ordinary live operation',
            'description': 'Deployed Triton Qwen live sender, deterministic firewall and receiver: '
                'four independent ordinary tasks, seven dependent workflow events with early stop, '
                'and separately declared lookup/move/return coverage. No frozen reply replay or oracle repair.',
            'category': 'functionality', 'project': 'WORD_GAMES', 'tested': True, 'confirmed': False,
            'confirmation_status': 'exploratory_structured_checks_passed' if passed else 'exploratory_normal_operation_failures',
            'independent_correct': self.independent_correct, 'independent_cases': 4,
            'workflow_correct': self.workflow_correct, 'workflow_completed': self.workflow_completed,
            'workflow_planned': 7, 'followup_correct': self.followup_correct, 'followup_cases': 3,
            'population_accuracy_established': False, 'independent_semantic_judge': False,
            'serving_configuration_changed': False, 'model_selects_metaphor': True,
            'analysis': self.analysis, 'analysis_sha256': self.analysis_sha256,
            'report': self.report, 'report_sha256': self.report_sha256}
