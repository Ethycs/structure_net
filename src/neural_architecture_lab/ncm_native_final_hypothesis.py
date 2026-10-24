"""Conservative fixed-checkpoint sealed320 complete-summary quality evidence."""
from dataclasses import dataclass

HYPOTHESIS_ID = 'ncm-lm1b-native655-source-only-sealed320-v1'
CLASSES = {'native655_sealed320_quality_qualified', 'native655_sealed320_quality_below_95', 'native655_sealed320_judgments_invalid'}


@dataclass(frozen=True)
class NativeFinalEvidence:
    classification: str
    quality_gate_passed: bool
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    checkpoint_sha256: str
    joint_valid_cases: int

    def record(self):
        if (self.classification not in CLASSES or type(self.quality_gate_passed) is not bool
                or self.quality_gate_passed != (self.classification == 'native655_sealed320_quality_qualified')):
            raise ValueError('matching fixed final classification and quality gate required')
        if (type(self.joint_valid_cases) is not int or not 0 <= self.joint_valid_cases <= 320
                or (self.quality_gate_passed and self.joint_valid_cases != 320)):
            raise ValueError('complete final validity count required for a qualified joint score')
        for value in (self.campaign_sha256, self.report_sha256, self.checkpoint_sha256):
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('exact SHA256 identities required')
        return {'id': HYPOTHESIS_ID, 'name': 'Full-native655 fixed-checkpoint sealed320 summary quality',
            'description': 'Single-use source-disjoint256short+64long final evaluation of the same full native checkpoint that qualified on both DEV cohorts. Complete raw sources through unchanged frozen Llama body, hard native generation,128stored-token semantic rubric and same-case joint244/256+61/64two-judge gates. No exact-summary matching or model-speed claim.',
            'category': 'architecture', 'tested': True, 'confirmed': False, 'underpowered': True,
            'classification': self.classification, 'quality_gate_passed': self.quality_gate_passed,
            'final_task_accuracy_measured': self.joint_valid_cases == 320, 'joint_valid_cases': self.joint_valid_cases,
            'final_source_cases': 320, 'source_prediction': True,
            'oracle_target_codes': False, 'exact_summary_matching': False, 'final_test_opened': True,
            'speed_tested': False, 'independent_adaptation_seeds': 1, 'campaign': self.campaign,
            'campaign_sha256': self.campaign_sha256, 'report': self.report, 'report_sha256': self.report_sha256,
            'checkpoint_sha256': self.checkpoint_sha256}
