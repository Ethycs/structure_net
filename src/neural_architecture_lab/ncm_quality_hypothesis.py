"""Conservative independent evaluation record for one frozen NCM checkpoint."""
from dataclasses import dataclass

HYPOTHESIS_ID = 'llama32-3b-ncm-independent-quality-v1'


def classification(valid,student_passes,teacher_passes,n=200):
    if n!=200 or any(type(x) is not int or not 0<=x<=n for x in (student_passes,teacher_passes)):
        raise ValueError('complete fixed primary counts required')
    if not valid:return 'independent_quality_evidence_invalid'
    if student_passes<190:return 'independent_summary_quality_below_95'
    if student_passes<teacher_passes:return 'independent_summary_quality_not_retained'
    return 'independent_summary_quality_point_target_met'


@dataclass(frozen=True)
class QualityEvidence:
    classification: str
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    evaluated_outputs: int

    def record(self):
        if self.classification not in {'independent_quality_evidence_invalid','independent_summary_quality_below_95',
            'independent_summary_quality_not_retained','independent_summary_quality_point_target_met'}:
            raise ValueError('known classification required')
        if self.evaluated_outputs!=480:raise ValueError('complete frozen evaluation required')
        for h in (self.campaign_sha256,self.report_sha256):
            if len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('SHA256 required')
        return {'id':HYPOTHESIS_ID,'name':'Independent dialogue-summary grading of Llama-distilled NCM',
            'description':'Frozen approximately 149M-parameter Next-Concept Model and original Llama3.2-3B on 200 unused human-written DialogSum dialogues and 40 stress cases; calibrated blinded source-grounded judge, fixed cross-family audit, strict factuality/coverage/coherence rubric and binomial uncertainty. Public proxy data, not production accuracy.',
            'category':'architecture','tested':True,'confirmed':False,'status':'completed_exploratory',
            'underpowered':True,'independent_adaptation_seeds':1,'confirmation_status':self.classification,
            'direct_experiment_count':self.evaluated_outputs,'primary_source_count':200,'stress_source_count':40,
            'application_accuracy_tested':False,'mobile_performance_tested':False,'speed_tested':False,
            'campaign':self.campaign,'campaign_sha256':self.campaign_sha256,
            'report':self.report,'report_sha256':self.report_sha256}
