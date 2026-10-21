"""Interpret strict model grades without upgrading an unqualified accuracy judge."""
from dataclasses import dataclass

HYPOTHESIS_ID='llama32-3b-ncm-independent-quality-v1'
CLASSIFICATION='ncm_transfer_failed_accuracy_metric_unqualified'


@dataclass(frozen=True)
class AssessedQualityEvidence:
    campaign: str
    campaign_sha256: str
    assessment: str
    assessment_sha256: str
    report: str
    report_sha256: str

    def record(self):
        for h in (self.campaign_sha256,self.assessment_sha256,self.report_sha256):
            if len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('SHA256 required')
        return {'id':HYPOTHESIS_ID,'name':'NCM summary transfer failure with an unqualified accuracy rubric',
            'description':'One frozen Llama-distilled NCM scored 0/200 public-dialogue and 0/40 stress strict passes and emitted unrelated project templates. The judge accepted only 11/40 human reference summaries; its strict pass rate is not calibrated human-rated accuracy. Original raw operational below-95 gate retained; source transfer failed, production accuracy unmeasured.',
            'category':'architecture','tested':True,'confirmed':False,'status':'completed_exploratory',
            'underpowered':True,'independent_adaptation_seeds':1,'confirmation_status':CLASSIFICATION,
            'operational_classification':'independent_summary_quality_below_95',
            'direct_experiment_count':480,'primary_source_count':200,'stress_source_count':40,
            'human_reference_diagnostic_outputs':40,'judge_qualified_for_accuracy':False,
            'application_accuracy_tested':False,'mobile_performance_tested':False,'speed_tested':False,
            'campaign':self.campaign,'campaign_sha256':self.campaign_sha256,
            'assessment':self.assessment,'assessment_sha256':self.assessment_sha256,
            'report':self.report,'report_sha256':self.report_sha256}
