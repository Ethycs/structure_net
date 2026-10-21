"""Conservative bounded NCM replica and factorization result records."""
from dataclasses import dataclass


def replica_classification(parity_cells,eligible_rates,teacher_rate):
    if parity_cells!=1800:return 'replica_output_parity_failed'
    return 'replica_throughput_target_met' if eligible_rates and max(eligible_rates)>=2*teacher_rate else 'replica_throughput_target_unmet'


def factor_classification(passes,teacher_passes,bytes_ratio,latency_ratio,tps_ratio):
    if passes<95 or passes<teacher_passes:return 'factorized_quality_failed'
    if bytes_ratio>.9:return 'factorized_compression_target_unmet'
    if min(latency_ratio,tps_ratio)<1.05:return 'factorized_smaller_speed_target_unmet'
    return 'factorized_smaller_and_faster_on_bounded_task'


@dataclass(frozen=True)
class OptimizationEvidence:
    hypothesis_id: str
    classification: str
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    timing_cells: int

    def record(self):
        descriptions={
            'llama32-3b-ncm-replicas-v1':('Independent NCM copies under the Llama memory budget',
                'Literal independent Llama-distilled NCM copies on one GPU, separate CUDA graphs/streams, exact-output parity and aggregate common-token throughput, separate weight/allocated/reserved memory budgets. Reused single-request Llama is not an optimal batched-serving baseline.'),
            'llama32-3b-ncm-factorization-v1':('Low-rank generator MLPs in Llama-distilled NCM',
                'Activation-RMS-scaled rank256 generator MLP factorization, preserved biases/codec/attention, and independent two-block noisy neighborhood repair. Task quality and measured complete-request speed remain separate from local hidden-state error.'),
            'llama32-3b-ncm-batching-v1':('Shared-weight NCM request batching',
                'One frozen NCM serves fixed batches1/4/9 on one GPU with native-output parity to original generation. Actual common output tokens and complete wave wall time exclude dummy outputs from static partial waves. Compared with fresh batch1; no optimized Llama batching claim.'),
            'llama32-3b-ncm-host-io-v1':('Lossless NCM fixed-vocabulary validation and serving',
                'Cache the immutable tokenizer vocabulary size while retaining every ID/EOS/PAD/text check. Compare independent copies and shared-weight batching against a fresh original Llama3B control, with native-output parity, complete-request throughput and distinct memory budgets.')}
        if self.hypothesis_id not in descriptions:raise ValueError('known optimization hypothesis required')
        for h in (self.campaign_sha256,self.report_sha256):
            if len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('SHA256 required')
        name,description=descriptions[self.hypothesis_id]
        return {'id':self.hypothesis_id,'name':name,'description':description,'category':'architecture',
            'tested':True,'confirmed':False,'status':'completed_exploratory','underpowered':True,
            'confirmation_status':self.classification,'direct_experiment_count':self.timing_cells,
            'campaign':self.campaign,'campaign_sha256':self.campaign_sha256,
            'report':self.report,'report_sha256':self.report_sha256,
            'application_accuracy_tested':False,'mobile_performance_tested':False}
