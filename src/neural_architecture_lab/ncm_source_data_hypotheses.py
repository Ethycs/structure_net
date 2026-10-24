"""Typed data-preparation records that cannot establish NCM task accuracy."""
from dataclasses import dataclass

CONTROLS = {
    'ncm-additional10000-source-tokenization-v1': (
        'all10000_source_tokenization_alignment_replayed_no_features_or_labels', 10000, 10000,
        'Complete additional TRAIN source tokenization and alignment',
        'CPU replay of all10000 reserved public TRAIN sources under original tokenizers, prompt/date and complete character alignment; exact first/longest historical geometry parity. Source metadata, not model quality.'),
    'ncm-lossless-FP32-source-feature-storage-control-v1': (
        'all32_payloads_bitwise_FP32_reconstruction_and_storage_replayed', 16, 32,
        'Lossless FP32 source-feature disk cache control',
        'All32 interleaved/byte-shuffled zlib1 payloads on16 fixed original TRAIN feature tensors reconstruct every FP32 byte. Storage reduction of data only; no model compression, task accuracy or token-throughput claim.'),
    'ncm-additional10000-frozen-body-lossless-FP32-features-v1': (
        'all10000_lossless_FP32_source_cache_payloads_and_bindings_replayed', 10000, 10000,
        'Larger original frozen-body FP32 source-feature cache',
        'All10000 complete source feature payloads bind original frozen28-layer body, source, prompt and alignment, with lossless FP32 storage, original-reference parity and CPU payload replay; no student quality claim.')}


@dataclass(frozen=True)
class SourceDataEvidence:
    hypothesis_id: str
    classification: str
    source_cases: int
    measurements: int
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str

    def record(self):
        if self.hypothesis_id not in CONTROLS:
            raise ValueError('known data-only hypothesis required')
        expected, cases, measurements, name, description = CONTROLS[self.hypothesis_id]
        if (self.classification != expected or type(self.source_cases) is not int or self.source_cases != cases
                or type(self.measurements) is not int or self.measurements != measurements):
            raise ValueError('complete audited data-only scope and classification required')
        for value in (self.campaign_sha256, self.report_sha256):
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('exact SHA256 identities required')
        return {'id': self.hypothesis_id, 'name': name, 'description': description, 'category': 'architecture',
            'tested': True, 'confirmed': False, 'underpowered': True, 'status': 'completed_data_control',
            'classification': self.classification, 'source_cases': self.source_cases,
            'direct_experiment_count': self.measurements, 'data_preparation_only': True,
            'student_task_accuracy_measured': False, 'model_speed_measured': False, 'speed_tested': False,
            'final_test_opened': False, 'campaign': self.campaign, 'campaign_sha256': self.campaign_sha256,
            'report': self.report, 'report_sha256': self.report_sha256}
