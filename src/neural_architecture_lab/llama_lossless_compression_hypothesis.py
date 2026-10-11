"""Exploratory storage findings do not imply accuracy or compute improvements."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LosslessCompressionEvidence:
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    classification: str
    archive_count: int
    all_tensors_bitwise: bool
    original_raw_bytes: int
    original_archive_bytes: int
    factors_residual_archive_bytes: int

    def __post_init__(self):
        if self.archive_count != 199 or min(self.original_raw_bytes, self.original_archive_bytes, self.factors_residual_archive_bytes) <= 0:
            raise ValueError("complete positive storage totals required")
        expected = ("lossless_roundtrip_failed" if not self.all_tensors_bitwise else "lossless_storage_reduction"
                    if min(self.original_archive_bytes, self.factors_residual_archive_bytes) < self.original_raw_bytes
                    else "lossless_no_storage_reduction")
        if self.classification != expected:
            raise ValueError("classification contradicts measured exactness or storage")
        for value in (self.campaign_sha256, self.report_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("SHA-256 digest required")

    def record(self):
        return {"id": "llama32-3b-lossless-component-compression-v1",
                "name": "Llama exact component archive compression",
                "description": "Exhaustive bitwise round trips of original weights, rank-215 factors and FP32 residual shards using reversible layouts and Zstandard. Storage compression only; dense decode cost and prior accuracy failure remain.",
                "question": "Which existing lossless components shrink under reversible entropy coding?",
                "category": "architecture", "tested": True, "confirmed": False,
                "confirmation_status": self.classification, "status": "completed_exploratory_storage_study",
                "campaign": self.campaign, "campaign_sha256": self.campaign_sha256,
                "report": self.report, "report_sha256": self.report_sha256,
                "direct_experiment_count": self.archive_count, "all_tensors_bitwise": self.all_tensors_bitwise,
                "original_raw_bytes": self.original_raw_bytes, "original_archive_bytes": self.original_archive_bytes,
                "factors_residual_archive_bytes": self.factors_residual_archive_bytes,
                "application_accuracy_tested": False, "inference_speed_tested": False, "trained_parameters": 0}


@dataclass(frozen=True)
class PartialCompressionEvidence:
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    original_raw_bytes: int
    original_archive_bytes: int

    def record(self):
        if not 0 < self.original_archive_bytes < self.original_raw_bytes:
            raise ValueError("partial result requires measured original archive reduction")
        return {"id": "llama32-3b-lossless-component-compression-v1",
            "name": "Llama exact component compression, partial archival screen",
            "description": "Original model archived and bitwise verified; factor stage failed on non-contiguous storage. User redirected to resident memory and execution speed. No full factor-plus-residual archive total or deployment claim.",
            "category": "architecture", "tested": True, "confirmed": False,
            "confirmation_status": "original_archive_verified_factor_stage_failed",
            "status": "closed_partial_after_runtime_fault_and_scope_redirection",
            "campaign": self.campaign, "campaign_sha256": self.campaign_sha256,
            "report": self.report, "report_sha256": self.report_sha256,
            "direct_experiment_count": 1, "requested_archives": 199, "failed_archives": 1, "unrun_archives": 197,
            "original_raw_bytes": self.original_raw_bytes, "original_archive_bytes": self.original_archive_bytes,
            "original_tensors_bitwise": True, "full_campaign_completed": False,
            "application_accuracy_tested": False, "inference_speed_tested": False}
