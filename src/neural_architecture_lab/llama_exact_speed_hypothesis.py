"""Exactness failures veto promotion, even when another workload is faster."""
from dataclasses import dataclass
import math
from typing import ClassVar

CANDIDATES = ("mm", "packed_bmm", "packed_mm", "parallel2", "parallel3")
BLAS_CANDIDATES = ("cublaslt_bmm", "cublaslt_mm")
GRID = ((128, 0), (512, 0), (128, 32))


@dataclass(frozen=True)
class SpeedCell:
    candidate: str
    length: int
    steps: int
    exact: bool
    speedup: float | None

    def __post_init__(self):
        if self.candidate not in CANDIDATES + BLAS_CANDIDATES or (self.length, self.steps) not in GRID:
            raise ValueError("unknown fixed cell")
        if self.exact != (self.speedup is not None):
            raise ValueError("only exact cells may supply speed evidence")
        if self.speedup is not None and (not math.isfinite(self.speedup) or self.speedup <= 0):
            raise ValueError("finite positive speedup required")


@dataclass(frozen=True)
class ExactSpeedEvidence:
    candidate_names: ClassVar[tuple] = CANDIDATES
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str
    cells: tuple[SpeedCell, ...]

    def __post_init__(self):
        expected = {(n, length, steps) for n in self.candidate_names for length, steps in GRID}
        if len(self.cells) != len(expected) or {(c.candidate, c.length, c.steps) for c in self.cells} != expected:
            raise ValueError("complete fixed grid required")
        for value in (self.campaign_sha256, self.report_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("SHA-256 required")

    def record(self):
        eligible = [n for n in self.candidate_names if all(c.exact for c in self.cells if c.candidate == n)]
        promoted = [{"candidate": c.candidate, "length": c.length, "decode_steps": c.steps}
                    for c in self.cells if c.candidate in eligible and c.speedup >= 1.10]
        classification = ("exact_speedup_over_graph_supported" if promoted else
                          "no_qualifying_exact_speedup" if eligible else "no_exact_candidate")
        return {"id": "llama32-3b-exact-speed-iteration-v1", "name": "Llama exact projection speed iteration",
                "description": "Five projection dispatch/packing/scheduling candidates against the existing complete-model CUDA graph baseline. Fifteen fixed cells; bitwise logits, KV and generation controls veto approximate candidates. Ten alternating timing pairs only for exact cells. No weight compression or application accuracy claim.",
                "category": "architecture", "tested": True, "confirmed": False,
                "confirmation_status": classification, "status": "completed_exploratory_speed_iteration",
                "campaign": self.campaign, "campaign_sha256": self.campaign_sha256,
                "report": self.report, "report_sha256": self.report_sha256,
                "direct_experiment_count": len(self.cells), "eligible_candidates": eligible,
                "promoted": promoted, "rejected_cells": sum(not c.exact for c in self.cells),
                "parameter_bytes": 6425499648, "trained_parameters": 0,
                "application_accuracy_tested": False, "two_for_one_supported": False, "weight_compression_used": False}


@dataclass(frozen=True)
class ExactBlasEvidence(ExactSpeedEvidence):
    candidate_names: ClassVar[tuple] = BLAS_CANDIDATES

    def record(self):
        record = super().record()
        record.update(id="llama32-3b-exact-blas-iteration-v1", name="Llama exact BLAS backend speed iteration",
                      description="Two cuBLASLt projection backend preference candidates against the existing complete-model cuBLAS CUDA graph baseline. Six fixed cells, bitwise logits/KV/generation controls and ten alternating timing pairs for exact cells. No application accuracy or compression claim.")
        return record
