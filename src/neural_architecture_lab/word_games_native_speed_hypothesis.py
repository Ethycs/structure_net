"""Conservative indexing of native Phi-4 throughput and task quality."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WordGamesNativeSpeedEvidence:
    rate: float
    pairs: int
    quality_cases_per_arm: int
    quality_passed: bool
    classification: str
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str

    def __post_init__(self):
        if self.pairs != 36 or self.quality_cases_per_arm != 12:
            raise ValueError("full request and live quality comparisons required")
        if not math.isfinite(self.rate) or self.rate <= 0:
            raise ValueError("positive measured rate required")
        expected = ("below_40tps" if self.rate < 40 else
                    "observed_40tps_with_no_suite_regression" if self.quality_passed else "quality_gate_failed")
        if self.classification != expected:
            raise ValueError("classification disagrees with primary gate")
        for h in (self.campaign_sha256, self.report_sha256):
            if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                raise ValueError("SHA-256 required")

    def record(self):
        return {"id": "word-games-phi4-native-40tps-v1", "name": "WORD_GAMES native Phi-4 toward 40 tokens/s",
                "description": "Pinned native llama.cpp CUDA and current-request prompt lookup for Phi-4 14B Q4_K_M, RTX 3060. Full wall time includes prefill; output counts exclude EOS and rejected drafts. Frozen 36-request replay and two 12-case live task suites. No cross-request cache reuse or training.",
                "project": "WORD_GAMES", "category": "inference", "tested": True, "confirmed": False,
                "confirmation_status": self.classification, "output_tokens_per_second": self.rate,
                "observed_quality_gate": self.quality_passed, "paired_requests": self.pairs,
                "quality_cases_per_arm": self.quality_cases_per_arm,
                "campaign": self.campaign, "campaign_sha256": self.campaign_sha256,
                "report": self.report, "report_sha256": self.report_sha256,
                "trained_parameters": 0, "tested_scope": "one GPU and twelve synthetic workflows"}
