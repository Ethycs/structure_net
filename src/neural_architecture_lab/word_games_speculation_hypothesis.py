"""Conservative indexing of measured smaller-Phi speculative decoding."""
from dataclasses import dataclass


@dataclass(frozen=True)
class WordGamesSpeculationEvidence:
    stage: str
    classification: str
    requested: int
    completed: int
    pairs: int
    exact_pairs: int
    campaign: str
    campaign_sha256: str
    report: str
    report_sha256: str

    def __post_init__(self):
        if self.stage not in ("pilot", "full") or self.completed != self.requested or self.completed < 1:
            raise ValueError("complete declared stage required")
        if not 0 <= self.exact_pairs <= self.pairs or self.pairs < 1:
            raise ValueError("invalid parity counts")
        if self.stage == "full" and (self.completed != 72 or self.pairs != 36):
            raise ValueError("complete 36-request full comparison required")
        if self.stage == "full" and self.classification == "observed_faster" and self.exact_pairs != self.pairs:
            raise ValueError("output parity gate required for speedup classification")
        for value in (self.campaign_sha256, self.report_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("SHA-256 required")

    def record(self):
        return {"id": "word-games-phi4-mini-speculation-v1", "name": "WORD_GAMES Phi-4-mini speculative decoding",
            "description": "Phi-4-mini 3.8B Q4_K_M drafts for Phi-4 14B Q4_K_M using candidate byte conversion across different tokenizers and target greedy verification. Matched frozen WORD_GAMES request replay on RTX 3060 plus RTX 2060 SUPER; timings include drafting overhead. Output parity is a separate gate. No training or universal speedup claim.",
            "project": "WORD_GAMES", "category": "inference", "tested": True, "confirmed": False,
            "confirmation_status": ("pilot_" if self.stage == "pilot" else "") + self.classification,
            "tested_scope": self.stage + " paired request replay; two GPUs; one model pair",
            "direct_experiment_count": self.completed, "pair_count": self.pairs, "exact_pairs": self.exact_pairs,
            "campaign": self.campaign, "campaign_sha256": self.campaign_sha256,
            "report": self.report, "report_sha256": self.report_sha256, "trained_parameters": 0}
