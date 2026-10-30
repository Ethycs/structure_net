"""Conservative evidence record for the WORD_GAMES project."""
from dataclasses import dataclass


@dataclass(frozen=True)
class WordGamesEvidence:
    campaign: str
    campaign_sha256: str
    audit: str
    audit_sha256: str
    report: str
    report_sha256: str
    total: int
    sender_correct: int
    boundary_correct: int
    receiver_correct: int
    disclosure_failures: int

    def __post_init__(self):
        for value in (self.campaign_sha256, self.audit_sha256, self.report_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("SHA-256 required")
        if self.total != 12:
            raise ValueError("all twelve planned cases required")
        for count in (self.sender_correct, self.boundary_correct, self.receiver_correct, self.disclosure_failures):
            if not isinstance(count, int) or not 0 <= count <= self.total:
                raise ValueError("invalid count")

    def record(self):
        return {"id": "word-games-phi4-authorized-view-v1", "name": "WORD_GAMES model-selected metaphor firewall",
            "description": "Two Phi-4 14B contexts separated by a typed host disclosure policy. Phi chooses a metaphor from public resource-placement task structure. Twelve synthetic cases distinguish sender action accuracy, receiver state interpretation and observed private-canary disclosure. Pilot prompt development is preserved; no general secrecy or task-adaptation claim.",
            "category": "information_flow", "project": "WORD_GAMES", "tested": True, "confirmed": False,
            "status": "completed_exploratory", "tested_scope": "one task family; twelve synthetic cases; one model",
            "confirmation_status": "synthetic_disclosure_checks_passed" if not self.disclosure_failures else "disclosure_failure",
            "direct_experiment_count": self.total, "sender_correct": self.sender_correct,
            "boundary_correct": self.boundary_correct, "receiver_correct": self.receiver_correct,
            "disclosure_failures": self.disclosure_failures, "model_selects_metaphor": True,
            "campaign": self.campaign, "campaign_sha256": self.campaign_sha256,
            "audit": self.audit, "audit_sha256": self.audit_sha256,
            "report": self.report, "report_sha256": self.report_sha256,
            "trained_parameters": 0, "independent_model_replications": 1}
