"""Conservative typed record for frozen-prefix execution experiments."""
from dataclasses import dataclass


@dataclass(frozen=True)
class NonattentionEvidence:
    campaign_path: str
    campaign_sha256: str
    report_path: str
    report_sha256: str
    classification: str
    screen_cells: int
    regression_cells: int
    regression_passes: int
    selected_execution: str | None

    def __post_init__(self):
        if self.screen_cells != 16 or self.regression_cells not in (0, 10):
            raise ValueError("requires the complete fixed screen and scheduled regression")
        if not 0 <= self.regression_passes <= self.regression_cells:
            raise ValueError("invalid pass count")
        if self.classification == "execution_regression_supported" and (self.regression_cells != 10 or self.regression_passes != 10):
            raise ValueError("support requires all ten joint passes")
        if self.classification not in ("execution_regression_supported", "execution_regression_not_confirmed", "no_execution_candidate_passed"):
            raise ValueError("unknown decision")
        if self.classification == "no_execution_candidate_passed" and (self.regression_cells or self.selected_execution is not None):
            raise ValueError("a failed screen cannot execute conditional regression")
        if self.regression_cells and self.selected_execution is None:
            raise ValueError("regression requires a selected execution")
        if any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in (self.campaign_sha256, self.report_sha256)):
            raise ValueError("requires SHA-256 hashes")

    def record(self):
        return {"id": "qwen3-8b-prefix6-nonattention-acceleration-v1",
                "name": "Qwen frozen-prefix nonattention acceleration",
                "description": "Measured FP16 fusion and INT8 MLP copies with official MoBA unchanged. Fixed validation screen and conditional reused-window regression assess prefix latency and readout fidelity; application accuracy is unmeasured.",
                "question": "Can execution changes outside official MoBA reduce frozen six-layer prefix latency by at least 5% while preserving readouts?",
                "category": "architecture", "tested": True, "confirmed": False,
                "confirmation_status": self.classification, "status": "completed_exploratory_execution_study",
                "tested_scope": "two_corpus_validation_screen_and_conditional_reused_window_regression",
                "evidence_count": self.screen_cells + self.regression_cells,
                "direct_experiment_count": self.screen_cells + self.regression_cells,
                "regression_passes": self.regression_passes, "selected_execution": self.selected_execution,
                "campaign": self.campaign_path, "campaign_sha256": self.campaign_sha256,
                "report": self.report_path, "report_sha256": self.report_sha256,
                "application_accuracy_tested": False, "trained_parameters": 0}
