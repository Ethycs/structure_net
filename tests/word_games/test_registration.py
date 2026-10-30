from pathlib import Path
import runpy

import pytest

Evidence = runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/neural_architecture_lab/word_games_hypothesis.py"))["WordGamesEvidence"]


def evidence(**change):
    return Evidence(**{**dict(campaign="campaign.json", campaign_sha256="a" * 64,
        audit="audit.json", audit_sha256="b" * 64, report="report.md", report_sha256="c" * 64,
        total=12, sender_correct=12, boundary_correct=12, receiver_correct=12, disclosure_failures=0), **change})


def test_perfect_synthetic_counts_do_not_confirm_general_firewall():
    r = evidence().record()
    assert r["tested"] and not r["confirmed"]
    assert r["confirmation_status"] == "synthetic_disclosure_checks_passed"
    assert evidence(disclosure_failures=1).record()["confirmation_status"] == "disclosure_failure"


@pytest.mark.parametrize("change", [{"total": 11}, {"receiver_correct": 13}, {"disclosure_failures": -1}])
def test_incomplete_or_impossible_counts_are_rejected(change):
    with pytest.raises(ValueError):
        evidence(**change)
