import pytest
import runpy
from pathlib import Path

from experiments.word_games.native_speed import aggregate, output_tokens


def test_visible_output_rate_excludes_eos_and_rejected_drafts_and_includes_prefill():
    response = {"tokens_predicted": 101, "stop_type": "eos", "timings": {
        "predicted_n": 101, "predicted_ms": 1000, "predicted_per_second": 100,
        "draft_n": 1000, "draft_n_accepted": 70}}
    r = aggregate([{"response": response, "elapsed_seconds": 2.5, "parsed": {}}])
    assert r["output_tokens_per_second"] == 40
    assert r["decode_only_tokens_per_second"] == 100
    assert r["drafted"] == 1000 and r["draft_accepted"] == 70
    assert r["truncated"] == 0
    assert output_tokens({**response, "stop_type": "limit"}) == 101


def test_aggregate_weights_by_time_instead_of_averaging_request_rates():
    def row(n, t):
        return {"elapsed_seconds": t, "parsed": {}, "response": {
            "tokens_predicted": n + 1, "stop_type": "eos", "timings": {
                "predicted_n": n + 1, "predicted_ms": t * 1000,
                "predicted_per_second": n / t}}}
    r = aggregate([row(10, 1), row(10, 9)])
    assert r["output_tokens_per_second"] == pytest.approx(2)


def test_registry_cannot_turn_pilot_or_decode_rate_into_40tps_claim():
    evidence = runpy.run_path(str(Path(__file__).resolve().parents[2] /
        "src/neural_architecture_lab/word_games_native_speed_hypothesis.py"))["WordGamesNativeSpeedEvidence"]
    args = dict(rate=40, pairs=36, quality_cases_per_arm=12, quality_passed=True,
                classification="observed_40tps_with_no_suite_regression",
                campaign="campaign.json", campaign_sha256="a" * 64,
                report="report.md", report_sha256="b" * 64)
    assert evidence(**args).record()["confirmed"] is False
    for change in ({"rate": 39.9}, {"quality_passed": False}, {"pairs": 3}):
        with pytest.raises(ValueError):
            evidence(**{**args, **change})


def test_live_pairing_preserves_authorization_and_reuses_only_synthetic_ids(tmp_path):
    from experiments.word_games.native_quality import matched_setup
    from experiments.word_games.scenarios import setup, cases
    from experiments.word_games.common import write
    case = cases()[0]
    _, baseline = setup(case)
    write(tmp_path / "quality/cases" / case["id"] / "context.json", baseline)
    engine, context = matched_setup(setup, tmp_path, case)
    assert context == baseline
    assert engine.context_summary(context_id=context["context_id"])["context_id"] == baseline["context_id"]
    def altered(c):
        e, ctx = setup(c)
        ctx["allowed_ops"] = []
        return e, ctx
    with pytest.raises(ValueError, match="authorization differs"):
        matched_setup(altered, tmp_path, case)
