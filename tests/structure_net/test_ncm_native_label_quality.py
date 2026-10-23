import json
import pytest
from experiments.structure_net.ncm_native_label_quality import parse, quote_check


def test_semantic_failures_do_not_create_another_teacher_draw():
    row = {"id":"a","summary":"","claims":[{"claim":"wrong assertion","quote":"invented quote"}]}
    parsed = parse(json.dumps({"summaries":[row]}),["a"])
    assert parsed["a"] == row
    assert not quote_check(parsed["a"],"The actual source.")
    row["summary"] = "word "*300
    assert parse(json.dumps({"summaries":[row]}),["a"])["a"]["summary"] == row["summary"]


def test_id_coverage_and_literal_quotes_are_separate_checks():
    row = {"id":"a","summary":"A request.","claims":[{"claim":"A request.","quote":"Please help."}]}
    assert quote_check(row,"Please help. Thank you.")
    assert not quote_check(row,"please help. Thank you.")
    with pytest.raises(ValueError):
        parse(json.dumps({"summaries":[row,row]}),["a"])
    with pytest.raises(ValueError):
        parse(json.dumps({"summaries":[row]}),["a","b"])


def test_quota_recovery_cannot_send_before_documented_reset(monkeypatch):
    from datetime import timedelta
    from experiments.structure_net import ncm_native_label_quota_recovery as recovery
    class Earlier:
        @staticmethod
        def now(zone):
            return recovery.NOT_BEFORE-timedelta(seconds=1)
    monkeypatch.setattr(recovery,"datetime",Earlier)
    monkeypatch.setattr(recovery.old,"teacher",lambda *_:pytest.fail("teacher must not run before reset"))
    with pytest.raises(ValueError,match="reset has not occurred"):
        recovery.main("pilot")
