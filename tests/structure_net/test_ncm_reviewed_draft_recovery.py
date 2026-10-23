import pytest

from experiments.structure_net import ncm_reviewed_draft_data as data
from experiments.structure_net import ncm_reviewed_draft_recovery as recovery


def rows_and_scores():
    rows = [{"id": str(i), "text": "summary "+str(i)} for i in range(3)]
    scores = [{**row, "valid": True, "passed": i != 1} for i, row in enumerate(rows)]
    return rows, scores


def test_teacher_selection_preserves_every_semantic_and_capacity_failure():
    rows, scores = rows_and_scores()
    accepted, rejected = data.select_accepted(rows, scores, lambda r: r["id"] != "2")
    assert accepted == rows[:1]
    assert [r["label"] for r in rejected] == rows[1:]
    assert [r["score"] for r in rejected] == scores[1:]
    assert rejected[0]["format_capacity_fit"] is True
    assert rejected[1]["format_capacity_fit"] is False


@pytest.mark.parametrize("change", ["missing", "reorder", "text", "invalid", "duplicate"])
def test_invalid_or_changed_first_teacher_decisions_cannot_admit_training(change):
    rows, scores = rows_and_scores()
    if change == "missing": scores.pop()
    elif change == "reorder": scores.reverse()
    elif change == "text": scores[0]["text"] = "different candidate"
    elif change == "invalid": scores[0].update(valid=False, passed=None)
    elif change == "duplicate": rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError):
        data.select_accepted(rows, scores, lambda r: True)


def test_recovery_is_idempotent_and_keeps_all_prospective_roots_distinct():
    recovery.configure()
    producing = recovery.study.producing
    recovery.configure()
    assert recovery.study.producing is producing
    assert recovery.study.expanded.DEST == data.DEST
    assert recovery.word_workflow.DEST == recovery.WORD_WORKFLOW
    assert recovery.final_audit.WORKFLOW == recovery.FINAL_WORKFLOW
    assert recovery.final_workflow.DEST == recovery.FINAL_WORKFLOW
    assert recovery.development.study is recovery.study
    assert recovery.final.candidate is recovery.evaluation
    roots = [data.PILOT, data.DEST, recovery.WORD_WORKFLOW, recovery.FINAL_WORKFLOW,
        recovery.study.CACHE_PILOT, recovery.study.CACHE, recovery.study.PILOT, recovery.study.TRAIN,
        recovery.evaluation.PILOT, recovery.evaluation.OUTPUTS, recovery.evaluation.PACKETS, recovery.evaluation.GRADE,
        recovery.development.DEST, recovery.final.DATA, recovery.final.FEATURE_PILOT,
        recovery.final.FEATURES, recovery.final.PILOT, recovery.final.OUTPUTS,
        recovery.final.PACKETS, recovery.final.GRADE, recovery.final_audit.DEST]
    assert len(roots) == len(set(roots))
    assert all("reviewed_draft" in str(root) for root in roots)
    assert recovery.study.Config().updates == 32768
    assert recovery.final.minimum(256) == 244 and recovery.final.minimum(64) == 61


def test_all_conditional_final_children_use_the_same_recovery_configuration():
    for module, scope in (("experiments.neural_architecture_lab.finalize_ncm_generated_history_word", "development"),
            ("experiments.structure_net.ncm_sealed_final_quality", "final"),
            ("experiments.neural_architecture_lab.finalize_ncm_sealed_final_quality", "final-audit")):
        result = recovery.translate_final_command(["python", "-u", "-m", module, "finalize"])
        assert result[-3:] == [recovery.MODULE, scope, "finalize"]
    with pytest.raises(ValueError, match="unexpected"):
        recovery.translate_final_command(["python", "-u", "-m", "unregistered_module", "finalize"])


@pytest.mark.parametrize("scope,stage", [("data", "pilot"), ("study", "cache-pilot"), ("eval", "packets")])
def test_dispatch_calls_the_selected_existing_implementation(monkeypatch, scope, stage):
    calls = []
    monkeypatch.setattr(recovery.sys, "argv", ["recovery", scope, stage])
    if scope == "data":
        monkeypatch.setattr(data, "encode", lambda pilot: calls.append(pilot))
    elif scope == "study":
        monkeypatch.setattr(recovery.study, "cache", lambda pilot: calls.append(pilot))
    else:
        monkeypatch.setattr(recovery.evaluation.reference, "prepare_packets", lambda: calls.append(True))
    recovery.main()
    assert calls == [True]


def test_module_entry_reuses_the_canonical_adapter(monkeypatch):
    import runpy
    calls = []
    monkeypatch.setattr(recovery, "main", lambda: calls.append(True))
    runpy.run_module(recovery.MODULE, run_name="__main__")
    assert calls == [True]
