import pytest

from experiments.structure_net import ncm_clean_semantic_development as run
from experiments.neural_architecture_lab import finalize_ncm_clean_semantic_development as audit


def source(name, sid, text, spans=()):
    return {"id": name, "source_id": sid, "group_id": name, "source": text,
        "source_sha256": run.development.text_hash(text), "spans": list(spans)}


def test_actual_source_guard_rejects_family_variants_and_normalized_duplicates():
    train = source("t", "sharegpt_topic_1", "A different training passage about navigation.")
    heldout = source("d", "sharegpt_topic_2", "Unrelated words cannot hide the same family.")
    with pytest.raises(ValueError, match="source family/text/turn"):
        run.source_guard([train], [heldout], {})
    train = source("t", "ultrachat_1", "Résumé CAFe and all important named restrictions remain.")
    heldout = source("d", "ultrachat_2", "résumé cafe AND ALL important named restrictions remain!")
    with pytest.raises(ValueError, match="source family/text/turn"):
        run.source_guard([train], [heldout], {})


def test_actual_source_guard_uses_same_informative_turn_threshold_and_pairwise_check():
    span = {"source_id": "greeting", "turn_text_sha256": "thanks"}
    a = source("a", "ultrachat_1", "Different first content.", [span])
    b = source("b", "ultrachat_2", "Other second content.", [span])
    run.source_guard([a], [b], {"thanks": 191})
    with pytest.raises(ValueError, match="source family/text/turn"):
        run.source_guard([a], [b], {"thanks": 192})
    with pytest.raises(ValueError, match="development source"):
        run.source_guard([], [a, {**a, "id": "another"}], {"thanks": 191})


def test_source_prediction_rejects_all_label_and_candidate_channels():
    for name in run.FORBIDDEN:
        with pytest.raises(ValueError, match="cannot receive labels"):
            run.predict(None, None, {name: "label"}, {}, None)


def test_parity_uses_full_population_reload_probes_outside_the_fixed32(tmp_path, monkeypatch):
    monkeypatch.setattr(run.word, "TRAIN", tmp_path)
    rows = [{**source(name, name, "source"), "split": "train", "source_ids": [1], "source_frame": 256,
        "alignment_sha256": "a", "native_prompt_sha256": "b", "word_ids": [99]} for name in ("outside32_a", "outside32_b")]
    probes = [{"id": r["id"], "raw_word_ids": [[7, 50256] + [0] * 190],
        "coarse_codes": [[0] * 608], "latent_sha256": "a" * 64} for r in rows]
    run.write_json(tmp_path / "source_only_reload_probe.json", probes)
    monkeypatch.setattr(run.word.forecaster, "data_inputs", lambda: (rows, {"features": "manifest"}, [{"id": "a_different_fixed_probe"}]))
    selected, _, expected = run.training_parity_inputs()
    assert [r["id"] for r in selected] == [r["id"] for r in rows]
    assert all(not run.FORBIDDEN & r.keys() for r in selected)
    assert expected[rows[0]["id"]]["raw_word_ids"] == probes[0]["raw_word_ids"][0]
    assert len(expected[rows[0]["id"]]["actual_coarse_codes"]) == 608


def test_semantic_joint_gate_uses_same_cases_and_keeps_invalid_denominator():
    secondary = [{"id": str(i), "valid": True, "passed": i < 61} for i in range(64)]
    primary = [{"id": str(i), "valid": True, "passed": i not in (58, 59, 60)} for i in range(64)]
    result = audit.joint_counts(secondary, primary)
    assert result["cases"] == 64 and result["passed"] == 58
    assert audit.joint_counts(secondary, None) is None
    primary[0].update(valid=False, passed=None)
    result = audit.joint_counts(secondary, primary)
    assert result["invalid"] == 1 and result["task_accuracy"] is None


def test_no_quality_selection_or_negative_first_judgment_overwrite():
    rows = [{"id": f"{split}-{i}", "split": split, "source_tokens": n, "text": ""}
        for split in run.reference.COHORTS for i, n in enumerate((100, 90, 120))]
    assert [r["id"] for r in run.judge_probes(rows)] == [rows[i]["id"] for i in (0, 2, 3, 5)]
    assert run.old.judge_pilot_valid({"cases": 4, "valid": 4, "passed": 0})
    cache = {}
    first = {"passed": False}
    run.reference.first.remember(cache, "key", first)
    with pytest.raises(ValueError, match="cannot be overwritten"):
        run.reference.first.remember(cache, "key", {"passed": True})
    assert cache["key"] == first


def test_pending_semantic_producer_binds_all_prior_firsts_without_future_model_reads(tmp_path, monkeypatch):
    first = run.producing(tmp_path / "first", "systems_producer_contract", pending=True)
    monkeypatch.setattr(run, "CONSTRUCTOR_SEED", run.CONSTRUCTOR_SEED + 1000)
    changed = run.producing(tmp_path / "changed", "systems_producer_contract", pending=True)
    assert first["fingerprint"] != changed["fingerprint"]
    assert first["input_hashes"][str(run.OLD_CACHE)] == run.CACHE_SHA
    assert first["config"]["canonical_prior_firsts"] == 1797
    assert str(run.word.TRAIN / "model.pt") not in first["input_hashes"]
    assert str(run.SOURCES / "cases.json") not in first["input_hashes"]
    assert first["config"]["selected_checkpoint_sha256"] is None
    assert not any("sealed_test_sources.json" in p or "final_short_sources.json" in p
        or "final_long_sources.json" in p for p in first["input_hashes"])
    run.reference.data.verify(first)


def test_semantic_admission_allows_fit_miss_and_rejects_rewritten_feature_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "configure", lambda: None)
    monkeypatch.setattr(run, "word_completed", lambda: True)
    monkeypatch.setattr(run.word, "completed", lambda *args: True)
    monkeypatch.setattr(run.reference.data, "terminal", lambda path: None)
    locations = ((run, "WORKFLOW"), (run.word, "TRAIN"), (run.word, "RESULTS"),
        (run.development, "FEATURES"), (run.development, "RESULTS"))
    for i, (module, name) in enumerate(locations):
        path = tmp_path / str(i)
        path.mkdir()
        monkeypatch.setattr(module, name, path)
    cases = {s: [{"id": f"{s}-{i}"} for i in range(64)] for s in run.reference.COHORTS}
    monkeypatch.setattr(run, "sources", lambda: cases)
    trained = run.word.TRAIN
    (trained / "model.pt").write_bytes(b"selected-model")
    checkpoint = run.file_hash(trained / "model.pt")
    monkeypatch.setattr(run, "CHECKPOINT", checkpoint)
    run.write_json(trained / "result.json", {"updates": 32768, "checkpoint_sha256": checkpoint})
    for name in ("source_only_reload_probe.json", "producer.json", "worker_terminal_receipt.json"):
        run.write_json(trained / name, {})
    campaign_path = run.word.RESULTS / "campaign_results.json"
    run.write_json(campaign_path, {"training_cases": 690, "actual_history_fit_gate": False,
        "artifacts": {str(p): run.file_hash(p) for p in trained.iterdir()}})
    features = run.development.FEATURES
    manifest_path = features / "manifest.json"
    run.write_json(manifest_path, {r["id"]: {} for rows in cases.values() for r in rows})
    result = {"manifest_sha256": run.file_hash(manifest_path), "cases": 128,
        "parent_body_hash_unchanged": run.reference.data.EXPECTED_BODY, "all_source_tokens_retained": True}
    run.write_json(features / "result.json", result)
    for name in ("producer.json", "worker_terminal_receipt.json"):
        run.write_json(features / name, {})
    run.write_json(run.development.RESULTS / "campaign_results.json", {
        "artifacts": {str(p): run.file_hash(p) for p in features.iterdir()}})
    admission = {"word_campaign_sha256": run.file_hash(campaign_path),
        "feature_manifest_sha256": run.file_hash(manifest_path)}
    run.write_json(run.WORKFLOW / "admission.json", admission)
    actual, _ = run.inputs()
    assert actual == cases  # A valid TRAIN fitting miss cannot suppress task evaluation.
    run.write_json(manifest_path, {"rewritten": {}})
    run.write_json(features / "result.json", {**result, "manifest_sha256": run.file_hash(manifest_path)})
    run.write_json(run.WORKFLOW / "admission.json", {**admission, "feature_manifest_sha256": run.file_hash(manifest_path)})
    with pytest.raises(ValueError, match="registered original-body"):
        run.inputs()
