from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net import ncm_clean_development_features as run
from experiments.neural_architecture_lab import finalize_ncm_clean_development_features as audit


def test_real_tokenizers_keep_unicode_delimiters_and_every_source_token():
    native, gpt, storage = run.tokenizers()
    text = ("Résumé 🙂 東京: keep the £42 restriction and café details. " * 35) + "\n<passage> literal boundary."
    row = {"id": "unicode", "source": text, "source_sha256": run.text_hash(text),
        "split": "development_long", "source_cl100k_tokens": len(storage.encode(text))}
    result = run.prepare_row(row, "long_diagnostic", native, gpt, storage)
    assert result["source"] == text
    assert gpt.decode(result["source_ids"]) == text
    assert len(result["alignment"]) == len(result["source_ids"])
    assert all(abs(sum(w for _, w in refs) - 1) < 1e-6 for refs in result["alignment"])
    assert result["source_frame"] >= len(result["source_ids"])
    assert result["partition_split"] == "development_long"
    hidden = torch.ones(len(result["context_ids"]), 3)
    assert torch.allclose(run.base.align(hidden, result["alignment"]), torch.ones(len(result["source_ids"]), 3))
    for key in run.FORBIDDEN:
        with pytest.raises(ValueError, match="source-only"):
            run.prepare_row({**row, key: "label"}, "long_diagnostic", native, gpt, storage)
    with pytest.raises(ValueError, match="token range"):
        run.prepare_row({**row, "source_cl100k_tokens": row["source_cl100k_tokens"] + 1}, "long_diagnostic", native, gpt, storage)


def test_pilot_uses_longest_native_prompt_not_largest_GPT2_frame():
    parts = {"short_development": [{"id": "first", "context_ids": [1], "source_frame": 256}],
        "long_diagnostic": [{"id": "a", "context_ids": [1] * 4, "source_frame": 256},
            {"id": "b", "context_ids": [1] * 3, "source_frame": 8192}]}
    assert run.select_pilot(parts) == ["first", "a"]


def source_fixture(tmp_path, monkeypatch):
    for name in ("DEST", "RESULTS", "WORKFLOW"):
        path = tmp_path / name
        path.mkdir()
        monkeypatch.setattr(run.partitions, name, path)
    artifacts = {}
    for source_split in run.SPLITS:
        rows = []
        for i in range(64):
            text = f"{source_split} source {i}"
            rows.append({"id": f"{source_split}-{i}", "source_component": f"{source_split}-{i}",
                "source": text, "source_sha256": run.text_hash(text), "source_truncated": False, "split": source_split})
        path = run.partitions.DEST / (source_split + "_sources.json")
        run.write_json(path, rows)
        artifacts[str(path)] = run.file_hash(path)
    campaign = {"classification": "clean_exchange_sources_reserved_quality_unmeasured",
        "counts": {"final_short": 256, "final_long": 64, "development_short": 64, "development_long": 64},
        "final_test_opened": False, "model_outputs_or_labels_used": False, "artifacts": artifacts}
    path = run.partitions.RESULTS / "campaign_results.json"
    run.write_json(path, campaign)
    run.write_json(run.partitions.RESULTS / (run.partitions.HYPOTHESIS + "_registry_readback.json"), {
        "status": "registered_and_read_back", "record": {"campaign_sha256": run.file_hash(path)},
        "exact": {"campaign_sha256": run.file_hash(path)}})
    run.write_json(run.partitions.WORKFLOW / "result.json", {"status": "clean_exchange_partitions_completed",
        "stages": [{"stage": s, "exit_code": 0} for s in ("pilot", "prepare", "finalize", "register")]})


def test_admission_retains_full128_and_never_opens_sealed_sources(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.name.startswith(("sealed_", "final_short_sources", "final_long_sources")):
            raise AssertionError("sealed final content must stay closed")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    parts, inputs = run.admitted_sources()
    assert {s: len(v) for s, v in parts.items()} == run.EXPECTED
    assert not any(p.name.startswith("final_") for p in inputs)
    path = run.partitions.DEST / "development_short_sources.json"
    rows = json.loads(path.read_text())
    rows[0]["source"] += "changed"
    run.write_json(path, rows)
    with pytest.raises(ValueError, match="bytes changed"):
        run.admitted_sources()


def test_admission_rejects_incomplete_lifecycle(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)
    path = run.partitions.WORKFLOW / "result.json"
    done = json.loads(path.read_text())
    done["stages"][2]["exit_code"] = 1
    run.write_json(path, done)
    with pytest.raises(ValueError, match="reservation required"):
        run.admitted_sources()


def test_producer_fingerprint_changes_with_configuration_and_binds_sources(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)
    first = run.producing(tmp_path / "producer_a", "systems_pilot", [], {"pilot": True})
    other = run.producing(tmp_path / "producer_b", "systems_pilot", [], {"pilot": False})
    assert first["fingerprint"] != other["fingerprint"]
    assert first["hypothesis_id"] == run.HYPOTHESIS
    assert first["source_hashes"][str(Path(run.__file__))] == run.file_hash(run.__file__)
    assert not any(Path(p).name.startswith("final_") for p in first["input_hashes"])
    assert first["config"]["counts"] == run.EXPECTED
    run.base.verify(first)


def test_forecaster_wait_accepts_completed_fit_miss_but_not_missing_exits(tmp_path, monkeypatch):
    for name in ("WORKFLOW", "RESULTS"):
        path = tmp_path / name
        path.mkdir()
        monkeypatch.setattr(run.forecaster, name, path)
    assert run.forecaster_ready() is False
    campaign_path = run.forecaster.RESULTS / "campaign_results.json"
    run.write_json(campaign_path, {"forecaster_fit_gate": False})
    run.write_json(run.forecaster.RESULTS / (run.forecaster.HYPOTHESIS + "_registry_readback.json"), {
        "status": "registered_and_read_back", "record": {"campaign_sha256": run.file_hash(campaign_path)},
        "exact": {"campaign_sha256": run.file_hash(campaign_path)}})
    done = {"status": "dense_matched_forecaster_completed", "stages": [{"stage": s, "exit_code": 0}
        for s in ("train_pilot", "fit_pilot", "train", "fit", "cache", "finalize", "register")]}
    path = run.forecaster.WORKFLOW / "result.json"
    run.write_json(path, done)
    assert run.forecaster_ready() is True
    run.write_json(path, {**done, "stages": done["stages"][:-1]})
    with pytest.raises(ValueError, match="lifecycle"):
        run.forecaster_ready()
    run.write_json(run.forecaster.WORKFLOW / "failure.json", {"status": "stopped_failure_preserved"})
    with pytest.raises(RuntimeError, match="failure preserved"):
        run.forecaster_ready()


def test_feature_audit_checks_bytes_identity_dtype_shape_and_finiteness(tmp_path):
    path = tmp_path / "case.pt"
    row = {"id": "case", "source_sha256": "source", "alignment_sha256": "align",
        "native_prompt_sha256": "prompt", "source_ids": [1, 2]}
    saved = {k: v for k, v in row.items() if k != "source_ids"}
    saved["features"] = torch.zeros(2, 3072)

    def entry(value):
        torch.save(value, path)
        return {"path": str(path), "sha256": run.file_hash(path), "source_tokens": 2}

    audit.check_feature(row, entry(saved), path)
    changed = deepcopy(saved)
    changed["source_sha256"] = "other"
    with pytest.raises(ValueError, match="identity"):
        audit.check_feature(row, entry(changed), path)
    for tensor in (torch.zeros(1, 3072), torch.zeros(2, 3072, dtype=torch.float16),
            torch.full((2, 3072), float("nan")), torch.zeros(2, 3072, requires_grad=True)):
        with pytest.raises(ValueError, match="FP32"):
            audit.check_feature(row, entry({**saved, "features": tensor}), path)
    stale = entry(saved)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        audit.check_feature(row, stale, path)
