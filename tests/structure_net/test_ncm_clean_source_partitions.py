from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.structure_net import ncm_clean_source_partitions as run
from experiments.neural_architecture_lab import finalize_ncm_clean_source_partitions as audit


def public_sample(sid, text):
    return {"haystack_session_ids": [sid], "haystack_sessions": [[{"role": "user", "content": text}]]}


def test_public_pool_merges_text_aliases_and_protects_reserved_identifier_families():
    text = " ".join(f"term{i}" for i in range(220))
    encoding = SimpleNamespace(encode=lambda text, **kwargs: text.split())
    reserved = [{"id": "reserved", "source_id": "sharegpt_protected_1", "group_id": "legacy-group",
        "source_sha256": "reserved-hash"}]
    samples = [public_sample("sharegpt_protected_4", text), public_sample("other_44", text),
        public_sample("unrelated_55", "Thanks!")]
    pool, _ = run.build_pool(samples, [], reserved, encoding)
    assert len(pool) == 1 and pool[0]["blocked"]
    assert pool[0]["candidate"] and len(pool[0]["origins"]) == 2
    assert "sharegpt_protected" in pool[0]["families"]


def test_export_is_a_complete_literal_public_turn_with_no_label_channels():
    text = " ".join(f"term{i}" for i in range(220))
    sample = public_sample("ultrachat_123", text)
    pool, _ = run.build_pool([sample], [], [], SimpleNamespace(encode=lambda text, **kwargs: text.split()))
    value = run.export(pool[0], "component", "development_short")
    assert value["source"] == text
    assert value["spans"][0]["start_char"] == 0 and value["spans"][0]["end_char"] == len(text)
    assert run.verify_cases([sample], [value]) == {value["id"]: run.text_hash(text)}
    assert not {"target", "target_text", "word_ids", "concept_ids"} & value.keys()


def test_closed_final_stat_check_never_opens_the_file(tmp_path, monkeypatch):
    path = tmp_path / "final_sources.json"
    path.write_text("sealed source")
    stat = path.stat()
    receipt = {"path": str(path), "creation_sha256": "a" * 64, "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino, "sealed_after_creation": True}
    def forbidden(*args, **kwargs):
        raise AssertionError("sealed source contents must stay closed")
    monkeypatch.setattr(Path, "open", forbidden)
    audit.seal_stat(receipt, path)
    receipt["size_bytes"] += 1
    with pytest.raises(ValueError, match="stat changed"):
        audit.seal_stat(receipt, path)


def test_independent_metadata_replay_retains_final_and_development_sizes(monkeypatch):
    monkeypatch.setattr(run, "FINAL", {0: 1, 1: 1})
    monkeypatch.setattr(run, "DEVELOPMENT", {0: 1, 1: 1})
    index = []
    for i in range(4):
        sha = run.core.digest(str(i))
        index.append({"id": run.core.digest(run.NAMESPACE + ":node:" + sha), "source_sha256": sha,
            "source_component": sha, "blocked": False, "candidate": True, "bin": i % 2,
            "tokens": 250 if i % 2 == 0 else 600})
    parts, _ = audit.replay_assignment(index)
    assert {k: len(v) for k, v in parts.items()} == {k: 1 for k in parts}
    index[0]["tokens"] = 100
    with pytest.raises(ValueError, match="length identity"):
        audit.replay_assignment(index)
