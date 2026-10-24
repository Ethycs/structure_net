import pytest

from experiments.structure_net import ncm_pinned_date_development as run


def test_explicit_date_is_supplied_to_text_and_token_paths():
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return kwargs
        def __call__(self, *args, **kwargs):
            return args, kwargs
    pinned = run.PinnedDateTokenizer(Tokenizer())
    assert pinned.apply_chat_template([], tokenize=False)["date_string"] == "23 Oct 2026"
    assert pinned.apply_chat_template([], tokenize=True)["date_string"] == "23 Oct 2026"
    assert pinned("source", add_special_tokens=False)[0] == ("source",)
    with pytest.raises(ValueError, match="conflicting"):
        pinned.apply_chat_template([], date_string="24 Oct 2026")


def test_date_change_must_be_unique_before_source_and_preserve_every_other_field():
    first = {"context_ids": [1, 2, 3, 4], "native_prompt_sha256": "prior", "source": "source",
        "alignment": [[[2, 1.0]], [[3, 1.0]]]}
    second = {**first, "context_ids": [1, 9, 3, 4], "native_prompt_sha256": run.q.digest([1, 9, 3, 4])}
    assert run.require_date_only_change(first, second) == [1]
    for changed in ({**second, "source": "changed"}, {**second, "extra": True},
            {**second, "context_ids": [1, 2, 9, 4]}, {**second, "context_ids": [9, 9, 3, 4]},
            {**second, "context_ids": [1, 9, 3]}, {**second, "native_prompt_sha256": "wrong"}):
        with pytest.raises(ValueError):
            run.require_date_only_change(first, changed)


def test_fingerprint_binds_date_latest_firsts_and_actual_selected_weights(tmp_path, monkeypatch):
    first = run.producing(tmp_path / "first", "systems_contract")
    monkeypatch.setattr(run, "DATE", "24 Oct 2026")
    second = run.producing(tmp_path / "second", "systems_contract")
    assert first["fingerprint"] != second["fingerprint"]
    assert first["config"]["canonical_prior_firsts"] == 1925
    assert first["input_hashes"][str(run.OLD_CACHE)] == run.CACHE_SHA
    assert first["input_hashes"][str(run.word.TRAIN / "model.pt")] == run.CHECKPOINT
    assert not any("final_short_sources.json" in p or "final_long_sources.json" in p
        or "sealed_test_sources.json" in p for p in first["input_hashes"])
    run.base.verify(first)


def test_unchanged_source_only_prediction_rejects_labels():
    for field in run.FORBIDDEN:
        with pytest.raises(ValueError, match="cannot receive labels"):
            run.predict(None, None, {field: "target"}, {}, None)
