import pytest
import torch

from experiments.structure_net import ncm_lm1b_native_word_readout as r


def fixture_logits():
    return torch.tensor([[[9., 8., 7., 6.]] * 6])


def test_exact_author_policy_is_deterministic_and_penalizes_adjacent_repetition():
    logits = fixture_logits()
    before = logits.clone()
    select = r.native_selector(torch.full((4,), .25, dtype=torch.float64))
    actual = select(logits)
    assert actual.shape == (1, 6) and actual.dtype == torch.long
    assert actual[0, 0] == 0 and actual[0, 1] != 0
    assert torch.equal(actual, select(logits)) and torch.equal(logits, before)


class FakeModel:
    def generate(self, features, mask):
        return {"codes": torch.zeros((1, 641), dtype=torch.long), "raw_word_ids": fixture_logits().argmax(-1)}

    def word_logits(self, features, mask, codes):
        return fixture_logits()


def test_proxy_preserves_concepts_and_original_words_before_new_readout():
    model = r.ReadoutProxy(FakeModel(), r.native_selector(torch.full((4,), .25, dtype=torch.float64)))
    out = model.generate(None, None)
    assert torch.equal(out["codes"], model.last_codes)
    assert torch.equal(model.last_plain_words, fixture_logits().argmax(-1))
    assert not torch.equal(out["raw_word_ids"], model.last_plain_words)
    assert torch.equal(model.last_logits, fixture_logits())


def test_proxy_rejects_any_unpenalized_neural_readout_mismatch():
    class BadModel(FakeModel):
        def word_logits(self, features, mask, codes):
            return -fixture_logits()
    model = r.ReadoutProxy(BadModel(), lambda v: v.argmax(-1))
    with pytest.raises(ValueError, match="plain argmax"):
        model.generate(None, None)


def test_latest_registered_cache_identity_and_neural_baseline_are_fixed():
    r.require_baseline()
    assert r.file_hash(r.OLD_CACHE) == r.CACHE_SHA
    assert len(r.q.read(r.OLD_CACHE)) == 2600
