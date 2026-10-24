from dataclasses import FrozenInstanceError

import pytest

from experiments.structure_net import ncm_lm1b_word_gradient_training as r


def test_immutable_fixed_continuation_budget_and_inclusive_schedule():
    c = r.Config()
    assert c.updates == 1024 and c.microbatch * c.accumulation == 16
    assert c.self_history_start == 0 and c.self_history_interval == 1
    assert r.rate(64) == c.max_lr and r.rate(1024) == c.min_lr
    with pytest.raises(ValueError):
        r.rate(0)
    with pytest.raises(ValueError):
        r.rate(1025)
    with pytest.raises(FrozenInstanceError):
        c.updates = 2


def test_self_history_routes_word_gradient_and_keeps_maximum_noise(monkeypatch):
    calls = []
    monkeypatch.setattr(r.objective, "backward_microbatch", lambda *args, **kw: calls.append((args[-1], kw)) or {})
    actual = r.backward_microbatch(None, None, None, None, None, None, 1, accumulation=4, self_history=True)
    assert calls == [(4097, {"accumulation": 4, "temperature": 1.})]
    assert actual["word_gradient_auxiliary_executed"] and actual["corruption_schedule_step"] == 4097


def test_other_microbatches_use_clean_noisy_objective_without_duplicate_history(monkeypatch):
    calls = []
    monkeypatch.setattr(r._original_objective, "backward_microbatch", lambda *args, **kw: calls.append((args[-1], kw)) or {})
    actual = r.backward_microbatch(None, None, None, None, None, None, 1024, accumulation=4, self_history=False)
    assert calls == [(5120, {"accumulation": 4, "self_history": False})]
    assert not actual["word_gradient_auxiliary_executed"]
