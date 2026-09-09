import pytest

from experiments.structure_net import qwen_moba_idle_latency as m


def test_contention_guard_rejects_new_workers_and_growing_allocations():
    initial = {"external_memory_mib": {"123": 166}}
    assert m.uncontended(initial, {"external_memory_mib": {"123": 166}})
    assert m.uncontended(initial, {"external_memory_mib": {}})
    assert not m.uncontended(initial, {"external_memory_mib": {"456": 1024}})
    assert not m.uncontended(initial, {"external_memory_mib": {"123": 512}})


def test_profile_restores_attention_functions_after_failure(monkeypatch):
    class EmptyModel:
        def named_modules(self):
            return iter(())
    old_full, old_sparse = m.q.flash_call, m.q.moba_attention
    def fail(*args):
        raise RuntimeError("profile failure")
    monkeypatch.setattr(m.b, "timed", fail)
    with pytest.raises(RuntimeError, match="profile failure"):
        m.profile_components(EmptyModel(), None, m.q.FULL)
    assert m.q.flash_call is old_full
    assert m.q.moba_attention is old_sparse
