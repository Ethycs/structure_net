from types import SimpleNamespace
from dataclasses import replace
import pytest
import torch

from experiments.structure_net.llama_exact_blas import BlasLlama, CANDIDATES
from neural_architecture_lab.llama_exact_speed_hypothesis import ExactBlasEvidence, SpeedCell, GRID


@pytest.mark.parametrize("name", CANDIDATES)
def test_backend_restored_and_weights_shared(monkeypatch, name):
    weight = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    reference = SimpleNamespace(config=None, replicas=1, rope=None, attention="flash", vectors={},
                                matrices={"projection.weight": {"kind": "dense", "cores": [weight]}})
    state = ["cublas"]
    def preference(value=None):
        if value is not None:
            state[0] = value
        return state[0]
    monkeypatch.setattr(torch.backends.cuda, "preferred_blas_library", preference)
    candidate = BlasLlama(reference, name)
    x = torch.ones(1, 2, 8)
    assert torch.equal(candidate.linear(x, "projection"), torch.bmm(x, weight.transpose(1, 2)))
    assert candidate.matrices is reference.matrices and state == ["cublas"]
    def fail(*args):
        assert state == ["cublaslt"]
        raise RuntimeError("kernel fault")
    monkeypatch.setattr(torch, "mm" if name.endswith("_mm") else "bmm", fail)
    with pytest.raises(RuntimeError, match="kernel fault"):
        candidate.linear(x, "projection")
    assert state == ["cublas"]


def test_backend_registry_uses_its_own_fixed_grid():
    cells = tuple(SpeedCell(n, length, steps, True, 1.0) for n in CANDIDATES for length, steps in GRID)
    evidence = ExactBlasEvidence("c", "a"*64, "r", "b"*64, cells)
    record = evidence.record()
    assert record["direct_experiment_count"] == 6
    assert record["eligible_candidates"] == list(CANDIDATES)
    assert record["id"] == "llama32-3b-exact-blas-iteration-v1" and not record["promoted"]
    bad = replace(cells[0], exact=False, speedup=None)
    fast = replace(cells[1], speedup=2.)
    assert not replace(evidence, cells=(bad, fast, *cells[2:])).record()["promoted"]
