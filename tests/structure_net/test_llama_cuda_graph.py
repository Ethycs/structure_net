import pytest
import torch
from experiments.structure_net.llama_cuda_graph import bitwise_tree, clone_tree, CapturedCall


def test_bitwise_controls_and_owned_outputs():
    source = (torch.tensor([0., -0.]), [(torch.arange(4), torch.tensor([3.]))], None)
    cloned = clone_tree(source)
    assert bitwise_tree(cloned, source)
    assert cloned[0].data_ptr() != source[0].data_ptr()
    cloned[0][1] = 0.
    assert not bitwise_tree(cloned, source)
    assert not bitwise_tree(torch.ones(2), torch.ones(2, dtype=torch.float16))


def test_cuda_required():
    with pytest.raises(ValueError, match="CUDA"):
        CapturedCall(lambda x: x + 1, torch.ones(1))


def test_timing_order_really_alternates(monkeypatch):
    from experiments.structure_net import run_llama_exact_deployment as study
    calls = []
    monkeypatch.setattr(study.common, "timed", lambda fn: ({"wall_ms": 1., "cuda_ms": 1.}, fn()))
    arms = {name: lambda name=name: calls.append(name) for name in ("eager", "cuda_graph")}
    rows, orders = study.paired(arms, 10)
    assert orders == [["eager", "cuda_graph"] if i % 2 == 0 else ["cuda_graph", "eager"] for i in range(10)]
    assert calls[2:] == [name for order in orders for name in order]
    assert all(len(row["samples"]) == 10 for row in rows.values())


def test_control_inputs_differ_after_selecting_the_actual_length():
    from experiments.structure_net.run_llama_exact_deployment import benchmark_pair
    rows = [[1, 2, 3, 4], [1, 2, 7, 8]]
    assert benchmark_pair(rows, 2) == [[1, 2], [7, 8]]
    assert benchmark_pair(rows, 4) == rows
    with pytest.raises(ValueError, match="must differ"):
        benchmark_pair([[1, 2], [1, 2]], 2)
