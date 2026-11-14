import torch
from experiments.structure_net.jspace_bench_tensor import CONFIG
from experiments.structure_net.jspace_bench_tensor_v2 import redundancy_report


def test_cpu_counting_preserves_exact_shared_occurrences_and_input_values():
    errors=torch.tensor([[.1,.7],[.1,.1],[.1,.7],[.1,.7]])
    before=errors.clone()
    result=redundancy_report(errors,errors,CONFIG)
    assert result["total_task_occurrences"]==5
    assert result["distinct_shared_elements"]==2
    assert result["redundant_occurrences"]==3
    assert result["pairwise_shared_counts"]==[[1,1,1,1],[1,2,1,1],[1,1,1,1],[1,1,1,1]]
    assert torch.equal(errors,before)
