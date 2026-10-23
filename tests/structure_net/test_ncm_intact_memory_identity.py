import torch
from experiments.structure_net.ncm_intact_memory_identity import IdentityMemoryCorrection


def test_zero_inference_returns_same_tensor_but_autograd_and_nonzero_run_correction():
    torch.manual_seed(42)
    module = IdentityMemoryCorrection(24, 32, memory_width=16, heads=2)
    source = torch.randn(2, 5, 3, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    hidden = torch.randn(2, 7, 24)[:, -1:]
    assert not hidden.is_contiguous()
    with torch.no_grad():
        memory = module.prepare(source, mask)
        actual = module(hidden, memory)
        assert actual is hidden and actual.stride() == hidden.stride()
    memory = module.prepare(source, mask)
    module(hidden, memory).sum().backward()
    assert module.output.weight.grad.abs().sum() > 0
    with torch.no_grad():
        module.output.weight.fill_(.1)
        changed = module(hidden, module.prepare(source, mask))
        assert changed is not hidden and not torch.equal(changed, hidden)
