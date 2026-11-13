import pytest
import torch

from experiments.structure_net.jspace_active_closure_audit import direct_patch


def test_exact_patch_copies_retained_values_and_detects_source_drift():
    source = torch.arange(100).reshape(25, 4).to(torch.bfloat16)
    donor = -source-1
    patch, diagnostics = direct_patch(source, donor, "question")
    changed = patch(source[None, :])
    assert torch.equal(changed[0, 4:24], donor[4:24])
    assert torch.equal(changed[0, :4], source[:4])
    assert torch.equal(changed[0, -1], source[-1])
    assert diagnostics["retained_bitwise_equal_to_donor"]
    with pytest.raises(ValueError, match="boundary"):
        patch((source+1)[None, :])


def test_terminal_copy_preserves_every_other_position():
    source = torch.ones(25, 4, dtype=torch.bfloat16)
    donor = torch.zeros_like(source)
    patch, diagnostics = direct_patch(source, donor, "final")
    changed = patch(source[None, :])
    assert torch.equal(changed[0, :-1], source[:-1])
    assert torch.equal(changed[0, -1], donor[-1])
    assert diagnostics["outside_unchanged_from_source"]
