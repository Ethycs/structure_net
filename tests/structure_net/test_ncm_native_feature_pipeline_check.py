import pytest
import torch

from experiments.structure_net import ncm_native_feature_pipeline_check as run


def test_byte_comparison_detects_signed_zero_and_low_bits():
    original = torch.zeros((2, 3072), dtype=torch.float32)
    changed = original.clone()
    changed[0, 0] = -0.
    assert not run.compare(original, changed)['FP32_bytes_exact']
    changed = original.clone()
    changed.view(torch.int32)[0, 0] = 1
    assert not run.compare(original, changed)['FP32_bytes_exact']
    assert run.compare(original, original.clone())['FP32_bytes_exact']
    with pytest.raises(ValueError):
        run.compare(original, original.bfloat16())


def test_probe_selection_and_feature_degeneracy():
    rows = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
    manifest = {'a': {'source_tokens': 20}, 'b': {'source_tokens': 10}, 'c': {'source_tokens': 15}}
    assert run.select(rows, manifest) == [rows[0], rows[2]]
    values = run.stats(torch.ones(3, 3072))
    assert values['RMS'] == 1. and values['centered_RMS'] == 0. and values['adjacent_duplicate_fraction'] == 1.
