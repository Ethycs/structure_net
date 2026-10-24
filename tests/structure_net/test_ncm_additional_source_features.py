import pytest
import torch

from experiments.structure_net import ncm_additional_source_features as run


def test_lossless_features_bind_original_source_prompt_alignment_and_body(tmp_path):
    row = {'id': 'a', 'source_sha256': 'source', 'alignment_sha256': 'alignment',
        'native_prompt_sha256': 'prompt', 'source_ids': [7]}
    tensor = torch.arange(3072, dtype=torch.float32).reshape(1, 3072)
    header, blob = run.codec.encode(tensor)
    header['binding'] = run.binding(row)
    hp, bp = tmp_path / 'h.json', tmp_path / 'b.zlib'
    run.write_json(hp, header)
    bp.write_bytes(blob)
    entry = {'header_path': str(hp), 'header_sha256': run.file_hash(hp), 'payload_path': str(bp),
        'payload_sha256': run.file_hash(bp), 'source_tokens': 1}
    assert torch.equal(run.load_feature(entry, row).view(torch.uint8), tensor.view(torch.uint8))
    for patch in ({'source_sha256': 'other'}, {'alignment_sha256': 'other'}, {'native_prompt_sha256': 'other'}, {'id': 'other'}):
        with pytest.raises(ValueError, match='binding'):
            run.load_feature(entry, {**row, **patch})


def test_first_and_both_longest_inputs_covered_without_duplicate_sources():
    index = [{'id': 'a', 'native_prompt_positions': 2, 'source_positions': 3},
        {'id': 'b', 'native_prompt_positions': 9, 'source_positions': 4},
        {'id': 'c', 'native_prompt_positions': 4, 'source_positions': 10}]
    assert run.probes(index) == ['a', 'b', 'c']
    index[1]['source_positions'] = 11
    assert run.probes(index) == ['a', 'b']
