import torch

from experiments.structure_net import ncm_public_feature_store as features


def test_pilot_contains_prior_probes_and_both_length_extremes():
    entries = [{'id': str(i), 'source_positions': i+1, 'native_prompt_positions': 30-i} for i in range(20)]
    selected = features.selected_positions(entries, [str(i) for i in range(5, 14)])
    assert selected[:9] == list(range(5, 14))
    assert len(selected) == len(set(selected))
    assert {0, 1, 2, 3, 16, 17, 18, 19} <= set(selected)


def test_parallel_encoder_preserves_every_feature_byte_and_binding():
    value = torch.arange(3072, dtype=torch.float32).reshape(1, 3072)/7
    header, payload = features.encode_verified(value, {'id': 'source', 'original_body_sha256': 'body'})
    assert header['binding']['original_body_sha256'] == 'body'
    assert torch.equal(features.store.codec.decode(header, payload).view(torch.uint8), value.view(torch.uint8))
