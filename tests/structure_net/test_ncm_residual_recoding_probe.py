from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net.ncm_residual_recoding_probe import feature_inputs, recode, study


def quantizer():
    first = torch.nn.Embedding.from_pretrained(torch.tensor([[0.], [2.]]))
    second = torch.nn.Embedding.from_pretrained(torch.tensor([[-2.], [0.], [2.]]))
    return SimpleNamespace(levels=[1, 1], embed_dim=1, embeddings=[first, second],
        _down_for_quant=lambda x, i, length: x,
        _up_to_full=lambda x, i, length: x,
        _apply_quant_resi=lambda x, i: x,
        _distances=lambda x, w: (x[:, None, :] - w[None, :, :]).square().sum(-1))


def test_recomputed_suffix_compensates_wrong_prefix_without_changing_it():
    vq = quantizer()
    z = torch.tensor([[[2.]]])
    gold = recode(vq, z, torch.empty(1, 0, dtype=torch.long), 0)
    assert gold.tolist() == [[1, 1]]  # 2 + 0
    wrong_prefix = torch.tensor([[0]])
    corrected = recode(vq, z, wrong_prefix, 1)
    assert corrected.tolist() == [[0, 2]]  # 0 + 2; original suffix gives 0
    assert torch.equal(z, torch.tensor([[[2.]]]))
    assert wrong_prefix.tolist() == [[0]]
    assert torch.equal(recode(vq, z, gold[:, :1], 1), gold)
    assert torch.equal(recode(vq, z, corrected, 2), corrected)


@pytest.mark.parametrize('prefix,count', [
    (torch.tensor([[0, 1]]), 1), (torch.tensor([[0]]), 2),
    (torch.tensor([[2]]), 1), (torch.tensor([[0.]]), 1),
    (torch.tensor([[0]]), True), (torch.empty(1, 0, dtype=torch.long), -1),
])
def test_malformed_or_out_of_range_prefix_is_rejected(prefix, count):
    with pytest.raises(ValueError):
        recode(quantizer(), torch.tensor([[[2.]]]), prefix, count)


def test_rejects_nonfinite_target_and_requires_full_fp32_latent():
    for z in (torch.tensor([[[float('nan')]]]), torch.tensor([[[2.]]]).half(), torch.ones(1, 2, 1)):
        with pytest.raises(ValueError):
            recode(quantizer(), z, torch.empty(1, 0, dtype=torch.long), 0)


def test_streaming_features_bind_metadata_header_and_compressed_payload(tmp_path):
    paths = [tmp_path / n for n in ('source.json', 'header.json', 'payload.zlib')]
    for p in paths:
        p.write_bytes(p.name.encode())
    index = [{'id': 'case', 'path': str(paths[0]), 'sha256': study.file_hash(paths[0])}]
    manifest = {'case': {'header_path': str(paths[1]), 'header_sha256': study.file_hash(paths[1]),
                         'payload_path': str(paths[2]), 'payload_sha256': study.file_hash(paths[2])}}
    assert feature_inputs([{'id': 'case'}], index, manifest) == paths
    paths[2].write_bytes(b'changed payload')
    with pytest.raises(ValueError, match='feature identity'):
        feature_inputs([{'id': 'case'}], index, manifest)
