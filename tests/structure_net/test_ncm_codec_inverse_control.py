from types import SimpleNamespace

import torch
import torch.nn.functional as F

from experiments.structure_net.ncm_codec_inverse_control import blind_inverse, dictionary_inverse, group_statistics


def test_group_statistics_invert_actual_groupnorm_and_token_dictionary():
    torch.manual_seed(5)
    table, position = torch.randn(11, 8), torch.randn(6, 8)
    ids = torch.tensor([[0, 1, 3, 5, 8, 10], [9, 5, 1, 2, 0, 7]])
    embedding = table[ids]+position
    normalized = F.group_norm(embedding.transpose(1,2), 2, eps=1e-5).transpose(1,2)
    mean, scale = group_statistics(embedding, 2, 1e-5)
    assert torch.equal(dictionary_inverse(normalized, mean, scale, position, table), ids)


def test_blind_inverse_uses_only_predictions_and_keeps_its_inputs_fixed():
    torch.manual_seed(10)
    table, position = torch.randn(16, 8), torch.randn(6, 8)
    ids = torch.tensor([[1, 3, 5, 7, 9, 11]])
    class Encoder:
        wte = SimpleNamespace(weight=table)
        wpe = SimpleNamespace(weight=position)
        def __call__(self, x):
            return self.wte.weight[x]+self.wpe.weight
    vae = SimpleNamespace(encoder=Encoder(), pre_quant_norm=SimpleNamespace(num_groups=2, eps=1e-5))
    normalized = F.group_norm(vae.encoder(ids).transpose(1,2), 2, eps=1e-5).transpose(1,2)
    original = normalized.clone(); initial = ids.clone()
    actual, history = blind_inverse(vae, normalized, initial)
    assert torch.equal(actual, ids) and len(history) == 4
    assert torch.equal(normalized, original) and torch.equal(initial, ids)
    history[0].fill_(0)
    assert torch.equal(history[-1], ids)
