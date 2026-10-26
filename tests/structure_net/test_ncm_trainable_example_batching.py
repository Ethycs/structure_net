from contextlib import nullcontext
from types import SimpleNamespace

import torch

from experiments.structure_net import ncm_trainable_example_batching as repaired
from experiments.structure_net import ncm_example_batching_codec_probe as old
from experiments.structure_net import ncm_lm1b_native_finetuning as native


class Generator(torch.nn.Module):
    levels = (1, 1)

    def __init__(self):
        super().__init__()
        self.l0_queries = torch.nn.Parameter(torch.ones(1, 1, 3))
        self.output = torch.nn.Linear(3, 2, bias=False)
        torch.nn.init.ones_(self.output.weight)

    def build_l0_query_input(self, batch, dtype=None):
        return self.l0_queries.expand(batch, -1, -1).to(dtype=dtype)

    def forward(self, value):
        return self.output(value)


def head():
    encoder = SimpleNamespace(wte=torch.nn.Embedding(102, 3))
    quantizer = SimpleNamespace(idx_to_ncm_input=lambda ids: ids.float().unsqueeze(-1).expand(-1, -1, 3))
    return SimpleNamespace(codec=SimpleNamespace(encoder=encoder, vq=quantizer),
        generator=Generator(), source_context=lambda *args: nullcontext())


def test_old_probe_loses_root_gradient_and_corrected_batch_matches_original_training():
    model = head()
    features, mask = torch.ones(2, 1, 3), torch.ones(2, 1, dtype=torch.bool)
    codes = torch.tensor([[1, 2], [3, 4]])
    old.serial_codec_logits(model, features, mask, codes)[:, 1].sum().backward()
    assert model.generator.l0_queries.grad is None
    model.generator.zero_grad(set_to_none=True)
    expected = []
    for i in range(2):
        output = native.teacher_logits(model, features[i:i+1], mask[i:i+1], codes[i:i+1])
        output[:, 1].sum().backward(); expected.append(output.detach())
    gradient = model.generator.l0_queries.grad.clone()
    assert gradient.abs().sum() > 0
    model.generator.zero_grad(set_to_none=True)
    actual = repaired.teacher_logits(model, features, mask, codes)
    actual[:, 1].sum().backward()
    assert torch.equal(actual, torch.cat(expected))
    assert torch.equal(model.generator.l0_queries.grad, gradient)
    assert model.codec.encoder.wte.weight.grad is None
