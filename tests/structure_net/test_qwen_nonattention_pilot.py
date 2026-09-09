import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
from experiments.structure_net.qwen_nonattention_pilot import PackedMLP


def test_packing_preserves_gate_up_order_and_original_weights():
    torch.manual_seed(7)
    original = Qwen3MLP(Qwen3Config(hidden_size=32, intermediate_size=96)).eval().requires_grad_(False)
    before = {name: p.clone() for name, p in original.named_parameters()}
    packed = PackedMLP(original)
    x = torch.randn(2, 16, 32)
    torch.testing.assert_close(packed(x), original(x), atol=1e-6, rtol=1e-5)
    for name, parameter in original.named_parameters():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
