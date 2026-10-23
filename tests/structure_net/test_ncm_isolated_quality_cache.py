import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.structure_net.ncm_isolated_quality_cache_v2 import roundtrip_logits
from experiments.structure_net.ncm_native_query_cache import NativeQueryBackbone
from experiments.structure_net.ncm_native_query_distillation import cpu_prefix


def test_real_native_query_contract_after_source_cache_serialization(tmp_path):
    torch.manual_seed(1)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, tie_word_embeddings=True, attn_implementation="sdpa")
    backbone = NativeQueryBackbone(LlamaForCausalLM(config).eval(), rank=2)
    memory = backbone.source_cache(torch.tensor([[1, 3, 4, 5, 6]]), torch.ones(1, 5, dtype=torch.long))
    targets = torch.tensor([7, 8, 2, -100])
    path = tmp_path / "cache.pt"
    torch.save({"prefix": cpu_prefix(memory, 0), "targets": targets}, path)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    result = roundtrip_logits(backbone, memory, restored, targets, 2)
    assert result == {"query_shape": [1, 4, 16], "logits_shape": [1, 4, 32], "exact": True}
