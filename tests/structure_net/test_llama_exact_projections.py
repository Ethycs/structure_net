import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.structure_net.llama_tensorized_model import TensorizedLlama, generate
from experiments.structure_net.llama_exact_projections import ProjectionLlama, CANDIDATES
from experiments.structure_net.llama_cuda_graph import bitwise_tree
from experiments.structure_net.run_llama_exact_speed import generation_trace


@pytest.fixture
def reference():
    torch.manual_seed(779)
    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, eos_token_id=2)
    return TensorizedLlama.from_hf(LlamaForCausalLM(cfg).eval(), device="cpu")


@pytest.mark.parametrize("variant", [n for n in CANDIDATES if not n.startswith("parallel")])
def test_packing_preserves_parameters_and_projection_order(reference, variant):
    candidate = ProjectionLlama(reference, variant)
    assert candidate.storage_bytes() == reference.storage_bytes()
    assert candidate.packing_verified
    x = torch.arange(48, dtype=torch.float32).reshape(1, 3, 16) / 47
    for suffix in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"):
        name = "model.layers.0." + suffix
        # CPU GEMM tiling may differ; packing semantics are independently exact.
        torch.testing.assert_close(candidate.linear(x, name), reference.linear(x, name))
    assert not candidate.pending
    for key, names, sizes in candidate.groups.values():
        for name, part in zip(names, candidate.matrices[key]["cores"][0].split(sizes, dim=1)):
            assert bitwise_tree(part, reference.matrices[name + ".weight"]["cores"][0])


def test_trace_keeps_all_logits_and_matches_generation(reference):
    ids = torch.tensor([[3, 7, 2, 8]])
    with torch.inference_mode():
        tokens, logits, cache = generation_trace(reference, ids, 4)
        assert bitwise_tree(tokens, generate(reference, ids, steps=4, stop_eos=False))
        assert len(logits) == 4 and len(cache) == 2
        assert cache[0][0].shape[2] == ids.shape[1] + 3
        assert bitwise_tree(tokens, torch.stack([v[:, -1].argmax(-1) for v in logits], dim=1))


def test_reject_unsupported_layout(reference):
    with pytest.raises(ValueError, match="require CUDA"):
        ProjectionLlama(reference, "parallel2")
    with pytest.raises(ValueError, match="known variant"):
        ProjectionLlama(reference, "guess")
    reference.replicas = 2
    with pytest.raises(ValueError, match="exactly one"):
        ProjectionLlama(reference, "mm")


def test_shared_packing_preserves_reference_shapes_strides_and_bits(reference):
    before = {name: spec["cores"][0].clone() for name, spec in reference.matrices.items()}
    candidate = ProjectionLlama(reference, "packed_bmm", share_packed_reference=True)
    for key, names, _ in candidate.groups.values():
        packed = candidate.matrices[key]["cores"][0]
        for name in names:
            view = reference.matrices[name + ".weight"]["cores"][0]
            original = before[name + ".weight"]
            assert view.stride() == original.stride()
            assert bitwise_tree(view, original)
            assert view.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
