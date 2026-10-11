import torch
from transformers import LlamaConfig, LlamaForCausalLM
from structure_net.components.layers.model_major_tensor import Tensorization, decompose
from experiments.structure_net.llama_tensorized_model import TensorizedLlama
from experiments.structure_net.llama_lossless_residual import ExactResidualLlama, bits, exact_output_correction, matrix_residual


def test_output_correction_repairs_cancellation_overflow_and_signed_zero():
    original = torch.tensor([0., -0., 1e-7, -1e-7, 65504., -65504., 2., -2.], dtype=torch.float16)
    predicted = torch.tensor([65504., 65504., -65504., 65504., -65504., 65504., float('inf'), float('nan')], dtype=torch.float16)
    corrected = exact_output_correction(predicted, original)
    assert torch.equal(bits(corrected), bits(original))


def test_matrix_residual_preserves_rare_small_values_and_signed_zero(tmp_path):
    original = torch.tensor([[0., -0., 1e-7, -1e-7], [1., -1., 65504., -65504.]], dtype=torch.float16)
    prediction = torch.full(original.shape, 1e8)
    residual, indices, repairs, recovered = matrix_residual(original, prediction)
    assert len(indices) > 0
    checkpoint = tmp_path/'residual.pt'
    torch.save({'residual': residual, 'indices': indices, 'repairs': repairs}, checkpoint)
    loaded = torch.load(checkpoint, weights_only=True)
    decoded = (prediction+loaded['residual']).half()
    decoded.flatten()[loaded['indices'].long()] = loaded['repairs']
    assert torch.equal(bits(decoded), bits(original))
    assert torch.equal(bits(recovered), bits(original))


def test_exact_complete_residual_network_preserves_both_models_and_cache():
    torch.manual_seed(712)
    torch.set_num_threads(1)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=True)
    hf = LlamaForCausalLM(config).eval().requires_grad_(False)
    matrices, vectors, factor_matrices = {}, {}, {}
    for name, value in hf.state_dict().items():
        if name == 'lm_head.weight':
            continue
        if value.ndim == 2:
            matrices[name] = {'kind':'dense', 'shape':list(value.shape), 'cores':[value]}
            factor_matrices[name] = decompose(value, Tensorization('lowrank', 'svd2', rank=2))
        else:
            vectors[name] = value
    original = {'config':config.to_dict(), 'matrices':matrices, 'vectors':vectors}
    factors = {'matrices':factor_matrices}
    reference = TensorizedLlama(original, 2, 'cpu', 'sdpa')
    exact = ExactResidualLlama(original, factors, 2, 'cpu', 'sdpa')
    ids = torch.randint(0,32,(2,7))
    with torch.inference_mode():
        expected, cache = reference(ids, use_cache=True, all_logits=True)
        actual, actual_cache = exact(ids, use_cache=True, all_logits=True)
        assert torch.equal(bits(actual), bits(expected))
        exact.factors['model.layers.0.self_attn.q_proj.weight']['cores'][0][0].mul_(123)
        corrected_after_factor_change,_ = exact(ids,all_logits=True)
        assert torch.equal(bits(corrected_after_factor_change),bits(expected))
        for a, b in zip(actual_cache, cache):
            assert all(torch.equal(bits(x), bits(y)) for x,y in zip(a,b))
        extra = torch.randint(0,32,(2,1))
        expected, _ = reference(extra, past=cache)
        actual, _ = exact(extra, past=actual_cache)
        assert torch.equal(bits(actual), bits(expected))
        for i in range(2):
            separate, _ = exact.slice(i)(ids[i:i+1])
            baseline, _ = reference.slice(i)(ids[i:i+1])
            assert torch.equal(bits(separate), bits(baseline))
    assert exact.storage_bytes() > reference.storage_bytes()
