"""Real tiny Qwen token/cache parity and actual CPU Inductor execution."""
from copy import deepcopy

import pytest
import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM, StaticCache

from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_quadratic_qwen import FoldedReadout, QuadraticQwenBlock, QuadraticQwenStack
from experiments.structure_net.jspace_quadratic_runtime import CompactTokenModel, NativeTokenModel, benchmark_cpu, resident_tensor_bytes


def components(layers=2):
    torch.manual_seed(991)
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_attention_heads=2, num_key_value_heads=1,
                        num_hidden_layers=layers, vocab_size=13, max_position_embeddings=32, attention_dropout=0.)
    config._attn_implementation = "eager"
    native = Qwen2ForCausalLM(config).bfloat16().float().eval().requires_grad_(False)
    blocks = []
    for layer in native.model.layers:
        basis = torch.linalg.qr(torch.randn(8, 8)).Q[:, :4]
        affine = AffineQwenBlock(layer, basis, torch.randn(1, 8) * .1, parameter_dtype=torch.float32,
            head_projection={"rope_pairs": torch.tensor([[1]]), "value_basis": torch.eye(4)[None, :, :2]})
        block = QuadraticQwenBlock(affine, learn_metric=True)
        with torch.no_grad(): block.metric.log_metric.fill_(.2)
        blocks.append(block.eval())
    readout = FoldedReadout(native.lm_head.weight, blocks[-1].basis, blocks[-1].anchor,
                           native.model.norm.weight, config.rms_norm_eps)
    embedding = native.model.embed_tokens.weight @ blocks[0].basis
    return native, blocks, readout, embedding


def runtimes(layers=2):
    native, blocks, readout, embedding = components(layers)
    compact = CompactTokenModel(embedding, blocks, readout, native.model.rotary_emb, native.config, capacity=16)
    return NativeTokenModel(native, capacity=16), compact, native


def test_compact_runtime_matches_all_query_coordinates_and_calls_folded_readout():
    native, blocks, readout, embedding = components()
    reference_stack = QuadraticQwenStack(deepcopy(blocks)).pack(dtype=torch.float32, buffer_dtype=torch.float32, strip_unused=False)
    reference_head = deepcopy(readout).eval(); ids = torch.tensor([[1, 4, 6, 2, 3]])
    runtime = CompactTokenModel(embedding, blocks, readout, native.model.rotary_emb, native.config, capacity=16)
    positions = torch.arange(ids.shape[1])[None]
    rope = native.model.rotary_emb(torch.zeros(1, ids.shape[1], 8), positions)
    with torch.no_grad():
        z = reference_stack.forward_coordinates(torch.nn.functional.embedding(ids, embedding),
            position_ids=positions, cache_position=positions[0], position_embeddings=rope)
        expected = reference_head(z[:, -1:])[:, 0]
        actual = runtime(ids)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert all(block.basis is None and block.anchor is None for block in runtime.stack.original.blocks)
    calls = []; handle = runtime.readout.register_forward_hook(lambda *_: calls.append(True))
    with torch.no_grad(): runtime(ids)
    handle.remove(); assert calls == [True]
    assert actual.shape == (1, native.config.vocab_size)


def test_native_last_query_matches_original_full_qwen_forward():
    runtime, _, native = runtimes(); ids = torch.tensor([[1, 2, 5, 8, 4]])
    with torch.inference_mode():
        expected = native(input_ids=ids, use_cache=False).logits[:, -1]
        actual = runtime(ids)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    # The tied native embedding/readout storage must only be counted once.
    seen = {p.untyped_storage().data_ptr(): p.untyped_storage().nbytes() for p in runtime.parameters()}
    assert resident_tensor_bytes(runtime) >= sum(seen.values())


@pytest.mark.parametrize("kind", ["native", "compact"])
def test_dynamic_and_static_caches_preserve_full_kv_and_causality(kind):
    native, compact, source = runtimes(); model = native if kind == "native" else compact
    ids = torch.tensor([[1, 4, 7, 2]])
    with torch.inference_mode():
        expected = model(ids)
        for cache in (DynamicCache(), StaticCache(source.config, max_batch_size=1, max_cache_len=8, device="cpu", dtype=torch.float32)):
            # Deliberately poison unused static cells: the absolute causal mask must exclude them.
            if isinstance(cache, StaticCache):
                for key, value in zip(cache.key_cache, cache.value_cache): key.fill_(99.); value.fill_(-77.)
            prefix = model(ids[:, :3], cache_position=torch.arange(3), past_key_value=cache)
            torch.testing.assert_close(prefix, model(ids[:, :3]), rtol=3e-5, atol=3e-6)
            actual = model(ids[:, 3:], cache_position=torch.tensor([3]), past_key_value=cache)
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
            assert len(cache.key_cache) == source.config.num_hidden_layers
            assert all(t.shape[-2] >= 4 for t in cache.key_cache)
        with pytest.raises(ValueError, match="explicit absolute"):
            model(ids[:, :1], past_key_value=DynamicCache())


def test_actual_inductor_cpu_benchmark_compiles_full_graph_and_excludes_setup(monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/tmp/structure-net-inductor-cache")
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")
    native, compact, _ = runtimes(layers=1)
    prompts = {length: [torch.arange(length)[None] % 13, (torch.arange(length)[None] + 3) % 13] for length in (4, 6)}
    original_threads = torch.get_num_threads()
    report = benchmark_cpu(native, compact, prompts, lengths=(4, 6), repeats=5, warmup=1)
    assert torch.get_num_threads() == original_threads
    assert report["all_compiled_greedy_equal"] and report["threads"] == 1
    assert report["compilation"]["fullgraph"] and not report["compilation"]["eager_graph_break_fallback"]
    assert report["compilation"]["options"]["freezing"] is True
    assert all(len(graphs) >= 1 for graphs in report["compilation"]["fx_graphs"].values())
    assert max(p["maximum_absolute_logit_delta"] for p in report["parity"]) < 1e-5
    assert len(report["setup_seconds_excluded"]) == 4
    for name in ("native", "compact"):
        for mode in ("eager", "compiled"):
            for length in ("4", "6"):
                assert len(report["latencies"][name][mode][length]["samples"]) == 10


def test_cpu_benchmark_rejects_unmatched_prompt_budget():
    native, compact, _ = runtimes(layers=1)
    with pytest.raises(ValueError, match="At least2"):
        benchmark_cpu(native, compact, {4: [torch.zeros(1, 4, dtype=torch.long)]}, lengths=(4,))
