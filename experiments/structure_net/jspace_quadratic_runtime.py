"""Compact token-to-logit execution and matched FP32 CPU Inductor benchmarks.

The compact runtime embeds directly into the first coordinate space, crosses
fixed bridges, and calls FoldedReadout. It never reconstructs native-width
hidden states. Both runtimes retain full K/V but compute only the final query
and final MLP output in the last block. CPU timings include embedding/readout,
exclude compilation/warmup, and are prefill latency, not generation token/s.
"""
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

from .jspace_compact_model import rotate_half
from .jspace_quadratic_qwen import FoldedReadout, QuadraticQwenBlock, QuadraticQwenStack


def resident_tensor_bytes(model):
    """Count owned tensor storage once, including tied/native embedding weights."""
    storages = {}
    for value in (*model.parameters(), *model.buffers()):
        storage = value.untyped_storage()
        storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


class _TokenRuntime(nn.Module):
    def _initialize_rotary(self, rotary, reference, native_width, capacity):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Positive static rotary capacity required")
        self.capacity = capacity
        with torch.no_grad():
            dummy = torch.zeros(1, capacity, native_width, device=reference.device, dtype=reference.dtype)
            positions = torch.arange(capacity, device=reference.device)[None]
            cos, sin = rotary(dummy, positions)
        self.register_buffer("rotary_cos", cos[0].detach().contiguous())
        self.register_buffer("rotary_sin", sin[0].detach().contiguous())

    def _positions_and_rope(self, input_ids, cache_position, past_key_value):
        if input_ids.ndim != 2 or not input_ids.shape[1] or input_ids.dtype != torch.long:
            raise ValueError("Nonempty LongTensor[batch,tokens] input required")
        if input_ids.shape[1] > self.capacity:
            raise ValueError("Input exceeds precomputed rotary capacity")
        if cache_position is None:
            if past_key_value is not None:
                raise ValueError("Cached execution requires explicit absolute cache_position")
            cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
        if cache_position.ndim != 1 or len(cache_position) != input_ids.shape[1] or cache_position.dtype != torch.long:
            raise ValueError("cache_position must be a LongTensor matching the input length")
        positions = cache_position[None].expand(input_ids.shape[0], -1)
        cos = self.rotary_cos.index_select(0, cache_position)[None].expand(input_ids.shape[0], -1, -1)
        sin = self.rotary_sin.index_select(0, cache_position)[None].expand(input_ids.shape[0], -1, -1)
        return cache_position, positions, (cos, sin)


class CompactTokenModel(_TokenRuntime):
    """Own supplied modules; construct folded embedding/readout before this call.

    Ein is [vocabulary, first_rank]. Blocks must be the complete contiguous
    sequence of the supplied config. Packing folds learned metrics/offsets;
    all native-width basis/anchor buffers are then removed as unused runtime
    state. The fixed bridges and FoldedReadout's exact geometry remain.
    """
    def __init__(self, embedding_weight_Ein, blocks, readout, rotary, config, *, capacity=416):
        super().__init__(); blocks = list(blocks)
        if (not blocks or len(blocks) != config.num_hidden_layers or
                any(not isinstance(block, QuadraticQwenBlock) or block.layer_idx != i for i, block in enumerate(blocks))):
            raise ValueError("A complete ordered list of compact decoder blocks is required")
        if (not isinstance(readout, FoldedReadout) or embedding_weight_Ein.ndim != 2 or
                embedding_weight_Ein.shape != (config.vocab_size, blocks[0].rank) or
                readout.weight.shape != (config.vocab_size, blocks[-1].rank)):
            raise ValueError("Folded embedding/readout dimensions disagree with compact blocks")
        reference = embedding_weight_Ein
        if any(next(block.parameters()).dtype != reference.dtype or next(block.parameters()).device != reference.device for block in blocks):
            raise ValueError("Compact embedding and blocks require matching parameter dtype/device")
        if readout.weight.dtype != reference.dtype or readout.weight.device != reference.device:
            raise ValueError("Compact readout must match parameter dtype/device")
        self.config = deepcopy(config)
        self.embedding = nn.Embedding.from_pretrained(reference.detach(), freeze=True)
        self.stack = QuadraticQwenStack(blocks, geometry_mode="fp32")
        self.stack.pack(dtype=reference.dtype, buffer_dtype=reference.dtype, strip_unused=True)
        self.readout = readout
        self._initialize_rotary(rotary, reference, config.hidden_size, capacity)
        for block in self.stack.original.blocks:
            block.basis = None; block.anchor = None
        self.runtime_scope = "compact coordinates throughout; folded metric and final RMS/readout; final-query-only last block; full KV"
        self.eval().requires_grad_(False)

    def forward(self, input_ids, cache_position=None, past_key_value=None):
        absolute, positions, rope = self._positions_and_rope(input_ids, cache_position, past_key_value)
        z = self.embedding(input_ids)
        last = torch.arange(input_ids.shape[1] - 1, input_ids.shape[1], device=input_ids.device)
        z = self.stack.forward_coordinates(z, position_ids=positions, cache_position=absolute,
            past_key_value=past_key_value, use_cache=past_key_value is not None, position_embeddings=rope, query_indices=last)
        return self.readout(z, position_ids=positions[:, -1:])[:, 0]


class NativeTokenModel(_TokenRuntime):
    """Native Qwen weights, with the same final-query-only optimization.

    No casts/quantization are introduced here. For the primary CPU arm the
    caller supplies FP32 storage of the same BF16-rounded native weights.
    Native block/RMS/SwiGLU execution is retained; final attention uses the
    native eager operations with all final K/V projections present.
    """
    def __init__(self, model, *, capacity=416):
        super().__init__(); self.config = deepcopy(model.config)
        if model.config.model_type != "qwen2" or len(model.model.layers) != model.config.num_hidden_layers:
            raise ValueError("Complete native Qwen2 decoder required")
        self.embedding = model.model.embed_tokens; self.layers = model.model.layers
        self.norm, self.readout = model.model.norm, model.lm_head
        self._initialize_rotary(model.model.rotary_emb, self.embedding.weight, model.config.hidden_size, capacity)
        self.runtime_scope = "native transformer; final-query-only last block; full KV; precomputed identical rotary tables"
        self.eval().requires_grad_(False)

    def _last_block(self, hidden, block, mask, positions, rope, cache):
        attention = block.self_attn; batch, length, _ = hidden.shape
        normalized = block.input_layernorm(hidden)
        dim = attention.head_dim
        q = attention.q_proj(normalized[:, -1:]).view(batch, 1, self.config.num_attention_heads, dim).transpose(1, 2)
        k = attention.k_proj(normalized).view(batch, length, self.config.num_key_value_heads, dim).transpose(1, 2)
        v = attention.v_proj(normalized).view(batch, length, self.config.num_key_value_heads, dim).transpose(1, 2)
        cos, sin = rope
        q = q * cos[:, None, -1:] + rotate_half(q) * sin[:, None, -1:]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        if cache is not None:
            k, v = cache.update(k, v, attention.layer_idx, {"cache_position": positions, "cos": cos, "sin": sin})
        group = self.config.num_attention_heads // self.config.num_key_value_heads
        if group != 1: k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
        scores = q @ k.transpose(-1, -2) * attention.scaling
        scores = scores + mask[..., -1:, :k.shape[-2]]
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        mixed = (probabilities @ v).transpose(1, 2).reshape(batch, 1, -1)
        hidden = hidden[:, -1:] + attention.o_proj(mixed)
        return hidden + block.mlp(block.post_attention_layernorm(hidden))

    def forward(self, input_ids, cache_position=None, past_key_value=None):
        absolute, positions, rope = self._positions_and_rope(input_ids, cache_position, past_key_value)
        hidden = self.embedding(input_ids)
        keys = input_ids.shape[1]
        if past_key_value is not None:
            maximum = past_key_value.get_max_cache_shape()
            keys = maximum if maximum is not None else past_key_value.get_seq_length() + input_ids.shape[1]
        allowed = torch.arange(keys, device=input_ids.device)[None, :] <= absolute[:, None]
        mask = torch.zeros((input_ids.shape[1], keys), device=hidden.device, dtype=hidden.dtype)
        mask = mask.masked_fill(~allowed, torch.finfo(hidden.dtype).min)[None, None]
        for block in self.layers[:-1]:
            hidden = block(hidden, attention_mask=mask, position_ids=positions, cache_position=absolute,
                position_embeddings=rope, past_key_value=past_key_value, use_cache=past_key_value is not None)[0]
        hidden = self._last_block(hidden, self.layers[-1], mask, absolute, rope, past_key_value)
        return self.readout(self.norm(hidden))[:, 0]


class _RecordingInductor:
    def __init__(self): self.graphs = []

    def __call__(self, graph, inputs, options=None):
        source = graph.code
        self.graphs.append({"code": source, "sha256": hashlib.sha256(source.encode()).hexdigest(),
                            "nodes": len(list(graph.graph.nodes)), "input_shapes": [list(x.shape) for x in inputs if isinstance(x, torch.Tensor)]})
        return torch._inductor.compile(graph, inputs, options=options)


@contextmanager
def _one_cpu_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    try: yield
    finally: torch.set_num_threads(old)


@torch.inference_mode()
def benchmark_cpu(native, compact, prompts_by_length, *, repeats=5, warmup=2, lengths=(128, 256), progress=None):
    """Paired static prefill+last-logit timing; all compilation is excluded.

    fullgraph=True forbids eager graph-break fallback. Captured FX graphs are
    retained; Inductor may use external optimized GEMM kernels within a graph.
    This measures inference on supplied inputs, not held-out task quality.
    """
    if repeats < 5 or warmup < 1 or set(prompts_by_length) != set(lengths):
        raise ValueError("Require at least5 repeats, warmup, and exactly the requested static lengths")
    models = {"native": native, "compact": compact}
    for name, model in models.items():
        if model.training or any(t.device.type != "cpu" or t.dtype != torch.float32 for t in model.parameters()):
            raise ValueError(name + " must be an evaluation model with CPU FP32 parameters")
    for length in lengths:
        prompts = prompts_by_length[length]
        if len(prompts) < 2 or any(p.device.type != "cpu" or p.dtype != torch.long or p.shape != (1, length) for p in prompts):
            raise ValueError("At least2 CPU LongTensor[1,length] prompts required for each length")
    backends = {name: _RecordingInductor() for name in models}
    options = {"freezing": True, "compile_threads": 1}
    timings = defaultdict(list); parity = []; setup = {}; warmed = {}
    with _one_cpu_thread():
        compiled = {name: torch.compile(model, backend=backends[name], fullgraph=True, dynamic=False, options=options)
                    for name, model in models.items()}
        for length in lengths:
            prompts = prompts_by_length[length]
            for name, model in models.items():
                started = time.perf_counter()
                for index, prompt in enumerate(prompts):
                    eager = model(prompt); result = compiled[name](prompt)
                    if not torch.isfinite(eager).all() or not torch.isfinite(result).all(): raise FloatingPointError("Nonfinite benchmark logits")
                    delta = result.float() - eager.float()
                    parity.append({"model": name, "length": length, "prompt": index,
                        "maximum_absolute_logit_delta": float(delta.abs().max()),
                        "logit_rmse": float(delta.square().mean().sqrt()),
                        "greedy_equal": bool(torch.equal(result.argmax(-1), eager.argmax(-1)))})
                    for _ in range(warmup): model(prompt); compiled[name](prompt)
                setup[f"{name}_{length}"] = time.perf_counter() - started
                warmed[name, length] = len(backends[name].graphs)
            for mode in ("eager", "compiled"):
                functions = models if mode == "eager" else compiled
                for repeat in range(repeats):
                    order = tuple(models) if repeat % 2 == 0 else tuple(reversed(models))
                    for index, prompt in enumerate(prompts):
                        for name in order:
                            start = time.perf_counter_ns(); functions[name](prompt); elapsed = (time.perf_counter_ns() - start) / 1e9
                            timings[name, mode, length].append({"seconds": elapsed, "repeat": repeat, "prompt": index})
            if any(len(backends[name].graphs) != warmed[name, length] for name in models):
                raise RuntimeError("Compilation occurred during timing")
            if progress is not None: progress({"phase": "cpu_length_complete", "length": length})
        if any(not backend.graphs for backend in backends.values()): raise RuntimeError("Inductor did not compile the model")
        # A new graph after warmup would put compilation inside measured latency.
        for name in models:
            if len(backends[name].graphs) != warmed[name, lengths[-1]]: raise RuntimeError("Compilation occurred during timing")
    latencies = {name: {mode: {str(length): {"median_seconds": statistics.median(x["seconds"] for x in timings[name, mode, length]),
                    "samples": timings[name, mode, length]} for length in lengths} for mode in ("eager", "compiled")} for name in models}
    return {"scope": "CPU FP32 batch1 static prefill through full-vocabulary last-token logits; final-query-only in both models",
        "threads": 1, "repeats": repeats, "lengths": list(lengths), "setup_seconds_excluded": setup,
        "resident_tensor_bytes": {name: resident_tensor_bytes(model) for name, model in models.items()},
        "resident_scope": "deduplicated model tensor storage; excludes compiler caches and temporary workspaces",
        "latencies": latencies, "compiled_speedup": {str(length): latencies["native"]["compiled"][str(length)]["median_seconds"] /
            latencies["compact"]["compiled"][str(length)]["median_seconds"] for length in lengths},
        "parity": parity, "all_compiled_greedy_equal": all(p["greedy_equal"] for p in parity),
        "compilation": {"backend": "inductor", "fullgraph": True, "dynamic": False, "options": options,
            "eager_graph_break_fallback": False, "fx_graphs": {name: backend.graphs for name, backend in backends.items()}}}
