"""Final-model evaluation building blocks, separate from progressive training.

No dataset is opened and no GPU is launched on import. The eventual runner
must authenticate accepted cut0 and an untouched confirmation set first.
Native eager is the matched quality reference; native SDPA is the primary
speed reference. All arms use BF16, the same raw prompts and generation setup.
"""
from copy import deepcopy
from contextlib import contextmanager
from collections.abc import Mapping
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache, GenerationConfig, StaticCache

from . import jspace_qwen3b_progressive_model as builder
from . import jspace_qwen_parallel_evaluate as scoring

export_packed = scoring.export_packed
reload_packed = scoring.reload_packed


def _guard(device, guard):
    device = torch.device(device)
    if device.type == "cuda" and guard is None:
        raise ValueError("CUDA evaluation requires an active reservation guard")
    if guard is not None:
        guard()
    return device


@contextmanager
def arithmetic_policy():
    """Explicit BF16 execution, with FP32 geometry free of TF32 rounding."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def attach_compact_model(stack, norm, embedding, rotary, config, generation_config):
    """Attach an entire packed suffix to an HF meta shell, retaining cache slots.

    Config.num_hidden_layers remains the native count, even though the decoder
    ModuleList has one stack. Only one actual tensor backs input E and LM head.
    Supplied modules are owned by the returned model; they are not duplicated.
    """
    cfg = deepcopy(config)
    if cfg.model_type != "qwen2" or not cfg.tie_word_embeddings:
        raise ValueError("The final3B model requires native Qwen2 with tied E")
    if (not stack.packed or not stack.stripped or
            tuple(stack.layer_indices) != tuple(range(cfg.num_hidden_layers))):
        raise ValueError("Only a complete packed, stripped cut0 stack may be attached")
    if embedding.shape != (cfg.vocab_size, cfg.hidden_size) or embedding.dtype != torch.bfloat16:
        raise ValueError("Original full-vocabulary BF16 embedding required")
    if norm.weight.shape != (cfg.hidden_size,) or norm.weight.dtype != torch.bfloat16:
        raise ValueError("Selected native-shape BF16 final norm required")
    tensors = [embedding, *(t for m in (stack, norm, rotary) for t in (*m.parameters(), *m.buffers()))]
    if any(t.device != embedding.device or t.is_meta for t in tensors):
        raise ValueError("All supplied model tensors must be materialized on one device")
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=False, attn_implementation="eager")
    model.model.layers = nn.ModuleList([stack])
    tied = nn.Parameter(embedding.detach(), requires_grad=False)
    model.model.embed_tokens.weight = tied
    model.lm_head.weight = tied
    model.model.norm, model.model.rotary_emb = norm, rotary
    model.generation_config = deepcopy(generation_config)
    model.eval().requires_grad_(False)
    stack.checkpoint_blocks = False
    if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
        raise ValueError("The compact HF shell retains unmaterialized tensors")
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError("Native embedding/readout tie was lost")
    model.final_evaluation_backend = "compact_explicit_attention_fused_pointwise"
    return model


def build_compact_model(model_path, fit_paths, selected_checkpoint, *, device="cpu", guard=None,
                        config=None, generation_config=None, rank=1024):
    """Strictly rebuild accepted cut0 on CPU, pack, then move only packed state."""
    device = _guard(device, guard)
    cfg = builder.native._config(model_path, config)
    generation = generation_config or GenerationConfig.from_pretrained(str(model_path), local_files_only=True)
    stack, norm, embedding, rotary, inheritance = builder.build_stack(
        model_path, fit_paths, 0, previous=selected_checkpoint, device="cpu", trainable=False,
        checkpoint_blocks=False, config=cfg, rank=rank)
    if inheritance["unchanged_layers"] != list(range(cfg.num_hidden_layers)) or inheritance["new_layer"] is not None:
        raise ValueError("Final evaluation must reconstruct a completed cut0 checkpoint")
    packed = builder.packed_clone(stack, device="cpu")
    del stack
    _guard(device, guard)
    result = attach_compact_model(packed.to(device), norm.to(device), embedding.to(device),
        rotary.to(device), cfg, generation)
    result.final_evaluation_inheritance = inheritance
    return result


def load_native_model(model_path, *, device="cpu", attention="eager", guard=None):
    """Local-only BF16 native eager/optimized-SDPA control, never an auto choice."""
    if attention not in ("eager", "sdpa"):
        raise ValueError("Choose explicit native eager or SDPA attention")
    device = _guard(device, guard)
    cfg = AutoConfig.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    if cfg.model_type != "qwen2" or not cfg.tie_word_embeddings:
        raise ValueError("Expected the pinned tied-E Qwen2.5 architecture")
    result = AutoModelForCausalLM.from_pretrained(str(model_path), config=cfg, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation=attention).to(device)
    result.eval().requires_grad_(False)
    if result.lm_head.weight is not result.model.embed_tokens.weight:
        raise ValueError("Native checkpoint lost the embedding/readout tie")
    result.final_evaluation_backend = "native_" + attention
    return result


def export_model_bundle(destination, model, native_source, source_fingerprint, *, position_capacity=416, guard=None):
    """Reuse the versioned standalone format: packed state plus tied E once."""
    from . import jspace_allblocks_artifact as artifact
    _guard(model.lm_head.weight.device, guard)
    if len(model.model.layers) != 1 or model.model.embed_tokens.weight is not model.lm_head.weight:
        raise ValueError("Standalone export requires the complete tied-E compact model")
    report = artifact.export_artifact(destination, model.model.layers[0], model.model.norm,
        native_source, source_fingerprint, embedding=model.lm_head.weight, position_capacity=position_capacity)
    if report["resident_tensor_bytes"] != builder.tensor_bytes(model):
        raise ValueError("Standalone payload differs from actual deployed tensor storage")
    return report


def load_model_bundle(directory, *, device="cpu", guard=None, load_tokenizer=True, expected_fingerprint=None):
    """Load solely the standalone bundle; no native weights, fits or compiler."""
    from . import jspace_allblocks_artifact as artifact
    device = _guard(device, guard)
    # Check the expected identity before allocation; the loader below verifies
    # every file hash once, including the multi-gigabyte tensor payload.
    manifest = json.loads((Path(directory)/"manifest.json").read_text())
    if expected_fingerprint is not None and manifest["source_fingerprint"] != expected_fingerprint:
        raise ValueError("Standalone bundle belongs to a different accepted checkpoint")
    model, tokenizer = artifact.load_artifact(directory, device=device, load_tokenizer=load_tokenizer)
    model.final_evaluation_backend = "compact_explicit_attention_fused_pointwise"
    return model, tokenizer


def _errors(actual, reference, atol, rtol):
    a, b = actual.float(), reference.float()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Cache comparison needs aligned finite outputs")
    delta = (a-b).abs()
    return dict(passed=bool((delta <= atol+rtol*b.abs()).all()), exact=bool(torch.equal(a, b)),
        max_abs=float(delta.max()), relative_rms=float((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-30)))


@torch.inference_mode()
def validate_cache(model, input_ids, *, guard=None, capacity=416, atol=.03125, rtol=1/128):
    """Compare full prefill and every fixed-token dynamic/static cached output.

    Future static cells are deliberately nonzero. This checks masks and cache
    slot widths; CUDA-graph parity is a separate exact-token runtime check.
    Full-prefill/tokenwise BF16 GEMM rounding is measured, not called bit exact.
    """
    device = _guard(model.lm_head.weight.device, guard)
    if (input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype != torch.long or
            input_ids.device != device or not 0 < input_ids.shape[1] <= capacity <= 416):
        raise ValueError("Cache validation needs one bounded native token sequence on the model device")
    if any(not math.isfinite(v) or v < 0 for v in (atol, rtol)):
        raise ValueError("Finite nonnegative parity tolerances required")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Cache validation requires a frozen evaluation model")
    dtype = model.lm_head.weight.dtype
    if dtype != torch.bfloat16:
        raise ValueError("Final cache validation requires BF16 execution")
    with arithmetic_policy():
        full = model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).last_hidden_state
        dynamic = DynamicCache()
        static = StaticCache(model.config, max_batch_size=1, max_cache_len=capacity, device=device, dtype=dtype)
        for value in (*static.key_cache, *static.value_cache):
            value.fill_(17)
        dynamic_rows, static_rows = [], []
        for index in range(input_ids.shape[1]):
            _guard(device, guard)
            position = torch.tensor([index], device=device)
            common = dict(input_ids=input_ids[:, index:index+1], position_ids=position[None],
                          cache_position=position, use_cache=True)
            dynamic_rows.append(model.model(**common, past_key_values=dynamic,
                attention_mask=torch.ones(1, index+1, device=device, dtype=torch.long)).last_hidden_state)
            mask = torch.full((1, 1, 1, capacity), torch.finfo(dtype).min, device=device, dtype=dtype)
            mask[..., :index+1] = 0
            static_rows.append(model.model(**common, past_key_values=static,
                attention_mask={"full_attention": mask}).last_hidden_state)
        dynamic_hidden, static_hidden = torch.cat(dynamic_rows, 1), torch.cat(static_rows, 1)
        comparisons = dict(dynamic_vs_prefill=_errors(dynamic_hidden, full, atol, rtol),
            static_vs_prefill=_errors(static_hidden, full, atol, rtol),
            static_vs_dynamic=_errors(static_hidden, dynamic_hidden, atol, rtol))
        full_logits = F.linear(full[:, -1], model.lm_head.weight)
        comparisons["dynamic_last_logits"] = _errors(F.linear(dynamic_hidden[:, -1], model.lm_head.weight), full_logits, atol, rtol)
        comparisons["static_last_logits"] = _errors(F.linear(static_hidden[:, -1], model.lm_head.weight), full_logits, atol, rtol)
    count = model.config.num_hidden_layers
    if len(dynamic.key_cache) != count or len(static.key_cache) != count:
        raise ValueError("Cache slots no longer match original native layer indices")
    widths = [k.shape[-1] for k in dynamic.key_cache]
    padding_zero = all(not torch.count_nonzero(k[:, :, :input_ids.shape[1], width:]) and
                       not torch.count_nonzero(v[:, :, :input_ids.shape[1], width:])
                       for k, v, width in zip(static.key_cache, static.value_cache, widths))
    return dict(passed=all(v["passed"] for v in comparisons.values()) and bool(padding_zero),
        comparisons=comparisons, atol=atol, rtol=rtol, cache_layers=count, dynamic_head_widths=widths,
        static_head_widths=[k.shape[-1] for k in static.key_cache], static_padding_zero=bool(padding_zero),
        capacity=capacity, input_tokens=input_ids.shape[1], future_cells_poisoned=True,
        scope="fixed-token BF16 prefill/cached parity; no task accuracy or throughput measurement")


@torch.inference_mode()
def score_model(model, cases, label_ids, *, guard=None, event=lambda **_kw: None):
    _guard(model.lm_head.weight.device, guard)
    with arithmetic_policy():
        return scoring.score_cases(model, cases, label_ids, guard=guard or (lambda: None), event=event)


@torch.inference_mode()
def score_development(model, items, label_ids, *, guard=None):
    """Replay selection arithmetic: batch4 zero-padded embeddings, four full-V rows.

    This is deliberately different from singleton choice-row confirmation
    scoring. BF16 GEMM shapes must match the trainer for the reload control.
    Input comes from raw token IDs; native boundary0 is exactly the tied lookup.
    """
    device = _guard(model.lm_head.weight.device, guard)
    if len(model.model.layers) != 1 or model.model.layers[0].layer_indices[0] != 0:
        raise ValueError("Development replay requires the complete compact cut0 model")
    cases = list(items)
    if not cases or len({c["identity"] for c in cases}) != len(cases):
        raise ValueError("Unique ordered development cases required")
    if any(c.get("split", "development") != "development" or
           not 0 < len(c["input_ids"]) <= 416 or len(c["loss_positions"]) != 4 or
           c["loss_positions"][-1] != len(c["input_ids"])-1 or
           any(not 0 <= p < len(c["input_ids"]) for p in c["loss_positions"]) for c in cases):
        raise ValueError("Only bounded development cases with original four positions are valid")
    records = []
    with arithmetic_policy():
        for start in range(0, len(cases), 4):
            _guard(device, guard)
            batch = cases[start:start+4]; lengths = [len(c["input_ids"]) for c in batch]
            length = max(lengths)
            hidden = torch.zeros(len(batch), length, model.config.hidden_size, device=device, dtype=torch.bfloat16)
            for row, case in enumerate(batch):
                ids = torch.tensor(case["input_ids"], device=device)
                hidden[row, :lengths[row]] = model.model.embed_tokens(ids)
            positions = torch.arange(length, device=device).expand(len(batch), -1)
            valid = positions < torch.tensor(lengths, device=device)[:, None]
            causal = torch.arange(length, device=device)[None, :] <= torch.arange(length, device=device)[:, None]
            mask = torch.zeros(len(batch), 1, length, length, device=device, dtype=hidden.dtype)
            mask.masked_fill_(~(causal[None] & valid[:, None])[:, None], torch.finfo(hidden.dtype).min)
            result = model.model.layers[0](hidden, attention_mask=mask, position_ids=positions,
                cache_position=positions[0], position_embeddings=model.model.rotary_emb(hidden, positions), use_cache=False)[0]
            losses = torch.tensor([c["loss_positions"] for c in batch], device=device)
            selected = result[torch.arange(len(batch), device=device)[:, None], losses]
            logits = F.linear(model.model.norm(selected), model.lm_head.weight).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite matched development replay logits")
            for row, case in enumerate(batch):
                count = len(case["choices"])
                if not 0 < count <= len(label_ids):
                    raise ValueError("Invalid development answer choices")
                choices = logits[row, -1, label_ids[:count]]
                records.append(dict(identity=case["identity"], family=case["family"], answer=case["answer"],
                    prediction=int(choices.argmax()), logits=choices.cpu().tolist()))
    return records


@torch.inference_mode()
def runtime_adapter(model, tokenizer, cases, *, seed, guard=None, repeats=5):
    """Same batch1/32-token runtime including prefill and KV copy in every arm."""
    device = _guard(model.lm_head.weight.device, guard)
    if device.type != "cuda" or (device.index or 0) != 0:
        raise ValueError("The frozen graph decoder requires leased cuda:0")
    if len(cases) != 8 or any(not 0 < len(c["input_ids"]) <= 384 for c in cases):
        raise ValueError("Use eight fixed development prompts with room for32 tokens")
    if repeats != 5:
        raise ValueError("Final runtime fixes five prespecified graph repeats")
    with arithmetic_policy():
        value = _runtime(model, tokenizer, cases, seed=seed, guard=guard, repeats=repeats)
    value.update(backend=model.final_evaluation_backend, precision="BF16", tf32=False,
        whole_model_tensor_bytes=builder.tensor_bytes(model),
        speed_scope="Warm batch1 raw-token dynamic prefill + reset/KVcopy + full-vocabulary generation32; graph setup excluded")
    value["prefill"]["mode"] = "uncaptured dynamic prefill using declared attention backend, including reset/copy into static cache"
    return value


@torch.inference_mode()
def _runtime(model, tokenizer, cases, *, seed, guard, repeats=5):
    """Frozen decode algorithm, with five repeats and explicit backend labeling."""
    from .jspace_qwen_affine import graph_decoder
    decoder = graph_decoder(model)
    setup = decoder.capture()
    inputs = [torch.tensor([c["input_ids"]], device=model.lm_head.weight.device) for c in cases]
    outputs = {"eager_static": [], "cuda_graph": [], "hf_eager": []}
    for ids in inputs:
        guard()
        static, graph = decoder.generate(ids, False), decoder.generate(ids, True)
        if not torch.equal(static, graph):
            raise ValueError("Captured and uncaptured static-cache generated tokens differ")
        outputs["eager_static"].append(static.cpu().tolist())
        outputs["cuda_graph"].append(graph.cpu().tolist())
    def hf(ids):
        return model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, use_cache=True,
            min_new_tokens=32, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)[0, -32:]
    timings = {}
    for arm in ("cuda_graph", "eager_static", "hf_eager"):
        generate = hf if arm == "hf_eager" else lambda ids: decoder.generate(ids, arm == "cuda_graph")
        # Warm every prompt shape once, not just the shortest first prompt.
        for ids in inputs:
            guard(); generate(ids)
        torch.cuda.synchronize()
        samples = []; seen = [None]*len(cases)
        for repeat in range(repeats if arm == "cuda_graph" else 1):
            order = list(range(len(cases))); random.Random(seed+repeat).shuffle(order)
            for index in order:
                guard()
                tokens, seconds = scoring._timed(lambda: generate(inputs[index]))
                if tokens.numel() != 32:
                    raise ValueError("Expected exactly32 generated tokens")
                samples.append(dict(arm=arm, repeat=repeat, identity=cases[index]["identity"],
                    input_tokens=inputs[index].numel(), tokens=32, seconds=seconds))
                if repeat == 0:
                    seen[index] = tokens.cpu().tolist()
        outputs[arm] = seen
        total = sum(v["seconds"] for v in samples)
        timings[arm] = dict(tokens_per_second=32*len(samples)/total,
                            total_tokens=32*len(samples), total_seconds=total, samples=samples)
    if outputs["cuda_graph"] != outputs["eager_static"]:
        raise ValueError("Timed graph/static generated tokens differ")
    prefill = []
    for repeat in range(repeats):
        order = list(range(len(cases))); random.Random(seed+repeat).shuffle(order)
        for index in order:
            guard()
            _, seconds = scoring._timed(lambda: decoder.prefill(inputs[index]))
            prefill.append(dict(repeat=repeat, identity=cases[index]["identity"],
                input_tokens=inputs[index].numel(), seconds=seconds))
    return dict(timings=timings, graph_capture_seconds=setup, graph_static_exact=True,
        generated_tokens=outputs,
        hf_graph_tokens_equal=[a == b for a, b in zip(outputs["hf_eager"], outputs["cuda_graph"])],
        prefill=dict(mode="uncaptured dynamic prefill plus static-cache reset/copy",
            input_tokens_per_second=sum(v["input_tokens"] for v in prefill)/sum(v["seconds"] for v in prefill), samples=prefill),
        scope="batch1; eight prespecified development prompts;32 output tokens including prefill;five graph repeats; setup excluded")


def final_gates(quality, *, native_bytes, compact_bytes, sdpa_before, sdpa_after, compact_rate,
                reload_exact, cache_valid, graph_static_exact):
    """Prespecified aggregate5pp, whole-state38%, optimized-native1.4x gates."""
    def measured(arm):
        if not isinstance(arm, Mapping) or type(arm.get("total_tokens")) is not int or arm["total_tokens"] <= 0:
            raise ValueError("Native SDPA controls need measured total_tokens and total_seconds")
        seconds = arm.get("total_seconds", 0)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Native SDPA seconds must be finite and positive")
        return arm["total_tokens"], seconds
    before_tokens, before_seconds = measured(sdpa_before)
    after_tokens, after_seconds = measured(sdpa_after)
    if before_tokens != after_tokens:
        raise ValueError("Native before/after controls must generate identical token counts")
    values = (native_bytes, compact_bytes, compact_rate)
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Positive finite measured sizes and rates required")
    overall = quality["overall"]
    if overall["count"] != 512 or not math.isfinite(overall["accuracy_drop"]):
        raise ValueError("Final quality gate requires512 paired untouched confirmation cases")
    reduction = 1-compact_bytes/native_bytes
    baseline = (before_tokens+after_tokens)/(before_seconds+after_seconds)
    speedup = compact_rate/baseline
    gates = dict(aggregate_quality=overall["accuracy_drop"] <= .05+1e-12,
        deployed_size=reduction >= .38-1e-12, optimized_native_speed=speedup >= 1.4-1e-12,
        packed_reload_exact=reload_exact is True, cached_prefill_valid=cache_valid is True,
        graph_static_tokens_exact=graph_static_exact is True)
    return dict(passed=all(gates.values()), gates=gates, measured_size_reduction=reduction,
        speedup_vs_sdpa=speedup, native_sdpa_pooled_rate=baseline,
        native_after_over_before=(after_tokens/after_seconds)/(before_tokens/before_seconds),
        thresholds=dict(maximum_accuracy_drop=.05, minimum_size_reduction=.38, minimum_speedup=1.4),
        family_accuracy_is_gate=False, family_diagnostics=quality["by_family"],
        quality_reference="matched native eager BF16", speed_reference="native SDPA BF16 before/after pooled generated tokens / elapsed seconds")
