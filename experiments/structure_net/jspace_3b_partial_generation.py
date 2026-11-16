"""Cache-free ordinary generation for a native-prefix/compact-suffix model.

This helper does not reserve GPUs, read evaluation data, or alter full-model
export admission. The calling experiment authenticates its candidate and guard.
"""
from copy import deepcopy
from contextlib import nullcontext

import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from . import jspace_3b_final_evaluate as execution
from . import jspace_qwen3b_progressive_model as builder
from .jspace_incremental_suffix import fixed_state


def training_rotary(config, device, *, guard=None):
    """Recreate nonpersistent RoPE on its original execution device, not via .to()."""
    device = execution._guard(device, guard)
    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with context:
        return Qwen2RotaryEmbedding(config=config, device=device).eval().requires_grad_(False)


def assert_fixed_state(stack, embedding, rotary, expected):
    """Check the entire UNPACKED training fingerprint, including nonpersistent RoPE."""
    actual = fixed_state(stack, embedding, rotary)
    if actual != expected:
        raise ValueError("Complete fixed geometry/readout/rotary fingerprint differs from training")
    return dict(passed=True, expected=expected, actual=actual)


class TrainingRotarySuffix(nn.Module):
    """Keep native-prefix RoPE untouched; use the trained suffix's own RoPE."""
    def __init__(self, stack, rotary):
        super().__init__()
        self.stack, self.rotary = stack, rotary
        self.attention_type = stack.attention_type

    def forward(self, hidden_states, *, position_ids=None, cache_position=None,
                position_embeddings=None, past_key_value=None, use_cache=False, **kwargs):
        if use_cache or past_key_value is not None:
            raise ValueError("Partial generation suffix requires cache-free execution")
        if position_ids is None:
            if cache_position is None:
                raise ValueError("Absolute positions required for the compact suffix")
            position_ids = cache_position.unsqueeze(0)
        return self.stack(hidden_states, position_ids=position_ids, cache_position=cache_position,
            position_embeddings=self.rotary(hidden_states, position_ids), use_cache=False, **kwargs)


def attach_suffix(model, stack, norm, *, cut, suffix_rotary=None):
    """Replace only the native suffix; retain native prefix, rotary and tied E."""
    depth = model.config.num_hidden_layers
    if (type(cut) is not int or not 0 < cut < depth or len(model.model.layers) != depth
            or tuple(stack.layer_indices) != tuple(range(cut, depth))
            or not stack.packed or not stack.stripped or stack.width != model.config.hidden_size):
        raise ValueError("A complete native model and contiguous packed partial suffix are required")
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError("Original tied embedding/readout required")
    reference = model.lm_head.weight
    if (norm.weight.shape != (model.config.hidden_size,) or norm.weight.dtype != reference.dtype
            or any(t.is_meta or t.device != reference.device
                   for module in (stack, norm) for t in (*module.parameters(), *module.buffers()))):
        raise ValueError("Materialized suffix and trained norm must share the native model device")
    suffix_rotary = deepcopy(model.model.rotary_emb) if suffix_rotary is None else suffix_rotary
    if any(t.is_meta or t.device != reference.device for t in suffix_rotary.buffers()):
        raise ValueError("Suffix rotary must be materialized on the model device")
    model.model.layers = nn.ModuleList([*list(model.model.layers[:cut]), TrainingRotarySuffix(stack, suffix_rotary)])
    model.model.norm = norm
    model.eval().requires_grad_(False)
    model.generation_config = deepcopy(model.generation_config)
    model.generation_config.use_cache = False
    model.partial_generation_scope = dict(cut=cut, compressed_layers=depth-cut,
        native_prefix_layers=cut, use_cache=False, separate_suffix_rotary=True)
    if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
        raise ValueError("The partial model retains unmaterialized tensors")
    return model


def build_partial(model_path, fit_paths, checkpoint, *, cut=22, device="cpu", guard=None,
                  expected_fixed_state=None):
    """Strict CPU reconstruction followed by a single model transfer."""
    device = execution._guard(device, guard)
    model = execution.load_native_model(model_path, device="cpu", attention="eager")
    stack, norm, embedding, cpu_rotary, inheritance = builder.build_stack(
        model_path, fit_paths, cut, checkpoint, device="cpu", trainable=False,
        checkpoint_blocks=False)
    if (inheritance["new_layer"] is not None or
            inheritance["unchanged_layers"] != list(range(cut, model.config.num_hidden_layers))):
        raise ValueError("Generation requires strict same-depth checkpoint reconstruction")
    if not torch.equal(embedding, model.lm_head.weight):
        raise ValueError("Compact builder changed the original tied embedding/readout")
    rotary = training_rotary(model.config, device, guard=guard)
    proof = None if expected_fixed_state is None else assert_fixed_state(stack, embedding, rotary, expected_fixed_state)
    packed = builder.packed_clone(stack, device="cpu")
    del stack, embedding, cpu_rotary
    # Copying these device-constructed bits through CPU does not recompute them.
    # Attach before the full transfer so discarded native suffix weights never reach CUDA.
    result = attach_suffix(model, packed, norm, cut=cut, suffix_rotary=rotary.to("cpu"))
    execution._guard(device, guard)
    result.to(device)
    result.partial_fixed_state_receipt = proof
    result.partial_generation_scope["suffix_rotary_construction_device"] = str(device)
    return result


@torch.inference_mode()
def generate(model, tokenizer, prompt_ids, *, max_new_tokens=64, position_capacity=416, guard=lambda: None):
    """Same greedy, EOS-respecting generation without a KV cache in either arm."""
    guard()
    if (not prompt_ids or type(max_new_tokens) is not int or max_new_tokens < 1 or
            len(prompt_ids)+max_new_tokens > position_capacity):
        raise ValueError("Untruncated input plus generation budget must fit the declared capacity")
    config = deepcopy(model.generation_config)
    config.do_sample = False
    config.num_beams = config.num_beam_groups = config.num_return_sequences = 1
    config.max_new_tokens = max_new_tokens
    config.use_cache = False
    config.cache_implementation = None
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.lm_head.weight.device)
    # A forward hook keeps the reservation guard live during autoregression.
    def check_reservation(_module, _args):
        guard()
        return None
    handle = model.register_forward_pre_hook(check_reservation)
    try:
        output = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=config)
    finally:
        handle.remove()
    guard()
    if not torch.equal(output[0, :len(prompt_ids)], ids[0]):
        raise ValueError("Generation changed the input token prefix")
    tokens = output[0, len(prompt_ids):].cpu().tolist()
    eos = config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
    return dict(input_ids=list(prompt_ids), output_ids=tokens,
                response=tokenizer.decode(tokens, skip_special_tokens=True),
                stop_reason="eos" if tokens and tokens[-1] in eos else "token_limit",
                generation_config=config.to_dict(), use_cache=False)
