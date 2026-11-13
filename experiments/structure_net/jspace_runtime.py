"""Bounded greedy decoding and prefill-only interventions for the J-space pilot.

The decoder is called directly and the vocabulary projection is applied only
to its final position. This avoids allocating prompt_length x vocabulary
logits when fitting a small local experiment around a HuggingFace Qwen model.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

import torch


_LEADING = " \t\n\r\"'`“”‘’"
_WORD = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", flags=re.UNICODE)
_NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?")


def parse_first_answer(raw: str, *, ended: bool = False) -> dict:
    """Read one complete first lexical word or number, never a token prefix.

    A delimiter or true EOS completes a word. Reaching a generation cap does
    not: ``Portu`` might be the start of ``Portuguese``. Leading whitespace and
    quote characters are ignored. This deliberately does not search through
    explanations for a desired answer. Return status ``pending`` until a
    boundary is available, ``invalid`` for a nonlexical first answer, or
    ``complete`` with the entire word/number.
    """
    if not isinstance(raw, str):
        raise TypeError("Generated text must be a string")
    text = raw.lstrip(_LEADING)
    if not text:
        return {"answer": None, "status": "empty" if ended else "pending"}
    numeric = bool(re.match(r"[+-]?\d", text))
    match = (_NUMBER if numeric else _WORD).match(text)
    if match is None:
        return {"answer": None, "status": "invalid"}
    answer = match.group()
    tail = text[match.end():]
    if not tail:
        return {"answer": answer if ended else None,
                "status": "complete" if ended else "pending"}
    first = tail[0]
    if first.isalnum() or first == "_":
        return {"answer": None, "status": "invalid"}
    # At a chunk boundary, punctuation could still join a compound word or a
    # decimal. Wait for the next token instead of crediting a partial answer.
    if not ended and (tail in ("-", "'", "’") or (numeric and "." not in answer and tail == ".")):
        return {"answer": None, "status": "pending"}
    return {"answer": answer, "status": "complete"}


def _eos_ids(model, tokenizer) -> set[int]:
    for obj in (getattr(model, "generation_config", None), getattr(model, "config", None), tokenizer):
        value = getattr(obj, "eos_token_id", None)
        if value is not None:
            return {int(v) for v in value} if isinstance(value, (tuple, list, set)) else {int(value)}
    return set()


def _decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    *,
    layer: int | None = None,
    patch: Callable[[torch.Tensor], torch.Tensor] | None = None,
    observe_layers: Sequence[int] = (),
    max_new_tokens: int = 8,
) -> dict:
    """Greedily complete one answer with optional prefill-only residual patch.

    ``layer`` and ``observe_layers`` are zero-based decoder block indices.
    Hooks see block-output residuals before the final model normalization.
    ``patch`` receives the entire [1, prompt_tokens, hidden] tensor and must
    preserve its shape, device and dtype. Observations at the patch layer are
    taken *after* the patch. Cached continuation tokens are never patched or
    recorded. Hooks are always removed, including on callback/model failures.

    Supports decoder-only HF models exposing ``model.model.layers`` and
    ``model.lm_head``, such as Qwen2. No full prompt vocabulary logits are made.
    """
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("A nonempty prompt is required")
    if patch is not None and layer is None:
        raise ValueError("A patch requires an intervention layer")
    decoder = getattr(model, "model", None)
    blocks = getattr(decoder, "layers", None)
    if blocks is None or not hasattr(model, "lm_head"):
        raise TypeError("Expected a causal decoder exposing model.layers and lm_head")
    requested = set(observe_layers)
    if layer is not None:
        requested.add(layer)
    if any(type(index) is not int or not 0 <= index < len(blocks) for index in requested):
        raise ValueError("Layer indices must name actual zero-based decoder blocks")
    observe = set(observe_layers)
    device = model.get_input_embeddings().weight.device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("Tokenizer must produce one nonempty input sequence")
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
    if attention_mask.shape != input_ids.shape:
        raise ValueError("Attention mask must match the prompt token shape")

    observed: dict[int, torch.Tensor] = {}
    handles = []
    prefill = True

    def hook_for(index):
        def hook(module, inputs, output):
            if not prefill:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise TypeError("Decoder layer must return a residual tensor or tuple beginning with one")
            if patch is not None and index == layer:
                updated = patch(hidden)
                if (not isinstance(updated, torch.Tensor) or updated.shape != hidden.shape
                        or updated.dtype != hidden.dtype or updated.device != hidden.device):
                    raise ValueError("Residual patch must preserve tensor shape, dtype and device")
                hidden = updated
            if index in observe:
                if not bool(torch.isfinite(hidden[0, -1]).all()):
                    raise FloatingPointError("Nonfinite prefill observation; generation aborted")
                observed[index] = hidden[0, -1].detach().float().cpu().clone()
            if isinstance(output, tuple):
                return (hidden, *output[1:])
            return hidden
        return hook

    generated: list[int] = []
    eos_ids = _eos_ids(model, tokenizer)
    stopped_by = "token_cap"
    parsed = {"answer": None, "status": "pending"}
    was_training = model.training
    try:
        for index in sorted(requested):
            handles.append(blocks[index].register_forward_hook(hook_for(index)))
        model.eval()
        with torch.inference_mode():
            output = decoder(input_ids=input_ids, attention_mask=attention_mask,
                             use_cache=True, return_dict=True)
            prefill = False
            for step in range(max_new_tokens):
                if not bool(torch.isfinite(output.last_hidden_state).all()):
                    raise FloatingPointError("Nonfinite decoder hidden states; generation aborted")
                logits = model.lm_head(output.last_hidden_state[:, -1, :])
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Nonfinite output logits; generation aborted")
                next_id = int(logits[0].argmax().item())
                generated.append(next_id)
                raw = _decode(tokenizer, generated)
                ended = next_id in eos_ids
                parsed = parse_first_answer(raw, ended=ended)
                if ended:
                    stopped_by = "eos"
                    break
                if parsed["status"] == "complete":
                    stopped_by = "answer_boundary"
                    break
                if parsed["status"] == "invalid":
                    stopped_by = "invalid_first_answer"
                    break
                if step + 1 == max_new_tokens:
                    break
                cache = output.past_key_values
                if cache is None:
                    raise RuntimeError("Decoder did not return a cache for continuation")
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (1, 1), dtype=attention_mask.dtype, device=device)), dim=1)
                output = decoder(input_ids=torch.tensor([[next_id]], device=device),
                                 attention_mask=attention_mask, past_key_values=cache,
                                 use_cache=True, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    status = parsed["status"]
    if stopped_by == "token_cap" and status == "pending":
        status = "truncated_or_ambiguous"
    return {"answer": parsed["answer"], "answer_status": status,
            "raw": _decode(tokenizer, generated), "token_ids": generated,
            "first_token_id": generated[0], "first_token_text": _decode(tokenizer, generated[:1]),
            "observed": observed, "total_generated_tokens": len(generated),
            "stopped_by": stopped_by}
