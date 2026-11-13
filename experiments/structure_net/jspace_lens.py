"""Bounded Jacobian-lens dictionary fitting for the StructureNet pilot.

The estimator follows Anthropic's Apache-2.0 reference implementation at
``581d398613e5602a5af361e1c34d3a92ea82ba8e`` (``jlens/fitting.py`` and hooks):
sum cotangents over valid target positions, average valid source positions,
then average prompts. Causal masking supplies the restriction target >= source.
The final block output is captured *before* the decoder's final normalization.

Selected-token fitting contracts the output cotangent with W_U[token] directly.
By linearity this computes selected rows of W_U @ mean(J), without materializing
J. It is NOT a random projection, PCA basis, or a gradient-of-probability lens.
The small corpus and token inventory are explicit restrictions of this pilot.
Sparse nonnegative decomposition is a separate operation: the linear span of
these rows must not be called the paper's J-space.

Reference estimator and hook approach adapted from Copyright 2026 Anthropic PBC,
SPDX-License-Identifier: Apache-2.0. Upstream source and license are available at
https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e
and LICENSE in that repository. No upstream installation/version upgrade is
required; this adapter supports the repository's locked Transformers 4.x.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterator, Sequence

import torch


UPSTREAM_REPOSITORY = "https://github.com/anthropics/jacobian-lens"
UPSTREAM_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
PAPER_URL = "https://www.transformer-circuits.pub/2026/workspace/index.html"


@dataclass
class LensFit:
    """FP32 CPU rows, optional full Jacobians, and serializable provenance."""

    token_rows: dict[int, torch.Tensor]
    metadata: dict[str, Any]
    jacobians: dict[int, torch.Tensor] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"token_rows": self.token_rows, "jacobians": self.jacobians,
                    "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path: str | Path) -> "LensFit":
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(**state)


def load_small_qwen(
    model_id_or_path: str,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    local_files_only: bool = True,
    revision: str | None = None,
) -> tuple[Any, Any]:
    """Load a frozen HF Qwen decoder; downloading is opt-in by the caller.

    Pass ``dtype=torch.float16, device='cuda'`` on an allocated RTX 2060.
    Eager attention makes the backward estimator independent of flash kernels.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"local_files_only": local_files_only, "trust_remote_code": False}
    if revision is not None:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path, torch_dtype=dtype, attn_implementation="eager", **kwargs
    ).to(device)
    model.eval().requires_grad_(False)
    _parts(model)
    return model, tokenizer


def _parts(model: Any) -> tuple[Any, Any, torch.Tensor]:
    decoder = getattr(model, "model", None)
    if decoder is None or not all(hasattr(decoder, n) for n in ("layers", "norm", "embed_tokens")):
        raise TypeError("expected a HuggingFace Qwen/Llama-style causal decoder")
    head = model.get_output_embeddings()
    if head is None or not hasattr(head, "weight"):
        raise TypeError("model has no linear unembedding weight")
    return decoder, decoder.layers, head.weight


def _layers(source_layers: Sequence[int], count: int) -> list[int]:
    sources = sorted(set(source_layers))
    if not sources or any(not isinstance(i, int) or i < 0 or i >= count - 1 for i in sources):
        raise ValueError(f"source_layers must be zero-based block outputs in [0, {count - 2}]")
    return sources


@contextmanager
def _record(layers: Any, indices: Sequence[int], graph_root: int | None = None) -> Iterator[dict[int, torch.Tensor]]:
    activations: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            if index == graph_root:
                hidden.requires_grad_(True)
            activations[index] = hidden
        return hook

    try:
        for index in sorted(set(indices)):
            handles.append(layers[index].register_forward_hook(make_hook(index)))
        yield activations
    finally:
        for handle in handles:
            handle.remove()


def _encode(tokenizer: Any, prompt: str, device: torch.device, max_seq_len: int) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
    return encoded["input_ids"].to(device)


def _fit(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int],
    token_ids: Sequence[int],
    full: bool,
    dim_batch: int,
    max_seq_len: int,
    skip_first: int,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> LensFit:
    if not prompts or isinstance(prompts, str) or not all(isinstance(p, str) for p in prompts):
        raise ValueError("prompts must be a nonempty sequence of strings")
    if dim_batch < 1 or skip_first < 0 or max_seq_len <= skip_first + 1:
        raise ValueError("need dim_batch > 0, skip_first >= 0 and max_seq_len > skip_first + 1")
    decoder, blocks, unembed = _parts(model)
    sources = _layers(source_layers, len(blocks))
    ids = list(token_ids)
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(i, int) or i < 0 or i >= unembed.shape[0] for i in ids):
        raise ValueError("token_ids must be unique vocabulary indices")
    # This is an inference diagnostic: no parameter optimizer or training graph.
    model.eval().requires_grad_(False)
    width = int(unembed.shape[1])
    head_rows = unembed[ids].detach().float().cpu()
    output_vectors = torch.eye(width) if full else head_rows
    n_rows = len(output_vectors)
    batch_size = min(dim_batch, n_rows)
    sums = {layer: torch.zeros(n_rows, width, dtype=torch.float32) for layer in sources}
    lengths: list[int] = []
    valid_counts: list[int] = []
    start_time = time.perf_counter()
    for prompt_index, prompt in enumerate(prompts):
        input_ids = _encode(tokenizer, prompt, decoder.embed_tokens.weight.device, max_seq_len)
        length = int(input_ids.shape[1])
        if length <= skip_first + 1:
            raise ValueError(f"prompt {prompt_index} too short: {length} tokens, need > {skip_first + 1}")
        lengths.append(length)
        valid_counts.append(length - skip_first - 1)
        with torch.enable_grad(), _record(blocks, [*sources, len(blocks) - 1], min(sources)) as captured:
            decoder(input_ids=input_ids.expand(batch_size, -1), use_cache=False)
            target = captured[len(blocks) - 1]
            source_states = [captured[layer] for layer in sources]
            for row_start in range(0, n_rows, batch_size):
                count = min(batch_size, n_rows - row_start)
                cotangent = torch.zeros_like(target)
                # Same target sum / source average as the official estimator.
                vectors = output_vectors[row_start:row_start + count].to(target)
                cotangent[:count, skip_first:length - 1, :] = vectors[:, None, :]
                grads = torch.autograd.grad(
                    target, source_states, grad_outputs=cotangent,
                    retain_graph=row_start + count < n_rows,
                )
                for layer, grad in zip(sources, grads):
                    rows = grad[:count, skip_first:length - 1, :].float().mean(dim=1).cpu()
                    if not torch.isfinite(rows).all():
                        raise FloatingPointError(f"nonfinite Jacobian at prompt {prompt_index}, layer {layer}")
                    sums[layer][row_start:row_start + count] += rows
                del grads, cotangent
        # The retained graph must not stay alive while the next prompt forwards.
        del captured, target, source_states
        if progress_callback is not None:
            progress_callback({"completed_prompts": prompt_index + 1, "total_prompts": len(prompts),
                               "elapsed_seconds": time.perf_counter() - start_time})
    means = {layer: total / len(prompts) for layer, total in sums.items()}
    rows = {layer: head_rows @ value for layer, value in means.items()} if full else means
    config = getattr(model, "config", None)
    metadata = {
        "format_version": 1,
        "method": "full_jacobian_then_token_contraction" if full else "selected_token_exact_vjp_contraction",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "paper_url": PAPER_URL,
        "model_name_or_path": str(getattr(config, "_name_or_path", "")),
        "model_revision": getattr(config, "_commit_hash", None),
        "model_dtype": str(unembed.dtype),
        "torch_version": str(torch.__version__),
        "source_layers": sources,
        "target_layer": len(blocks) - 1,
        "layer_convention": "zero_based_block_output_before_final_norm",
        "d_model": width,
        "token_ids": ids,
        "token_strings": [tokenizer.decode([i]) for i in ids],
        "n_prompts": len(prompts),
        "prompts_sha256": hashlib.sha256(json.dumps(list(prompts), ensure_ascii=False).encode()).hexdigest(),
        "sequence_lengths": lengths,
        "valid_positions_per_prompt": valid_counts,
        "dim_batch": batch_size,
        "backward_passes_per_prompt": (n_rows + batch_size - 1) // batch_size,
        "max_seq_len": max_seq_len,
        "skip_first": skip_first,
        "aggregation": "sum_valid_target_positions_mean_valid_source_positions_equal_prompt_mean",
        "dictionary_formula": "selected rows of W_U @ J_bar; no final-norm folding",
        "bounded_dictionary_pilot": len(ids) < unembed.shape[0],
        "approximation": "finite calibration corpus; truncated contexts; selected token inventory; model floating point precision",
        "not_claimed": "full J-space recovery, orthogonal subspace, global closure, causal task success, or compression",
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    return LensFit(rows, metadata, means if full else {})


def fit_token_rows(
    model: Any, tokenizer: Any, prompts: Sequence[str], *, source_layers: Sequence[int],
    token_ids: Sequence[int], dim_batch: int = 4, max_seq_len: int = 64, skip_first: int = 16,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> LensFit:
    """Fit exactly selected rows of W_U Jbar, conditional on the finite corpus."""
    return _fit(model, tokenizer, prompts, source_layers=source_layers, token_ids=token_ids,
                full=False, dim_batch=dim_batch, max_seq_len=max_seq_len, skip_first=skip_first,
                progress_callback=progress_callback)


def fit_full_jacobians(
    model: Any, tokenizer: Any, prompts: Sequence[str], *, source_layers: Sequence[int],
    token_ids: Sequence[int], dim_batch: int = 4, max_seq_len: int = 64, skip_first: int = 16,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> LensFit:
    """Reference/full estimator; also contracts the requested token rows."""
    return _fit(model, tokenizer, prompts, source_layers=source_layers, token_ids=token_ids,
                full=True, dim_batch=dim_batch, max_seq_len=max_seq_len, skip_first=skip_first,
                progress_callback=progress_callback)


def token_directions(result: LensFit, layer: int, *, normalize: bool = False) -> torch.Tensor:
    """Return token-major rows [n_tokens, width], preserving token ID order."""
    rows = result.token_rows[layer]
    if normalize:
        return rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return rows


@torch.no_grad()
def capture_residuals(
    model: Any, tokenizer: Any, prompt: str, *, source_layers: Sequence[int],
    positions: Sequence[int] | None = None, max_seq_len: int = 128,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Return FP32 CPU block outputs and CPU token IDs; no generation/cache."""
    decoder, blocks, _ = _parts(model)
    indices = sorted(set(source_layers))
    if not indices or any(i < 0 or i >= len(blocks) for i in indices):
        raise ValueError("source layer out of range")
    model.eval()
    input_ids = _encode(tokenizer, prompt, decoder.embed_tokens.weight.device, max_seq_len)
    with _record(blocks, indices) as captured:
        decoder(input_ids=input_ids, use_cache=False)
        output = {layer: (state[0] if positions is None else state[0, list(positions)]).float().cpu()
                  for layer, state in captured.items()}
    return output, input_ids.cpu()


@torch.no_grad()
def apply_full_lens(model: Any, result: LensFit, residual: torch.Tensor, layer: int) -> torch.Tensor:
    """Actual lens logits using final normalization, available for full fits only.

    Token-row inner products are useful probes but cannot in general reproduce
    the normalization of the transported vector; do not label them probabilities.
    """
    if layer not in result.jacobians:
        raise ValueError("normalized lens application requires the full Jacobian fit")
    decoder, _, unembed = _parts(model)
    transported = residual.float() @ result.jacobians[layer].to(residual.device).T
    return model.get_output_embeddings()(decoder.norm(transported.to(unembed))).float().cpu()
