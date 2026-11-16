"""Strict compact construction, bounded native-prefix execution and path parity.

GPU admission belongs to the caller. Model-path and checkpoint inputs must be
bound by its frozen study manifest. Parity is a pre-optimizer systems check:
it sequentially places two execution paths on the card, preserving actual
parameters, geometry and the single original vocabulary allocation.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import gc
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, GenerationConfig

from . import jspace_3b_final_evaluate as execution
from . import jspace_3b_partial_generation as partial
from . import jspace_3b_progressive_stage as progressive
from . import jspace_3b_joint as joint
from . import jspace_qwen3_train as offload
from . import jspace_qwen3b_progressive_model as builder


def build_student(arm, source_plan, checkpoint, device, guard):
    """Strictly reconstruct the supplied saved suffix without expanding it."""
    device = execution._guard(device, guard)
    p = json.loads(Path(source_plan).read_text()) if isinstance(source_plan, (str, Path)) else source_plan
    progressive.source.check_plan(p)
    if arm in ("progressive", "progressive16"):
        if p.get("schema") != progressive.SCHEMA:
            raise ValueError("Progressive reconstruction requires its frozen stage plan")
        state = builder.load_previous(checkpoint)
        stack, norm, embedding, _stale_rotary, proof = progressive._build(p, state, "cpu", True)
        for key, value in (("stack", stack), ("norm", norm)):
            if not progressive.old.state_equal(value.state_dict(), state[key]):
                raise ValueError("Progressive strict reconstruction changed saved weights")
        del state, _stale_rotary
        rotary = partial.training_rotary(builder.native._config(p["model"]["path"]), device, guard=guard)
        stack, norm, embedding = stack.to(device), norm.to(device), embedding.to(device)
    elif arm in ("all36", "all_at_once"):
        if p.get("schema") != joint.SCHEMA:
            raise ValueError("All36 reconstruction requires its frozen all-block plan")
        stack, norm, embedding, rotary, proof = joint.build(p, checkpoint, device, guard or (lambda: None))
    else:
        raise ValueError("Unknown response-recovery arm")
    if proof.get("new_layer") is not None or proof.get("unchanged_layers") != list(stack.layer_indices):
        raise ValueError("Recovery must strictly reconstruct the complete saved suffix")
    proof = dict(proof, checkpoint=str(Path(checkpoint).resolve()),
                 checkpoint_sha256=offload.sha(checkpoint), source_fingerprint=p["fingerprint"],
                 cut=stack.layer_indices[0], strict_state_preserved=True)
    return stack, norm, embedding, rotary, proof


@torch.no_grad()
def old_mc_evaluate(arm, source_plan, response_model, guard=None):
    """Replay all 256 exposed MC DEV cases through the actual unpacked suffix.

    This is a retention gate on already exposed development material. It never
    opens the historical audit or final confirmation sets, and it cannot admit
    a checkpoint that fails the separate complete-response/generation gates.
    """
    p = json.loads(Path(source_plan).read_text()) if isinstance(source_plan, (str, Path)) else source_plan
    progressive.source.check_plan(p)
    if arm in ("progressive", "progressive16") and p.get("schema") == progressive.SCHEMA:
        dataset = progressive._dataset(p)
    elif arm in ("all36", "all_at_once") and p.get("schema") == joint.SCHEMA:
        dataset = joint.dataset(p)
    else:
        raise ValueError("Arm and frozen MC source plan disagree")
    was_training = response_model.training
    try:
        response_model.eval()
        cases = [c for c in dataset.items if c["split"] == "development"]
        if len(cases) != 256 or p["native_correct"] != 195:
            raise ValueError("Expected the exposed 256-case development benchmark and original native reference")
        metrics, records = joint.training.score_unpacked(response_model.stack, response_model.norm,
            response_model.embedding, response_model.rotary, dataset, p["label_ids"],
            guard or response_model.guard or (lambda: None))
        eligibility = joint.broad.eligibility(records, p["native_records"], cases)
        return dict(metrics=metrics, records=records, native_correct=p["native_correct"],
                    native_count=len(cases), eligibility=eligibility,
                    scope="Reused MC DEV retention only; separate generation gate is mandatory")
    finally:
        dataset.close()
        response_model.train(was_training)


def _shell(model_path, embedding, device, guard, *, config=None):
    """An HF native shell, with no native layer or normalization materialized."""
    device = execution._guard(device, guard)
    cfg = builder.native._config(model_path, config)
    if (not cfg.tie_word_embeddings or embedding.shape != (cfg.vocab_size, cfg.hidden_size)
            or embedding.dtype != torch.bfloat16 or embedding.device != device):
        raise ValueError("Original tied BF16 vocabulary must already occupy the execution device")
    with torch.device("meta"):
        shell = AutoModelForCausalLM.from_config(cfg, trust_remote_code=False, attn_implementation="eager")
    tied = nn.Parameter(embedding.detach(), requires_grad=False)
    shell.model.embed_tokens.weight = shell.lm_head.weight = tied
    # The original HF native teacher is constructed on CPU then moved to CUDA.
    # Preserve those native-prefix frequency bits; only the compressed suffix
    # uses the separately device-constructed training rotary.
    shell.model.rotary_emb = partial.training_rotary(cfg, "cpu").to(device)
    shell.model.norm = nn.Identity()
    shell.generation_config = GenerationConfig.from_model_config(cfg)
    shell.generation_config.use_cache = False
    return shell


def _prefix_layers(model_path, cut, config, device, guard):
    blocks = []
    for index in range(cut):
        execution._guard(device, guard)
        blocks.append(builder.native.load_native_block(model_path, index, device=device,
                                                       dtype=torch.bfloat16, config=config))
    return blocks


@contextmanager
def attach_native_prefix(response_model, model_path, device=None, guard=None):
    """Temporarily attach only the frozen original prefix, sharing embedding E.

    Prefix closure owns native blocks; dropping it releases them before
    training. It runs the current generated tokens, without any training-cache
    lookup. The original prefix's terminal residual is not normalized.
    """
    device = response_model.embedding.device if device is None else torch.device(device)
    guard = response_model.guard if guard is None else guard
    execution._guard(device, guard)
    if response_model.native_prefix is not None:
        raise ValueError("A native prefix is already attached")
    if response_model.cut == 0:
        yield dict(cut=0, loaded_layers=0, shared_embedding=True)
        return
    if response_model.embedding.device != device:
        raise ValueError("Native prefix and response suffix must share an execution device")
    shell = _shell(model_path, response_model.embedding, device, guard)
    shell.model.layers = nn.ModuleList(_prefix_layers(model_path, response_model.cut,
                                                     shell.config, device, guard))
    shell.eval().requires_grad_(False)
    if any(t.is_meta for t in (*shell.parameters(), *shell.buffers())):
        raise ValueError("Native prefix retained an unmaterialized module")
    backbone = shell.model
    assert backbone.embed_tokens.weight.data_ptr() == response_model.embedding.data_ptr()
    del shell

    @torch.no_grad()
    def prefix(ids, valid):
        execution._guard(device, guard)
        return backbone(input_ids=ids, attention_mask=valid, use_cache=False,
                        return_dict=True).last_hidden_state

    response_model.native_prefix = prefix
    try:
        yield dict(cut=response_model.cut, loaded_layers=len(backbone.layers),
                   shared_embedding=True, final_norm="Identity", cache_lookup=False)
    finally:
        response_model.native_prefix = None
        del prefix, backbone
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()


def _packed_reference(response_model, model_path, device, guard):
    """Independent HF model using packed same-checkpoint compact blocks."""
    cfg = builder.native._config(model_path)
    packed = builder.packed_clone(response_model.stack, device="cpu").to(device)
    # The frozen BF16 Triton pointwise kernels require sm80. On a 2060,
    # exercise the packed Torch fallback without changing storage precision.
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
        packed.enabled = False
    norm = deepcopy(response_model.norm).to(device).eval().requires_grad_(False)
    # Preserve the precise construction-device bits of the student's RoPE.
    suffix_rotary = deepcopy(response_model.rotary).to(device).eval().requires_grad_(False)
    embedding = response_model.embedding
    if response_model.cut == 0:
        return execution.attach_compact_model(packed, norm, embedding, suffix_rotary,
            cfg, GenerationConfig.from_model_config(cfg))
    shell = _shell(model_path, embedding, device, guard, config=cfg)
    prefix = _prefix_layers(model_path, response_model.cut, cfg, device, guard)
    # attach_suffix discards these unmaterialized-free placeholders before any
    # forward. Loading the native suffix merely to discard it wastes memory.
    shell.model.layers = nn.ModuleList(prefix + [nn.Identity() for _ in range(cfg.num_hidden_layers-response_model.cut)])
    return partial.attach_suffix(shell, packed, norm, cut=response_model.cut, suffix_rotary=suffix_rotary)


def _unpacked_reference(response_model, model_path, device, guard):
    """HF native control flow with an independently cloned UNPACKED suffix.

    The scientific student trains and generates in unpacked Torch. This control
    isolates its adapter from the distinct packed export transformation.
    """
    stack = builder.cpu_module_clone(response_model.stack).to(device).eval().requires_grad_(False)
    stack.checkpoint_blocks = False
    stack.enabled = False
    norm = deepcopy(response_model.norm).to(device).eval().requires_grad_(False)
    rotary = deepcopy(response_model.rotary).to(device).eval().requires_grad_(False)
    shell = _shell(model_path, response_model.embedding, device, guard)
    prefix = _prefix_layers(model_path, response_model.cut, shell.config, device, guard)
    shell.model.layers = nn.ModuleList(prefix+[partial.TrainingRotarySuffix(stack, rotary)])
    shell.model.norm = norm
    shell.eval().requires_grad_(False)
    if any(t.is_meta for t in (*shell.parameters(), *shell.buffers())):
        raise ValueError("Unpacked HF reference retained an unmaterialized tensor")
    return shell


@torch.no_grad()
def check_full_vocabulary_parity(response_model, model_path, prompt_ids, *, guard=None,
                                 steps=3, atol=.03125, rtol=1/128, reference_mode="packed"):
    """Compare two same-weight paths over a prompt and generated continuations.

    The caller MUST run before registering optimizer gradient hooks. Compact
    stack/norm/RoPE are temporarily moved to CPU; the original embedding stays
    on its device and is shared by the separate HF reference. Thus external
    references to E and parameter identities remain intact.
    """
    device = response_model.embedding.device
    guard = response_model.guard if guard is None else guard
    execution._guard(device, guard)
    if (type(steps) is not int or steps < 1 or not prompt_ids
            or len(prompt_ids)+steps-1 > response_model.max_length
            or reference_mode not in ("packed", "unpacked")):
        raise ValueError("Parity requires bounded prompt and generated prefixes")
    if response_model.native_prefix is not None or any(
            getattr(p, "_post_accumulate_grad_hooks", None) for p in response_model.parameters()):
        raise ValueError("Run parity before optimizer registration and without an attached native prefix")
    parameters = tuple(map(id, response_model.parameters()))
    fixed = offload.fixed_state(response_model.stack, response_model.embedding, response_model.rotary)
    weights = offload.fingerprint(response_model.named_parameters())
    training = response_model.training
    prefixes, outputs = [], []
    reference = None
    try:
        response_model.eval()
        with execution.arithmetic_policy(), attach_native_prefix(response_model, model_path, device, guard):
            ids = list(prompt_ids)
            for _ in range(steps):
                logits = response_model.next_logits(torch.tensor([ids], device=device)).detach().cpu()
                if not torch.isfinite(logits).all(): raise ValueError("Nonfinite response-adapter parity logits")
                prefixes.append(list(ids)); outputs.append(logits)
                ids.append(int(logits.argmax()))
        # This is explicitly before optimizer hooks, never a live trainer move.
        for item in (response_model.stack, response_model.norm, response_model.rotary): item.to("cpu")
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
        construct = _packed_reference if reference_mode == "packed" else _unpacked_reference
        reference = construct(response_model, model_path, device, guard)
        rows = []
        with execution.arithmetic_policy():
            for ids, expected in zip(prefixes, outputs):
                execution._guard(device, guard)
                tensor = torch.tensor([ids], device=device)
                actual = reference(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                   use_cache=False).logits[0, -1].detach().cpu()
                metrics = execution._errors(actual, expected, atol, rtol)
                rows.append(dict(input_ids=ids, vocabulary=actual.numel(),
                    adapter_argmax=int(expected.argmax()), hf_packed_argmax=int(actual.argmax()),
                    greedy_equal=bool(actual.argmax()==expected.argmax()), **metrics))
        return dict(passed=all(r["passed"] and r["greedy_equal"] for r in rows),
                    rows=rows, atol=atol, rtol=rtol, same_checkpoint=True,
                    reference=f"independent HF {reference_mode} compact path", reference_mode=reference_mode,
                    adapter="unpacked response path",
                    input_history="prompt then adapter-generated prefixes",
                    triton_allowed=reference_mode == "packed" and device.type == "cuda"
                                   and torch.cuda.get_device_capability(device)[0] >= 8,
                    confirmed=False)
    finally:
        if reference is not None: del reference
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
        for item in (response_model.stack, response_model.norm, response_model.rotary): item.to(device)
        response_model.train(training)
        if tuple(map(id, response_model.parameters())) != parameters:
            raise ValueError("Parity moved or replaced registered parameter identities")
        if (offload.fixed_state(response_model.stack, response_model.embedding, response_model.rotary) != fixed
                or offload.fingerprint(response_model.named_parameters()) != weights):
            raise ValueError("Parity changed learned or fixed student state")


def check_unpacked_execution_parity(response_model, model_path, prompt_ids, **kwargs):
    """Qualify the actual unpacked scientific execution, without export packing."""
    if "reference_mode" in kwargs:
        raise ValueError("This control requires the independently cloned unpacked HF reference")
    return check_full_vocabulary_parity(response_model, model_path, prompt_ids,
                                       reference_mode="unpacked", **kwargs)
