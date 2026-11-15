"""Local raw-text inference for the custom 36-block compressed Qwen2 bundle."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import torch
import transformers

from . import jspace_allblocks_artifact as artifact

RUNTIME_FILES = (
    "jspace_allblocks_artifact.py", "jspace_qwen_affine_block.py", "jspace_qwen_compact_stack.py",
    "jspace_qwen_affine_fast.py", "jspace_compact_model.py", "jspace_qwen_affine_stack.py",
)
MIB = 1024**2


def verify_runtime(directory):
    """Check executable identity and local text assets before loading weights/CUDA."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != artifact.SCHEMA:
        raise ValueError("Unsupported compressed bundle schema")
    versions = {"torch": str(torch.__version__), "transformers": transformers.__version__}
    if manifest.get("versions") != versions:
        raise ValueError("Runtime version mismatch; use the bundle's pinned repository and Pixi environment")
    hashes = manifest.get("runtime_source_sha256", {})
    if set(hashes) != set(RUNTIME_FILES):
        raise ValueError("Runtime source inventory mismatch")
    for name in RUNTIME_FILES:
        if artifact.sha(Path(artifact.__file__).parent / name) != hashes[name]:
            raise ValueError("Runtime source mismatch: " + name)
    capacity = manifest.get("position_capacity")
    if type(capacity) is not int or not 0 < capacity <= 416:
        raise ValueError("This CLI supports a declared position capacity of at most 416")
    files = manifest.get("files", {})
    if not {"config.json", "generation_config.json", "tokenizer_config.json", "weights.safetensors"} <= set(files):
        raise ValueError("Incomplete bundle file inventory")
    # The existing custom loader verifies all files, including the large tensor
    # payload. Verify the small text assets here before invoking the tokenizer.
    for name, digest in files.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory):
            raise ValueError("Bundle file escapes its directory")
        if name != "weights.safetensors" and artifact.sha(path) != digest:
            raise ValueError("Bundle text/metadata hash mismatch: " + name)
    return manifest


def tokenize_prompt(tokenizer, prompt, *, plain=False, max_new_tokens=32, capacity=416):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Provide nonempty prompt text")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if plain:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    else:
        if not tokenizer.chat_template:
            raise ValueError("Bundle has no chat template; use --plain for plain completion")
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, truncation=False)
    ids = encoded["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("Exactly one nonempty tokenized prompt is required")
    length = ids.shape[1]
    if length + max_new_tokens > capacity:
        raise ValueError(f"Input ({length}) + requested output ({max_new_tokens}) exceeds calibrated capacity ({capacity}); no truncation applied")
    return dict(input_ids=ids, attention_mask=encoded.get("attention_mask", torch.ones_like(ids)))


def wait_for_lease():
    from .jspace_full_lease import wait_ready
    return wait_ready([0])


def cuda_reservation(*, reservation=None, cuda=None, visible=None):
    """Authorize before any CUDA query; recheck the same lease during decoding."""
    reservation = wait_for_lease if reservation is None else reservation
    cuda = torch.cuda if cuda is None else cuda
    lease = reservation()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "") if visible is None else visible
    allocation = lease["allocation"]
    if "," in visible or not visible or visible not in allocation:
        raise ValueError("Exactly one visible GPU UUID covered by the active lease is required")
    mib = allocation[visible]
    if type(mib) is not int or mib <= 256:
        raise ValueError("GPU reservation must exceed 256 MiB allocator headroom")
    if cuda.is_initialized():
        raise RuntimeError("CUDA inference requires a fresh process for the reservation memory cap")
    total = int(cuda.get_device_properties(0).total_memory)
    limit = (mib - 256) * MIB
    if not 0 < limit <= total or cuda.memory.get_allocator_backend() != "native":
        raise ValueError("Reservation requires a valid device budget and native PyTorch allocator")
    cuda.set_per_process_memory_fraction(limit / total, device=0)
    cuda.reset_peak_memory_stats(0)

    def guard():
        current = reservation()
        if current["id"] != lease["id"] or current["allocation"] != allocation:
            raise RuntimeError("GPU reservation changed during inference")
        if cuda.max_memory_reserved(0) > limit:
            raise MemoryError("Inference exceeded the reservation allocator limit")

    return guard, dict(lease_id=lease["id"], physical_uuid=visible,
                       lease_bytes=mib*MIB, allocator_limit_bytes=limit)


def infer(directory, prompt, *, plain=False, max_new_tokens=32, device="cpu"):
    if device not in ("cpu", "cuda:0"):
        raise ValueError("Use cpu or a single reserved cuda:0 device")
    directory = Path(directory).resolve()
    manifest = verify_runtime(directory)
    tokenizer = artifact.AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    batch = tokenize_prompt(tokenizer, prompt, plain=plain, max_new_tokens=max_new_tokens,
                            capacity=manifest["position_capacity"])
    guard, reservation = cuda_reservation() if device == "cuda:0" else (lambda: None, None)
    guard()
    if device == "cuda:0":
        torch.backends.cuda.matmul.allow_tf32 = False
    model, _ = artifact.load_artifact(directory, device=device, load_tokenizer=False)
    # Refuse a bundle changed between the CPU preflight and the model loader.
    if model.artifact_manifest != manifest:
        raise ValueError("Bundle manifest changed while loading")
    guard()
    generation = deepcopy(model.generation_config)
    generation.do_sample = False
    generation.num_beams = generation.num_beam_groups = generation.num_return_sequences = 1
    generation.max_new_tokens = max_new_tokens
    # Preserve pinned EOS/pad tokens, repetition penalty and other processors.
    batch = {key: value.to(device) for key,value in batch.items()}
    hook = model.register_forward_pre_hook(lambda _module, _args: guard())
    try:
        with torch.inference_mode():
            output = model.generate(**batch, generation_config=generation)
        if device == "cuda:0":
            torch.cuda.synchronize(0)
        guard()
    finally:
        hook.remove()
    sequences = output.sequences if hasattr(output, "sequences") else output
    tokens = sequences[0, batch["input_ids"].shape[1]:].tolist()
    return dict(text=tokenizer.decode(tokens, skip_special_tokens=True), generated_token_ids=tokens,
                input_tokens=batch["input_ids"].shape[1], generated_tokens=len(tokens),
                position_capacity=manifest["position_capacity"], source_fingerprint=manifest["source_fingerprint"],
                device=device, greedy=True, runtime_versions=manifest["versions"], reservation=reservation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--plain", action="store_true", help="Tokenize raw completion text without the chat template")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--json", action="store_true", help="Include token counts and runtime identity")
    args = parser.parse_args(argv)
    result = infer(args.bundle, args.prompt, plain=args.plain, max_new_tokens=args.max_new_tokens, device=args.device)
    print(json.dumps(result, ensure_ascii=False) if args.json else result["text"])
    return result


if __name__ == "__main__":
    main()
