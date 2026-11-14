"""Development-only first-block rank/head ablations with35 original blocks."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import torch

from . import jspace_quadratic_recipe as recipe
from .jspace_prefix_localize import development_cases
from .jspace_qwen_affine_block import AffineQwenBlock
from .jspace_quadratic_qwen import QuadraticQwenBlock
from .jspace_residual_carry import ResidualCarryBlock
from .jspace_quadratic_evaluate import stream_score

VARIANTS = ("full_rank_full_head", "full_rank_half_head", "half_rank_full_head", "mean_rank_full_head")
DEFAULT_MEAN_ROOT = recipe.ROOT / "data/experiments/jspace_mean_preserving_basis/20261114_v1"


def build_first(native, original_fit, variant, *, mean_fit=None, guard=None):
    """Native-weight initialization only; no selected recovery checkpoint."""
    if variant not in VARIANTS:
        raise ValueError("Unknown first-block ablation")
    parameter = next(native.parameters())
    if parameter.device.type == "cuda":
        if guard is None:
            raise ValueError("GPU construction requires a reservation guard")
        guard()
    if native.self_attn.layer_idx != 0:
        raise ValueError("Only native block0 may be replaced")
    native = native.float().eval().requires_grad_(False)
    width, device = native.self_attn.q_proj.in_features, parameter.device
    if variant.startswith("full_rank"):
        basis = torch.eye(width, device=device, dtype=torch.float32)
        means = torch.zeros(1, width, device=device, dtype=torch.float32)
    else:
        fit = mean_fit if variant == "mean_rank_full_head" else original_fit
        if fit is None:
            raise ValueError("Mean-repaired variant requires its fitted mean/basis")
        basis, means = fit["basis"].to(device), fit["means"].to(device)
        if basis.shape != (width, width//2) or means.shape != (1, width):
            raise ValueError("Expected half-width fitted basis and global native mean")
    head = {key: original_fit[key].to(device) for key in ("rope_pairs", "value_basis")} if variant == "full_rank_half_head" else None
    affine = AffineQwenBlock(native, basis, means, trainable=False,
        parameter_dtype=torch.float32, head_projection=head)
    wrapped = QuadraticQwenBlock(affine, mode="softmax", learn_metric=False)
    return ResidualCarryBlock(wrapped).pack(dtype=torch.float32, buffer_dtype=torch.float32)


def run(root, source, variant, mean_root=DEFAULT_MEAN_ROOT):
    from .jspace_full_lease import wait_ready
    from .jspace_qwen_parallel_model import load_native_block, load_readout_norm
    root, source, mean_root = (Path(path).resolve() for path in (root, source, mean_root))
    if root.exists():
        raise FileExistsError(root)
    if variant not in VARIANTS:
        raise ValueError("Unknown first-block ablation")
    parent = recipe.verify(source)
    cases, _ = development_cases(source)
    fit_path = recipe.fit_path(source, 0)
    original_manifest = recipe.checked(source, source / "basis", hashes=False)
    if recipe.sha(fit_path) != original_manifest["artifacts"][str(fit_path.relative_to(source))]:
        raise ValueError("Changed original block0 fit")
    inputs = {str(path): recipe.sha(path) for path in
        (source / "plan.json", source / "tasks.json", source / "items.json", source / "basis/manifest.json", fit_path)}
    original_fit, mean_fit = recipe.load(fit_path), None
    mean_fingerprint = None
    if variant == "mean_rank_full_head":
        plan, manifest = recipe.read(mean_root / "plan.json"), recipe.read(mean_root / "basis/manifest.json")
        if (plan["fingerprint"] != recipe.digest({k:v for k,v in plan.items() if k != "fingerprint"}) or
                plan["source_fingerprint"] != parent["fingerprint"] or
                manifest["status"] != "completed" or manifest["fingerprint"] != plan["fingerprint"]):
            raise ValueError("Mean-repaired basis provenance mismatch")
        path = mean_root / "basis/block_00/fit.pt"
        if recipe.sha(path) != manifest["artifacts"][str(path.relative_to(mean_root))]:
            raise ValueError("Changed mean-repaired block0 fit")
        mean_fit, mean_fingerprint = recipe.load(path), plan["fingerprint"]
        inputs.update({str(p):recipe.sha(p) for p in (path, mean_root / "plan.json", mean_root / "basis/manifest.json")})
    sources = [Path(__file__), Path(__file__).with_name("jspace_residual_carry.py"),
        Path(__file__).with_name("jspace_prefix_localize.py"),
        recipe.ROOT / "tests/structure_net/test_jspace_first_block_ablation.py"]
    plan = dict(schema="jspace.first-block-ablation.v1", source=str(source),
        source_fingerprint=parent["fingerprint"], variant=variant, replaced_layers=[0],
        original_suffix_layers=list(range(1, 36)), mean_basis_fingerprint=mean_fingerprint,
        initializer="original native BF16-rounded weights, promoted toFP32; no recovery checkpoint",
        attention="softmax; no learned metric", residual="full native residual with compact update and dynamic RMS complement energy",
        native_embedding_and_readout=True, split="development", dtype="float32",
        identities=[case["identity"] for case in cases], input_sha256=inputs,
        source_sha256={str(path):recipe.sha(path) for path in sources},
        scope="Mechanism ablation only; no recovery training, confirmation or deployment speed claim")
    plan["fingerprint"] = recipe.digest(plan)
    lease = wait_ready([0]); guard = lambda: wait_ready([0])
    torch.cuda.init(); torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    root.mkdir(parents=True)
    for path in sources:
        target = root / "producer" / path.relative_to(recipe.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    recipe.write(root / "plan.json", plan)
    try:
        def loader(layer):
            guard()
            native = load_native_block(parent["model"]["path"], layer, device="cuda:0", dtype=torch.bfloat16).float()
            return build_first(native, original_fit, variant, mean_fit=mean_fit, guard=guard) if layer == 0 else native
        embedding, norm, _ = load_readout_norm(parent["model"]["path"], device="cpu", dtype=torch.bfloat16)
        embedding, norm = embedding.float(), norm.float()
        value = stream_score(cases, input_embedding=embedding, block_loader=loader,
            final_readout=(norm, embedding), label_ids=parent["label_ids"], scratch_dir=root / "scratch",
            rotary=recipe.rotary_module(parent, "cuda:0"), device="cuda:0", guard=guard,
            layer_count=36, microbatch=4, progress=lambda **kw:recipe.event(root,"progress",detail=kw))
        rows = value["records"]; correct = sum(row["prediction"] == row["answer"] for row in rows)
        peak = torch.cuda.max_memory_reserved()
        if peak > sum(lease["allocation"].values())*1024**2:
            raise RuntimeError("First-block ablation exceeded its reservation")
        guard()
        value.update(status="completed", fingerprint=plan["fingerprint"], variant=variant,
            correct=correct, count=len(rows), accuracy=correct/len(rows),
            by_family={family:dict(correct=sum(row["prediction"]==row["answer"] for row in rows if row["family"]==family),
                count=sum(row["family"]==family for row in rows)) for family in sorted({row["family"] for row in rows})},
            peak_reserved_bytes=peak, lease_id=lease["id"], scope=plan["scope"])
        recipe.write(root / "result.json", value)
        recipe.write(root / "manifest.json", dict(status="completed", fingerprint=plan["fingerprint"],
            artifacts={"plan.json":recipe.sha(root/"plan.json"), "result.json":recipe.sha(root/"result.json")}))
        return {k:value[k] for k in ("status", "variant", "correct", "count", "accuracy")}
    except BaseException as error:
        recipe.write(root / "failed.json", dict(error=repr(error), fingerprint=plan["fingerprint"]))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=recipe.DEFAULT_ROOT)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--mean-root", type=Path, default=DEFAULT_MEAN_ROOT)
    print(run(**vars(parser.parse_args())))
