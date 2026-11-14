"""Development-only test of mean-preserving J-space residual-carry blocks."""
import argparse
from pathlib import Path

import torch

from . import jspace_quadratic_recipe as recipe
from .jspace_full_lease import wait_ready
from .jspace_qwen_parallel_model import load_native_block, load_readout_norm
from .jspace_quadratic_evaluate import stream_score


def run(root, basis_root, source, *, mode="softmax", cut=36):
    from .jspace_mean_preserving_basis import build_block
    root, basis_root, source = map(lambda p: Path(p).resolve(), (root, basis_root, source))
    if root.exists():
        raise FileExistsError(root)
    if mode not in ("softmax", "quadratic") or type(cut) is not int or not 1 <= cut <= 36:
        raise ValueError("Known attention and compressed prefix in1..36 required")
    parent = recipe.verify(source)
    lease = wait_ready([0])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    cases = recipe.read(source / "tasks.json")["splits"]["development"]
    sources = [Path(__file__), Path(__file__).with_name("jspace_mean_preserving_basis.py"),
               Path(__file__).with_name("jspace_residual_carry.py")]
    plan = dict(source=str(source), source_fingerprint=parent["fingerprint"],
        basis_root=str(basis_root), basis_manifest_sha256=recipe.sha(basis_root / "basis/manifest.json"),
        mode=mode, cut=cut, split="development", dtype="float32", initializer="native_projected",
        source_sha256={str(p): recipe.sha(p) for p in sources},
        identities=[c["identity"] for c in cases],
        scope="exploratory mean-preserving geometry intervention; no recovery training or confirmation")
    plan["fingerprint"] = recipe.digest(plan)
    root.mkdir(parents=True)
    recipe.write(root / "plan.json", plan)
    guard = lambda: wait_ready([0])

    def loader(layer):
        guard()
        if layer < cut:
            return build_block(source, basis_root, mode, layer, device="cuda:0",
                               parameter_dtype=torch.float32, guard=guard).pack(
                                   dtype=torch.float32, buffer_dtype=torch.float32).eval().requires_grad_(False)
        return load_native_block(parent["model"]["path"], layer,
            device="cuda:0", dtype=torch.bfloat16).float()

    try:
        embedding, norm, _ = load_readout_norm(parent["model"]["path"], dtype=torch.bfloat16)
        embedding, norm = embedding.float(), norm.float()
        value = stream_score(cases, input_embedding=embedding, block_loader=loader,
            final_readout=(norm, embedding), label_ids=parent["label_ids"],
            scratch_dir=root / "scratch", rotary=recipe.rotary_module(parent, "cuda:0"),
            device="cuda:0", guard=guard, microbatch=4,
            progress=lambda **kw: recipe.event(root, "progress", detail=kw))
        correct = sum(r["prediction"] == r["answer"] for r in value["records"])
        value.update(status="completed", fingerprint=plan["fingerprint"], correct=correct,
            count=len(cases), accuracy=correct/len(cases),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), lease_id=lease["id"])
        if value["peak_reserved_bytes"] > sum(lease["allocation"].values()) * 1024**2:
            raise RuntimeError("Mean-preserving probe exceeded reservation")
        guard()
        recipe.write(root / "result.json", value)
        print({k: value[k] for k in ("status", "correct", "count", "accuracy")})
        return value
    except BaseException as error:
        recipe.write(root / "failed.json", dict(error=repr(error), fingerprint=plan["fingerprint"]))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--basis-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=recipe.DEFAULT_ROOT)
    parser.add_argument("--mode", choices=recipe.ARMS, default="softmax")
    parser.add_argument("--cut", type=int, default=36)
    run(**vars(parser.parse_args()))
