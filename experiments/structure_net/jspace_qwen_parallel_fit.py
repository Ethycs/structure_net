"""Independent per-block basis and graph fits after grouped teacher capture.

The capture producer owns teacher execution. These phases consume completed
local artifacts, and the relation worker never loads a full language model.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .jspace_bench_data import sha, write
from .jspace_benchmark_groebner import fit_benchmark_groebner
from .jspace_logit_sketch import subspace_energy_fraction
from .jspace_qwen_affine_block import AffineQwenBlock
from .jspace_qwen_progressive_calibration import StreamedBFloat16JacobianRows, write_bfloat16_bits


def _guard(device, guard):
    device = torch.device(device)
    if device.type == "cuda" and guard is None:
        raise ValueError("CUDA fitting requires the caller's reservation guard")
    if guard is not None: guard()
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _event(progress, phase, **values):
    value = {"phase": phase, **values}
    if progress is None: print(json.dumps(value, allow_nan=False), flush=True)
    else: progress(value)


def _save(path, value):
    with Path(path).open("xb") as stream: torch.save(value, stream)


def _metadata(capture_dir):
    path = Path(capture_dir) / "metadata.json"
    value = json.loads(path.read_text())
    if value["status"] != "completed": raise ValueError("Grouped capture is incomplete")
    return value


def _read_bits(path, index, device):
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.dtype != np.dtype("int16"): raise ValueError("Expected lossless BF16-bit activation cache")
    try: return torch.from_numpy(np.array(matrix[index], copy=True)).view(torch.bfloat16).to(device)
    finally: matrix._mmap.close()


@torch.no_grad()
def finalize_basis(capture_dir, output_group_dir, *, layer_ids=None, rank=1024, head_dim=64,
                   device="cpu", guard=None, progress=None):
    """Use the frozen affine producer's Gramian, RoPE-pair and V-basis math."""
    device = _guard(device, guard); capture_dir, output_group_dir = Path(capture_dir), Path(output_group_dir)
    metadata = _metadata(capture_dir); selected = metadata["layers"] if layer_ids is None else list(layer_ids)
    if not selected or len(set(selected)) != len(selected) or any(str(i) not in metadata["blocks"] for i in selected):
        raise ValueError("Select unique blocks from the completed capture")
    outputs = {}
    for layer in selected:
        _guard(device, guard); paths = metadata["blocks"][str(layer)]
        stats = torch.load(paths["statistics"], map_location="cpu", weights_only=True)
        grams = stats["grams"].to(device=device, dtype=torch.float32)
        width = grams.shape[-1]; native_head = paths["shape"]["head_dim"]
        if grams.shape != (2, width, width) or not 0 < rank <= width or not 0 < head_dim <= native_head or head_dim % 2:
            raise ValueError("Require two probe groups and valid residual/head dimensions")
        if not torch.isfinite(grams).all(): raise FloatingPointError("Nonfinite input Gramian")
        _event(progress, "basis_eigendecomposition", layer=layer, rank=rank, width=width)
        eigenvalues, basis = torch.linalg.eigh(grams.mean(0)); eigenvalues = eigenvalues.flip(0); basis = basis.flip(1).contiguous()
        orthogonality = float((basis.T @ basis - torch.eye(width, device=device)).abs().max())
        if eigenvalues.sum() <= 0 or not torch.isfinite(basis).all() or orthogonality > 1e-4:
            raise ValueError("Invalid task-Jacobian eigenspace")
        independent = [torch.linalg.eigh(g)[1].flip(1)[:, :rank] for g in grams]
        stability = {"projector_overlap": float((independent[0].T @ independent[1]).square().sum() / rank),
            "random_subspace_overlap_reference": rank / width,
            "group0_basis_on_group1_energy": float(subspace_energy_fraction(grams[1], independent[0])),
            "group1_basis_on_group0_energy": float(subspace_energy_fraction(grams[0], independent[1])),
            "combined_energy": float(eigenvalues[:rank].sum() / eigenvalues.sum())}
        pair_scores = stats["pair_scores"].to(device); mask_grams = stats["mask_grams"].to(device)
        scores = pair_scores.mean(0); pairs = scores.topk(head_dim // 2, dim=-1).indices.sort(-1).values
        moments, value_means = stats["value_moments"].to(device), stats["value_means"].to(device)
        values, vectors = torch.linalg.eigh(moments); value_basis = vectors[:, :, -head_dim:].flip(-1).contiguous()
        removed = torch.ones_like(scores).scatter_(1, pairs, 0.).flatten(); mask_metric = mask_grams.mean(0)
        quadratic = float(removed @ mask_metric @ removed)
        reconstructed = (value_basis @ (value_basis.transpose(1, 2) @ value_means[:, :, None]))[:, :, 0]
        mean_loss = (reconstructed - value_means).square().sum(1) / value_means.square().sum(1).clamp_min(1e-30)
        if (scores.sum(1) <= 0).any() or (values.sum(1) <= 0).any() or mask_metric.trace() <= 0:
            raise ValueError("Degenerate head calibration energy")
        fit = {"basis": basis.cpu(), "means": stats["means"].cpu(), "position_counts": stats["counts"].cpu(),
            "grams": grams.cpu(), "eigenvalues": eigenvalues.cpu(), "rope_pairs": pairs.cpu(), "value_basis": value_basis.cpu(),
            "pair_scores": pair_scores.cpu(), "pair_mask_grams": mask_grams.cpu(), "value_moments": moments.cpu(),
            "value_means": value_means.cpu(), "rank": rank, "head_dim": head_dim}
        info = {"layer": layer, "rank": rank, "head_dim": head_dim, "teacher_fingerprint": metadata["teacher_fingerprint"],
            "normalization": metadata["normalization"], "orthogonality_max": orthogonality, "stability": {str(rank): stability},
            "calibration_cases": len(metadata["calibration_indices"]), "unobserved_positions": (stats["counts"] == 0).nonzero().flatten().tolist(),
            "head_projection": {"rope_pairs": pairs.cpu().tolist(), "native_head_dim": native_head,
                "pair_sensitivity_fraction": (scores.gather(1, pairs).sum(1) / scores.sum(1)).cpu().tolist(),
                "joint_removed_mask_linearized_output_energy": quadratic,
                "joint_removed_mask_energy_over_mask_gramian_trace": quadratic / float(mask_metric.trace()),
                "value_second_moment_fraction": (values[:, -head_dim:].sum(1) / values.sum(1)).cpu().tolist(),
                "value_mean_projection_loss_fraction": mean_loss.cpu().tolist()},
            "input_sha256": {str(capture_dir / "metadata.json"): sha(capture_dir / "metadata.json"), paths["statistics"]: sha(paths["statistics"])} }
        block_dir = output_group_dir / f"block_{layer:02d}"; block_dir.mkdir(parents=True, exist_ok=False)
        _save(block_dir / "fit.pt", fit); write(block_dir / "fit.json", info)
        outputs[str(layer)] = {"directory": str(block_dir), "fit": str(block_dir / "fit.pt"), "metadata": str(block_dir / "fit.json")}
        _event(progress, "basis_complete", layer=layer, combined_energy=stability["combined_energy"])
    return outputs


def _module_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def capture_compiled_development(block_dir, capture_dir, layer_id, native_block, forward_block, *,
                                  device="cpu", guard=None, progress=None):
    """Capture non-GB BF16 SwiGLU at the intended residual/head interface.

    Callback: forward_block(compiled_block, x[1,L,d], case). The caller supplies
    native Qwen mask/RoPE plumbing using a standalone frozen native block.
    This phase never loads a full teacher or touches the native block's weights.
    """
    device = _guard(device, guard); block_dir, capture_dir = Path(block_dir), Path(capture_dir)
    if (block_dir / "interface.json").exists() or (block_dir / "compiled_dev.npy").exists():
        raise FileExistsError("Compiled development interface already exists")
    metadata = _metadata(capture_dir); paths = metadata["blocks"][str(layer_id)]
    items = json.loads((capture_dir / "items.json").read_text())["items"]
    fit = torch.load(block_dir / "fit.pt", map_location="cpu", weights_only=True)
    if native_block.training or any(p.requires_grad for p in native_block.parameters()):
        raise ValueError("Native block must be frozen and in evaluation mode")
    if native_block.self_attn.q_proj.weight.device != device:
        raise ValueError("Caller must place the native block on the reserved compute device")
    native_digest = _module_digest(native_block)
    block = AffineQwenBlock(native_block, fit["basis"][:, :fit["rank"]].to(device), fit["means"].to(device),
        trainable=False, parameter_dtype=torch.bfloat16, capture_mlp=True,
        head_projection={"rope_pairs": fit["rope_pairs"].to(device), "value_basis": fit["value_basis"].to(device)}).eval()
    positions = metadata["activation_positions"]; width = paths["shape"]["intermediate"]
    target = block_dir / "compiled_dev.npy"
    array = np.lib.format.open_memmap(target, mode="w+", dtype=np.int16,
        shape=(len(metadata["development_indices"]) * positions, width))
    try:
        for row, index in enumerate(metadata["development_indices"]):
            _guard(device, guard); case = items[index]
            x = _read_bits(paths["boundary"], slice(case["offset"], case["stop"]), device)[None]
            block.last_mlp_hidden = None
            forward_block(block, x, case)
            if block.last_mlp_hidden is None: raise ValueError("Callback did not execute the compiled block")
            selected = torch.tensor(case["activation_positions"], device=device)
            write_bfloat16_bits(array, slice(row * positions, (row + 1) * positions), block.last_mlp_hidden[0, selected])
        array.flush()
    finally:
        block.set_capture_mlp(False); array._mmap.close()
    if block.mlp_output.dtype != torch.bfloat16: raise ValueError("Intended output map must already use BF16 weights")
    output_weight = block.mlp_output.detach().float().cpu()
    _save(block_dir / "output_weight.pt", output_weight)
    if _module_digest(native_block) != native_digest: raise ValueError("Native block changed during interface capture")
    inputs = [capture_dir / "metadata.json", capture_dir / "items.json", block_dir / "fit.pt", block_dir / "fit.json",
              Path(paths["boundary"]), Path(paths["mlp_calibration"]), Path(paths["mlp_development"]),
              Path(paths["mlp_jacobian_bf16"]), target, block_dir / "output_weight.pt"]
    interface = {"layer": layer_id, "teacher_fingerprint": metadata["teacher_fingerprint"], "native_block_state_sha256": native_digest,
        "rank": fit["rank"], "head_dim": fit["head_dim"], "capture_block": paths,
        "calibration_families": metadata["calibration_families"], "development_families": metadata["development_families"],
        "native_calibration": paths["mlp_calibration"], "native_development": paths["mlp_development"],
        "jacobian": paths["mlp_jacobian_bf16"], "compiled_development": str(target), "output_weight": str(block_dir / "output_weight.pt"),
        "output_weight_precision": "FP32 storage of the actual BF16-rounded compiled non-GB mlp_output",
        "input_sha256": {str(p): sha(p) for p in inputs}, "scope": "calibration and development only; same native cached boundary inputs"}
    write(block_dir / "interface.json", interface)
    _event(progress, "compiled_interface_complete", layer=layer_id, development_rows=len(metadata["development_families"]))
    return interface


class _ProgressRows(StreamedBFloat16JacobianRows):
    def __init__(self, *args, progress=None, layer=None, guard=None, **kwargs):
        super().__init__(*args, **kwargs); self.progress, self.layer, self.guard, self.passes = progress, layer, guard, 0
    def __getitem__(self, index):
        start, stop, _ = index.indices(len(self))
        if start == 0:
            if self.guard is not None: self.guard()
            _event(self.progress, "jacobian_pass_start", layer=self.layer, pass_number=self.passes + 1, rows=len(self))
        value = super().__getitem__(index)
        if stop == len(self):
            self.passes += 1
            _event(self.progress, "jacobian_pass_complete", layer=self.layer, pass_number=self.passes, rows_read=self.rows_read)
        return value


@torch.no_grad()
def fit_relation(block_dir, *, device="cpu", guard=None, seed=861, anchors=24, remove_count=1024,
                 pad_to_multiple=128, svd_rank=64, oversample=16, power_iterations=1,
                 ridge=.01, eligibility_nrmse=.5, chunk_rows=256, progress=None):
    """One independent FP32 fit; streamed Jacobian, fixed graph width and budget."""
    device = _guard(device, guard); block_dir = Path(block_dir)
    if (block_dir / "relation.pt").exists() or (block_dir / "relations.json").exists(): raise FileExistsError("Relation already exists")
    interface = json.loads((block_dir / "interface.json").read_text())
    _event(progress, "relation_verify_inputs", layer=interface["layer"])
    for name, expected in interface["input_sha256"].items():
        if sha(name) != expected: raise ValueError("Changed fitting input: " + name)
    training = _read_bits(interface["native_calibration"], slice(None), device).float()
    native_dev = _read_bits(interface["native_development"], slice(None), device).float()
    compiled_dev = _read_bits(interface["compiled_development"], slice(None), device).float()
    output_weight = torch.load(interface["output_weight"], map_location="cpu", weights_only=True).to(device=device, dtype=torch.float32)
    if not torch.equal(output_weight, output_weight.bfloat16().float()): raise ValueError("Output map lost intended BF16 rounding provenance")
    _event(progress, "relation_fit_start", layer=interface["layer"], calibration_rows=len(training), development_rows=len(native_dev))
    with _ProgressRows(interface["jacobian"], device=device, max_rows=chunk_rows,
                       progress=progress, layer=interface["layer"], guard=guard) as rows:
        result = fit_benchmark_groebner(training, interface["calibration_families"], rows, output_weight,
            native_dev, interface["development_families"], compressed_dev_activations=compiled_dev,
            anchor_widths=(anchors,), remove_count=remove_count, seed=seed, svd_rank=svd_rank, oversample=oversample,
            power_iterations=power_iterations, ridge=ridge, eligibility_nrmse=eligibility_nrmse, chunk_rows=chunk_rows)
        streamed = rows.metadata()
    chosen = result["fits"][str(anchors)]
    _save(block_dir / "relation.pt", chosen)
    logical = training.shape[1] - remove_count + 1 + anchors + anchors * (anchors + 1) // 2
    report = {k: v for k, v in result.items() if k != "fits"}
    report.update(layer=interface["layer"], teacher_fingerprint=interface["teacher_fingerprint"],
        training_families=dict(Counter(interface["calibration_families"])), development_families=dict(Counter(interface["development_families"])),
        input_sha256={**interface["input_sha256"], str(block_dir / "interface.json"): sha(block_dir / "interface.json")},
        streamed_jacobian=streamed, fixed_anchor_width=anchors, removed_channels=remove_count,
        runtime_padding={"multiple": pad_to_multiple, "logical_width": logical,
                         "padded_width": math.ceil(logical / pad_to_multiple) * pad_to_multiple if pad_to_multiple else logical},
        source="fixed accepted-teacher grouped full-context BF16 VJPs and native task activations",
        relation_sha256=sha(block_dir / "relation.pt"),
        scope="approximate fitted graph ideal; exact graph certificate is not an identity of the teacher network; no test selection")
    write(block_dir / "relations.json", report); _guard(device, guard)
    _event(progress, "relation_fit_complete", layer=interface["layer"], anchors=anchors, removed=remove_count)
    return report
