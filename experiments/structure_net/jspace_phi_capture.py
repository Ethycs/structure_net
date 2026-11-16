"""The original grouped full-vocabulary VJP estimator on fused Phi QKV.

Native Phi forward execution is unchanged. The derivative target is the
actual fused qkv_proj output; Q/K gradients are slices of that VJP, not
gradients requested from disconnected hook-created views. Statistics, probe
normalization, all-token Jacobian rows, training means and artifact layout
remain those of GroupedTeacherCapture and feed its existing basis fitter.

The caller supplies a frozen teacher with model.layers[absolute_layer] and
lm_head.weight (Phi's UNTIED output head), plus a forward callback returning
final normalized [1,L,width] hidden states. A cached native prefix is allowed.
No full language model or replacement projection modules are constructed.
"""
from types import SimpleNamespace

import torch

from .jspace_phi_affine_block import compatible_view
from .jspace_logit_sketch import project_sign_probe_groups
from .jspace_qwen_progressive_calibration import GroupedTeacherCapture, write_bfloat16_bits


def capture_compiled_development(block_dir, capture_dir, layer_id, native_block, forward_block,
                                 *, device='cpu', guard=None, progress=None):
    """Reuse the existing non-polynomial MLP interface and rank-fit artifacts.

    The callback receives a shared affine block and must provide native Phi
    masks/RoPE. The local view changes no module and executes no forward; it
    only exposes fused-weight slices plus original state/freeze inspection.
    """
    from .jspace_qwen_parallel_fit import capture_compiled_development as original
    view = compatible_view(native_block)
    if native_block.self_attn.layer_idx != layer_id:
        raise ValueError("Compiled Phi interface layer must match native block")
    view.parameters = native_block.parameters
    view.state_dict = native_block.state_dict
    return original(block_dir, capture_dir, layer_id, view, forward_block,
                    device=device, guard=guard, progress=progress)


class PhiGroupedTeacherCapture(GroupedTeacherCapture):
    def __init__(self, root, model, layer_ids, cases, calibration_indices, **kwargs):
        if kwargs.get('guard') is not None: kwargs['guard']()
        layers = tuple(layer_ids)
        if not layers or any(type(i) is not int or i < 0 for i in layers):
            raise ValueError("Nonnegative integer native Phi layer indices required")
        native = {i: model.model.layers[i] for i in layers}
        for i, block in native.items():
            compatible_view(block)  # Validate actual dense fused Phi geometry.
            if block.self_attn.layer_idx != i:
                raise ValueError("Phi hook layer index must match its absolute position")
            if any(p.dtype != torch.bfloat16 for p in block.parameters()):
                raise ValueError("Lossless grouped Phi capture requires BF16 teacher execution")
        if model.lm_head.weight.dtype != torch.bfloat16 or model.lm_head.weight.ndim != 2:
            raise ValueError("Full-vocabulary untied Phi readout must execute in BF16")
        # Parent constructor only needs q_proj.in_features for allocation. The
        # facade is local metadata, never assigned to the original module.
        views = {i: SimpleNamespace(self_attn=SimpleNamespace(config=b.self_attn.config,
                    head_dim=b.self_attn.head_dim,
                    q_proj=SimpleNamespace(in_features=b.self_attn.qkv_proj.in_features)), mlp=b.mlp)
                 for i, b in native.items()}
        facade = SimpleNamespace(training=model.training, parameters=model.parameters, buffers=model.buffers,
                                 lm_head=model.lm_head, model=SimpleNamespace(layers=views))
        super().__init__(root, facade, layers, cases, calibration_indices, **kwargs)
        self.model, self.modules = model, native
        self.metadata.update(architecture="phi3_fused_qkv_phi4", readout="untied lm_head.weight",
            derivative_targets=["native block input", "actual fused qkv_proj output", "actual down_proj input"],
            qk_derivatives="Q/K slices of fused-output VJP; native fused forward unchanged",
            phi_geometry={str(i): compatible_view(b).self_attn.config.to_dict() for i, b in native.items()})

    def capture_case(self, index, forward):
        if self.failed or self.finished:
            raise ValueError("Capture is failed or already finished")
        if type(index) is not int or index in self.seen or not 0 <= index < len(self.cases):
            raise ValueError("Invalid or duplicate case index")
        self._check_teacher(); calibration = index in self.cal_rows
        if calibration and torch.is_inference_mode_enabled():
            raise ValueError("Jacobian capture cannot run inside inference_mode")
        captured, handles = {i: {} for i in self.layers}, []

        def boundary_hook(module, args, layer):
            value = args[0]
            if layer == self.layers[0]:
                value = value.detach().clone().requires_grad_(True) if calibration else value.detach()
            captured[layer]["input"] = value
            return (value, *args[1:])

        def qkv_hook(module, args, output, layer):
            spec = self.shapes[layer]
            qn, kn = spec['heads'] * spec['head_dim'], spec['kv_heads'] * spec['head_dim']
            if output.shape[-1] != qn + 2 * kn:
                raise ValueError("Unexpected fused Phi QKV output width")
            captured[layer]['qkv'] = output
            captured[layer]['q'], captured[layer]['k'], captured[layer]['v'] = output.split((qn, kn, kn), -1)

        try:
            for layer, block in self.modules.items():
                handles.append(block.register_forward_pre_hook(lambda m, a, i=layer: boundary_hook(m, a, i)))
                handles.append(block.self_attn.qkv_proj.register_forward_hook(lambda m, a, o, i=layer: qkv_hook(m, a, o, i)))
                handles.append(block.mlp.down_proj.register_forward_pre_hook(
                    lambda m, a, i=layer: captured[i].__setitem__('mlp', a[0])))
            with torch.enable_grad():
                output = forward()
                if output.shape != (1, self.lengths[index], self.readout.shape[1]):
                    raise ValueError("forward must return normalized [1,L,d] teacher states")
                if calibration and not output.requires_grad:
                    raise ValueError("Phi suffix/final RMSNorm was detached")
            for handle in handles: handle.remove()
            handles.clear()
            if any(set(values) != {'input', 'qkv', 'q', 'k', 'v', 'mlp'} for values in captured.values()):
                raise ValueError("Requested native Phi block was not executed")
            super()._record_forward(index, {i: {k: v for k, v in values.items() if k != 'qkv'}
                                            for i, values in captured.items()}, output)
            if calibration:
                with torch.enable_grad(): self._record_vjps(index, captured, output)
            self._check_teacher(); self.seen.add(index); self.order.append(index)
        except BaseException:
            self.failed = True
            raise
        finally:
            for handle in handles: handle.remove()

    def _record_vjps(self, index, captured, output):
        normalized = output[0, self.cases[index]['loss_positions']]
        vectors = project_sign_probe_groups(self.readout, self.output_positions, self.probes,
                                           self.generator, groups=self.groups)
        targets = [captured[i][key] for i in self.layers for key in ('input', 'qkv', 'mlp')]
        length = self.lengths[index]
        for group in range(self.groups):
            for probe in range(self.probes):
                scalar = (normalized.to(vectors.dtype) * vectors[group, probe]).sum()
                gradients = torch.autograd.grad(scalar, targets,
                    retain_graph=not (group == self.groups - 1 and probe == self.probes - 1))
                for offset, layer in enumerate(self.layers):
                    gx, gqkv, gm = gradients[3 * offset:3 * offset + 3]
                    stats, spec = self.statistics[layer], self.shapes[layer]
                    qn, kn = spec['heads'] * spec['head_dim'], spec['kv_heads'] * spec['head_dim']
                    gq, gk, _ = gqkv.split((qn, kn, kn), -1)
                    row = self.jac_offsets[index] + (group * self.probes + probe) * length
                    write_bfloat16_bits(self.arrays[layer]['mlp_jacobian_bf16'], slice(row, row + length), gm[0])
                    x = gx[0].detach().float()
                    if not torch.isfinite(x).all(): raise FloatingPointError("Nonfinite Phi input VJP")
                    stats['input_grams'][group].addmm_(x.T, x, alpha=1 / self.probes)
                    kv, h = spec['kv_heads'], spec['head_dim']; sharing = spec['heads'] // kv
                    q = (captured[layer]['q'].detach().float() * gq.float())[0].reshape(length, kv, sharing, h).sum((0, 2))
                    k = (captured[layer]['k'].detach().float() * gk.float())[0].reshape(length, kv, h).sum(0)
                    mask = q[:, :h // 2] + q[:, h // 2:] + k[:, :h // 2] + k[:, h // 2:]
                    stats['pair_scores'][group].add_(mask.square() / self.probes)
                    flat = mask.flatten()
                    stats['mask_grams'][group].add_(flat[:, None] * flat[None, :] / self.probes)
