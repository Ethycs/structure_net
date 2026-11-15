"""Dense prealignment of one compact entrance into an unchanged parent suffix.

The caller owns provenance, GPU reservations, task qualification and subsequent
joint recovery. This helper never reconstructs a native-width student state,
changes the architecture, or treats feature loss as a task-accuracy gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import random

import torch

from . import jspace_incremental_suffix as evidence
from . import jspace_qwen_parallel_train as core
from . import jspace_single_block_distill as single
from .jspace_qwen_affine_stack import forward_reduced
from .jspace_qwen_compact_stack import CompactQwenStack
from .jspace_streaming_adamw import StreamingCPUAdamW

EntrancePairs = single.NativePairs


@dataclass(frozen=True)
class EntranceConfig:
    epochs: int = 12
    microbatch: int = 4
    accumulation: int = 4
    learning_rate: float = 2e-5
    gradient_clip: float = 1.
    seed: int = 20861
    floor_fraction: float = 1e-4
    optimizer_chunk_bytes: int = 32 * 1024**2
    max_cpu_state_bytes: int = 2 * 1024**3

    def __post_init__(self):
        core.RecoveryConfig(epochs=self.epochs, microbatch=self.microbatch,
            accumulation=self.accumulation, learning_rate=self.learning_rate,
            gradient_clip=self.gradient_clip, seed=self.seed,
            optimizer_chunk_bytes=self.optimizer_chunk_bytes)
        if (not math.isfinite(self.floor_fraction) or not 0 < self.floor_fraction < 1
                or type(self.max_cpu_state_bytes) is not int or self.max_cpu_state_bytes <= 0):
            raise ValueError("Positive CPU-state cap and fractional training-energy floor required")


def entrance_coordinates(stack, hidden, positions, attention_mask, position_embeddings):
    """Execute the same entrance and first bridge used by CompactQwenStack."""
    if stack.packed or stack.stripped or len(stack.blocks) < 2:
        raise ValueError("Unpacked expanded stack with an inherited parent required")
    first, second = stack.blocks[:2]
    locations = [b._positions(hidden, positions[0], positions, None) for b in (first, second)]
    z = stack._geometry(hidden, first.basis.T)
    z, _ = forward_reduced(first, z, locations[0], attention_mask=attention_mask,
        cache_position=positions[0], position_embeddings=position_embeddings)
    bridge = stack.bridges[0]
    shift = first._rows(bridge.shift, locations[bridge.position_source], *z.shape[:2])
    return stack._geometry(z, bridge.weight, shift)


class EntranceTrainer:
    """Train only block0's original parameters; keep every other tensor fixed.

    ``data`` is EntrancePairs over native boundary(cut) and boundary(cut+1).
    Its native BF16 bits are promoted for loss arithmetic, then projected with
    the actual stored BF16 parent basis. The optional quality(stack) callback
    must score a detached clone and return JSON-compatible diagnostics; its
    result never enters feature selection. No tensors may be replaced by it.
    """
    def __init__(self, stack, norm, readout, rotary, data, *, config=EntranceConfig(),
                 guard=None, quality=None):
        if not isinstance(stack, CompactQwenStack) or stack.packed or stack.stripped or len(stack.blocks) < 2:
            raise ValueError("An unpacked expanded CompactQwenStack is required")
        self.stack, self.norm, self.readout, self.rotary, self.data = stack, norm, readout, rotary, data
        self.config, self.guard, self.quality = config, guard, quality
        self.block = stack.blocks[0]
        self.device = next(self.block.parameters()).device
        if self.device.type == "cuda" and guard is None:
            raise ValueError("CUDA prealignment requires the caller's reservation guard")
        self._guard()
        if (stack.geometry_mode != "bf16" or stack.geometry_compute_dtype is not None
                or data.inputs.shape[1] != stack.width or stack.blocks[1].basis.dtype != torch.bfloat16):
            raise ValueError("Native-width pairs and stored BF16 geometry without compute override required")
        if any(b.position_count > 1 and max(len(c['input_ids']) for c in data.items) > b.position_count
               for b in stack.blocks[:2]):
            raise ValueError("Cached prompts exceed entrance/parent position capacity")
        stack.eval().requires_grad_(False); norm.eval().requires_grad_(False)
        readout.requires_grad_(False); rotary.eval().requires_grad_(False)
        for p in (*stack.parameters(), *norm.parameters()):
            p.grad = None
        readout.grad = None
        self.block.requires_grad_(True)
        self.parameters = list(self.block.parameters())
        for p in self.parameters:
            p.data = p.data.to(torch.bfloat16)
        self.parameter_ids = tuple(id(p) for p in stack.parameters())
        self.state_keys = tuple(stack.state_dict())
        self.optimizer = StreamingCPUAdamW(self.parameters, lr=config.learning_rate, weight_decay=0.,
            chunk_bytes=config.optimizer_chunk_bytes, max_cpu_state_bytes=config.max_cpu_state_bytes)
        self.updates, self.finished, self.discarded = 0, False, False
        self.fixed_before = self.fixed_signature()
        self.normalization = self.fit_floor()

    def _guard(self):
        if self.guard is not None:
            self.guard()

    def fixed_signature(self):
        return evidence.fingerprint(
            [("stack."+n, v) for n, v in self.stack.named_buffers()]
            + [(f"parent.{i}."+n, v) for i, b in enumerate(self.stack.blocks[1:]) for n, v in b.named_parameters()]
            + [("norm."+n, v) for n, v in self.norm.named_parameters()]
            + [("norm_buffer."+n, v) for n, v in self.norm.named_buffers()]
            + [("readout", self.readout)]
            + [("rotary."+n, v) for n, v in [*self.rotary.named_parameters(), *self.rotary.named_buffers()]])

    def _unchanged(self):
        return (self.fixed_signature() == self.fixed_before
            and tuple(id(p) for p in self.stack.parameters()) == self.parameter_ids
            and tuple(self.stack.state_dict()) == self.state_keys)

    def parent_coordinates(self, native):
        return self.stack._geometry(native.to(torch.bfloat16), self.stack.blocks[1].basis.T)

    @torch.no_grad()
    def fit_floor(self):
        energies = []
        for start in range(0, len(self.data.training), self.config.microbatch):
            self._guard(); batch = self.data.batch(self.data.training[start:start+self.config.microbatch], self.device)
            before = self.parent_coordinates(batch['hidden']).float()
            after = self.parent_coordinates(batch['target']).float()
            energies.append((after-before).square().mean(-1)[batch['valid']].cpu())
        values = torch.cat(energies); median = float(values.median())
        if median <= 0 or not torch.isfinite(values).all():
            raise ValueError("Degenerate training teacher-update energy in parent coordinates")
        return dict(train_case_count=len(self.data.training), train_token_count=len(values),
            median_teacher_update_energy=median, floor_fraction=self.config.floor_fraction,
            floor=self.config.floor_fraction*median,
            denominator="mean((BF16(native_next @ U_parent)-BF16(native_input @ U_parent))^2), FP32 loss arithmetic")

    def losses(self, batch):
        hidden = batch['hidden'].to(torch.bfloat16); positions = batch['positions']
        # Match the existing BoundaryDataset: causal + valid keys, with padded
        # query rows ignored only by the equal-case/valid-token feature loss.
        length = hidden.shape[1]
        causal = torch.arange(length, device=self.device)[None, :] <= torch.arange(length, device=self.device)[:, None]
        allowed = causal[None] & batch['valid'][:, None, :]
        mask = torch.zeros((len(hidden), 1, length, length), dtype=torch.bfloat16, device=self.device)
        mask.masked_fill_(~allowed[:, None], torch.finfo(torch.bfloat16).min)
        result = entrance_coordinates(self.stack, hidden, positions, mask, self.rotary(hidden, positions))
        with torch.no_grad():
            before = self.parent_coordinates(batch['hidden'])
            target = self.parent_coordinates(batch['target'])
        return single.feature_losses(result, before, target, batch['valid'], floor=self.normalization['floor'])

    def train_window(self, indices, *, learning_rate=None):
        if self.finished or self.discarded:
            raise ValueError("Finished/discarded trainer cannot update")
        if not indices or any(i not in self.data.training for i in indices):
            raise ValueError("Optimizer updates require training cases only")
        self._guard(); self.block.train(); self.optimizer.zero_grad(set_to_none=True)
        if learning_rate is not None:
            self.optimizer.param_groups[0]['lr'] = learning_rate
        totals = {}
        for start in range(0, len(indices), self.config.microbatch):
            self._guard(); chosen = indices[start:start+self.config.microbatch]
            values = self.losses(self.data.batch(chosen, self.device)); weight = len(chosen)/len(indices)
            (values['loss']*weight).backward()
            for key, value in values.items():
                totals[key] = totals.get(key, 0.) + float(value.detach())*weight
        if any(p.grad is None for p in self.parameters):
            raise RuntimeError("Every entrance parameter must participate")
        if any(p.grad is not None for b in self.stack.blocks[1:] for p in b.parameters()):
            raise RuntimeError("Inherited parent received gradients")
        update = self.optimizer.step(max_grad_norm=self.config.gradient_clip)
        self.optimizer.zero_grad(set_to_none=True); self.updates += 1
        return dict(**totals, update=self.updates, cases=len(indices), optimizer=update,
            learning_rate=self.optimizer.param_groups[0]['lr'], only_entrance_gradients=True)

    def pilot(self, indices):
        if self.updates or self.finished or self.discarded:
            raise ValueError("Pilot requires an untouched disposable trainer")
        def masters():
            return {name: evidence.fingerprint([(name, self.optimizer.state[p]['master'])])
                    for name, p in self.block.named_parameters()}
        before = masters()
        update = self.train_window(indices)
        after = masters()
        changed = {name: before[name] != after[name] for name in before}
        self.discarded = True
        unchanged = self._unchanged()
        return dict(passed=all(changed.values()) and unchanged, discarded=True, systems_only=True,
            update=update, fp32_master_before=before, fp32_master_after=after,
            fp32_master_changed_by_parameter=changed, fixed_state_unchanged=unchanged,
            gradient_evidence='All gradients present; optimizer validates finite gradients; each first-step FP32 parameter master changes')

    @torch.no_grad()
    def development(self):
        self.block.eval(); totals = {}
        for start in range(0, len(self.data.development), self.config.microbatch):
            self._guard(); chosen = self.data.development[start:start+self.config.microbatch]
            for key, value in self.losses(self.data.batch(chosen, self.device)).items():
                totals[key] = totals.get(key, 0.) + float(value)*len(chosen)
        return {key: value/len(self.data.development) for key, value in totals.items()}

    def fit(self, path, *, event=None):
        if self.updates or self.finished or self.discarded:
            raise ValueError("Fit requires an untouched initializer; discard pilot trainers")
        path = Path(path); path.mkdir(parents=True, exist_ok=False)
        event = event or (lambda *_a, **_k: None)
        best = None; best_epoch = None; history = []
        effective = self.config.microbatch*self.config.accumulation
        total = self.config.epochs*math.ceil(len(self.data.training)/effective)
        torch.manual_seed(self.config.seed)
        for epoch in range(self.config.epochs+1):
            updates = []
            if epoch:
                order = list(self.data.training); random.Random(self.config.seed+epoch).shuffle(order)
                for start in range(0, len(order), effective):
                    lr = self.config.learning_rate*.5*(1+math.cos(math.pi*self.updates/total))
                    update = self.train_window(order[start:start+effective], learning_rate=lr)
                    updates.append(update); event('update', epoch=epoch, **update)
            metrics = self.development(); quality = None
            if self.quality is not None:
                self._guard(); before = evidence.fingerprint(self.block.named_parameters())
                quality = self.quality(self.stack)
                if before != evidence.fingerprint(self.block.named_parameters()) or not self._unchanged():
                    raise RuntimeError("Quality callback mutated the live prealignment model")
            report = dict(metrics=metrics, quality_diagnostic=quality)
            filename = path/f'epoch_{epoch:02d}_development.json'; evidence.base.write(filename, report)
            changed = best is None or metrics['loss'] < best['loss']
            row = dict(epoch=epoch, metrics=metrics, training=updates, quality_diagnostic=quality,
                development_sha256=evidence.base.sha(filename), checkpoint_updated=changed)
            if changed:
                pending = path/'selected_entrance.pt.pending'
                evidence.save(pending, dict(block=core.recursive_cpu(self.block.state_dict()),
                    layer_idx=self.block.layer_idx, normalization=self.normalization, config=asdict(self.config)))
                os.replace(pending, path/'selected_entrance.pt')
                best, best_epoch = metrics, epoch
                row['checkpoint_sha256'] = evidence.base.sha(path/'selected_entrance.pt')
            history.append(row); event('development', epoch=epoch, selected_epoch=best_epoch, **metrics)
        saved = torch.load(path/'selected_entrance.pt', map_location=self.device, weights_only=True)
        self.block.load_state_dict(saved['block'], strict=True)
        if self.updates != total or self.development() != best or not self._unchanged():
            raise RuntimeError("Prealignment budget, selected reload or fixed-state proof failed")
        self.finished = True
        result = dict(status='completed', config=asdict(self.config), completed_epochs=self.config.epochs,
            updates=self.updates, layer_idx=self.block.layer_idx, inherited_layers=list(self.stack.layer_indices[1:]),
            initial=history[0]['metrics'], selected=best, selected_epoch=best_epoch, initializer_selected=best_epoch == 0,
            history=history, normalization=self.normalization, strict_reload_exact=True, fixed_state_unchanged=True,
            fixed_state=self.fixed_before, selected_checkpoint_sha256=evidence.base.sha(path/'selected_entrance.pt'),
            optimizer_memory=self.optimizer.memory_stats(), selection='lowest_development_feature_loss_earliest_tie',
            feature_execution='differentiable BF16 entrance + fixed bridge; FP32 loss',
            task_quality_guaranteed=False, confirmed=False)
        evidence.base.write(path/'result.json', result)
        return result
