"""All-block Torch distillation with bounded CPU gradient/FP32 Adam storage.

This executor deliberately does not use the Ampere-only BF16 Triton kernels.
It keeps the original BF16 arithmetic and block checkpointing while moving
completed leaf gradients to CPU. Experiment admission and data provenance are
owned by the caller; this module neither reserves hardware nor selects data.
"""
from dataclasses import replace
import math
import time

import torch
from torch.nn import functional as F

from . import jspace_3b_broad_recovery as broad
from . import jspace_3b_feature_recovery as feature
from . import jspace_qwen3_train as offload
from . import jspace_qwen_parallel_train as core
from .jspace_qwen_compact_stack import CompactQwenStack

CONFIG = replace(broad.CONFIG, microbatch=1, accumulation=16,
                 max_cpu_state_bytes=32 * 1024**3)


def memory_plan(stack, norm, readout, config=CONFIG):
    parameters = [*stack.parameters(), *norm.parameters()]
    count = sum(p.numel() for p in parameters)
    model = sum(t.numel()*t.element_size() for m in (stack, norm)
                for t in (*m.parameters(), *m.buffers()))
    gradient = sum(p.numel()*p.element_size() for p in parameters)
    return dict(trainable_scalars=count, cpu_master_moments_bytes=12*count,
                cpu_gradient_bank_bytes=gradient,
                model_and_readout_bytes=model+readout.numel()*readout.element_size(),
                snapshot_bytes=model,
                cpu_additional_bound=12*count+gradient+model+6*config.optimizer_chunk_bytes,
                persistent_gpu_parameter_gradients=False,
                excludes='CUDA context, transient activations, allocator fragmentation and OS page cache')


class JointTrainer:
    """Train every supplied block jointly; never move/replace live parameters."""

    def __init__(self, stack, norm, readout, rotary, dataset, label_ids, *,
                 targets=None, alpha=0., config=CONFIG, guard=None,
                 host_reserve_bytes=8*1024**3):
        self.stack, self.norm, self.readout, self.rotary = stack, norm, readout, rotary
        self.dataset, self.config, self.guard = dataset, config, guard
        self.device, self.dtype = readout.device, getattr(torch, config.dtype)
        if self.device.type == 'cuda' and guard is None:
            raise ValueError('CUDA joint training requires an active reservation guard')
        self._guard()
        if (not isinstance(stack, CompactQwenStack) or stack.packed or stack.stripped
                or config.dtype != 'bfloat16' or config.microbatch != 1
                or config.microbatch*config.accumulation != 16):
            raise ValueError('Unpacked BF16 stack and one-example microbatches, effective batch16 required')
        if (dataset.width != stack.width or readout.shape != (dataset.vocabulary, stack.width)
                or readout.dtype != self.dtype or next(stack.parameters()).device != self.device
                or not math.isfinite(alpha) or alpha < 0
                or (targets is None) != (alpha == 0)
                or targets is not None and set(targets.lookup) != set(dataset.training)):
            raise ValueError('Model, boundary, teacher or TRAIN feature targets disagree')
        if any(p.dtype != self.dtype or p.device != self.device
               for p in (*stack.parameters(), *norm.parameters())):
            raise ValueError('Initializer must explicitly store every learned parameter in BF16 on the model device')
        if any(b.position_count > 1 and max(dataset.lengths) > b.position_count for b in stack.blocks):
            raise ValueError('Input exceeds the fitted positional range')
        if not label_ids or any(not 0 <= t < dataset.vocabulary for t in label_ids):
            raise ValueError('Valid label-token IDs required')
        if any(not 0 <= c['answer'] < len(c['choices']) <= len(label_ids)
               or c['targets'][-1] != label_ids[c['answer']] for c in dataset.items):
            raise ValueError('Answer labels disagree with endpoint targets')
        self.label_ids = list(label_ids)
        self.feature_targets, self.alpha = targets, alpha
        stack.checkpoint_blocks = config.checkpoint_blocks
        stack.requires_grad_(True); norm.requires_grad_(True)
        readout.requires_grad_(False); rotary.requires_grad_(False)
        self.parameters = [*stack.parameters(), *norm.parameters()]
        self.parameter_ids = tuple(map(id, self.parameters))
        self.memory = memory_plan(stack, norm, readout, config)
        self.host_admission = offload.require_host_memory(
            self.memory['cpu_additional_bound'], reserve_bytes=host_reserve_bytes)
        self.optimizer = offload.CPUGradientAdamW(self.parameters,
            lr=config.learning_rate, weight_decay=0., chunk_bytes=config.optimizer_chunk_bytes,
            max_cpu_state_bytes=config.max_cpu_state_bytes)
        self.updates, self.finished = 0, False

    def _guard(self):
        if self.guard is not None:
            self.guard()

    def objectives(self, batch):
        hidden, coordinates = feature.forward_features(self.stack, batch['hidden'],
            position_ids=batch['position_ids'], attention_mask=batch['attention_mask'],
            position_embeddings=self.rotary(batch['hidden'], batch['position_ids']))
        selected = hidden[torch.arange(len(batch['indices']), device=self.device)[:, None],
                          batch['loss_positions']]
        logits = F.linear(self.norm(selected), self.readout).float()
        endpoint, kl, ce = core.output_objective(logits, batch['teacher'], batch['targets'], self.config)
        anchor = (endpoint.new_zeros(()) if self.feature_targets is None else
                  feature.feature_loss(coordinates, batch, self.feature_targets))
        return endpoint, kl, ce, anchor

    def train_window(self, indices, *, learning_rate=None):
        indices = list(indices)
        if (self.finished or not indices or len(indices) > 16
                or any(i not in self.dataset.training for i in indices)):
            raise ValueError('An unfinished TRAIN-only window of at most16 examples is required')
        if tuple(map(id, (*self.stack.parameters(), *self.norm.parameters()))) != self.parameter_ids:
            raise RuntimeError('Live parameter identities changed after gradient hook registration')
        self._guard(); self.stack.train(); self.norm.train(); self.optimizer.zero_grad()
        if learning_rate is not None:
            self.optimizer.param_groups[0]['lr'] = learning_rate
        started = time.monotonic(); transferred_before = self.optimizer.offload_bytes
        totals = dict(loss=0., endpoint_loss=0., kl=0., ce=0., feature_loss=0.)
        for index in indices:
            self._guard()
            batch = self.dataset.batch([index], device=self.device, dtype=self.dtype)
            endpoint, kl, ce, anchor = self.objectives(batch)
            loss = endpoint+self.alpha*anchor
            if not all(torch.isfinite(v) for v in (loss, endpoint, kl, ce, anchor)):
                raise FloatingPointError('Nonfinite all-block objective; optimizer has not stepped')
            (loss/len(indices)).backward()
            for key, value in zip(totals, (loss, endpoint, kl, ce, anchor)):
                totals[key] += float(value.detach())/len(indices)
            del batch, endpoint, kl, ce, anchor, loss
        received = self.optimizer.received
        present = [all(received.get(p, 0) == len(indices) for p in b.parameters())
                   for b in self.stack.blocks]
        if (not all(present) or any(received.get(p, 0) != len(indices) for p in self.norm.parameters())
                or any(p.grad is not None for p in self.parameters)):
            raise RuntimeError('Every leaf must have all microbatch gradients in its CPU bank')
        self._guard()
        update = self.optimizer.step(max_grad_norm=self.config.gradient_clip)
        gradient_bytes = self.optimizer.offload_bytes-transferred_before
        self.optimizer.zero_grad(); self.updates += 1
        return dict(**totals, alpha=self.alpha,
            normalization_energy=None if self.feature_targets is None else self.feature_targets.energy,
            cases=len(indices), update=self.updates, indices=indices,
            learning_rate=self.optimizer.param_groups[0]['lr'], gradients_present_by_block=present,
            gradient_device_to_host_bytes=gradient_bytes, optimizer=update,
            train_window_seconds=time.monotonic()-started)

    @torch.inference_mode()
    def development(self):
        self._guard(); self.optimizer.zero_grad(); self.stack.eval(); self.norm.eval()
        return score_unpacked(self.stack, self.norm, self.readout, self.rotary,
                              self.dataset, self.label_ids, self._guard, config=self.config)

    def snapshot(self):
        self._guard()
        offload.require_host_memory(self.memory['snapshot_bytes'], reserve_bytes=8*1024**3)
        return dict(stack=core.recursive_cpu(self.stack.state_dict()),
                    norm=core.recursive_cpu(self.norm.state_dict()))

    def close(self):
        self.optimizer.close()


@torch.inference_mode()
def score_unpacked(stack, norm, E, rotary, data, label_ids, guard=lambda: None, *, config=CONFIG):
    """Singleton Torch evaluation without a second GPU-resident packed copy."""
    if stack.packed or stack.stripped:
        raise ValueError('This Turing executor evaluates the actual unpacked Torch model')
    rows = []; totals = dict(loss=0., kl=0., ce=0.)
    for index in data.development:
        guard()
        batch = data.batch([index], device=E.device, dtype=E.dtype)
        output, _ = feature.forward_features(stack, batch['hidden'],
            position_ids=batch['position_ids'], attention_mask=batch['attention_mask'],
            position_embeddings=rotary(batch['hidden'], batch['position_ids']))
        selected = output[torch.arange(1, device=E.device)[:, None], batch['loss_positions']]
        logits = F.linear(norm(selected), E).float()
        loss, kl, ce = core.output_objective(logits, batch['teacher'], batch['targets'], config)
        if not all(torch.isfinite(v) for v in (loss, kl, ce)):
            raise FloatingPointError('Nonfinite all-block development objective')
        for key, value in zip(totals, (loss, kl, ce)):
            totals[key] += float(value)
        rows.append(broad.cache.sanity._record(data.items[index], norm(output[:, -1]), E, label_ids))
    if not rows:
        raise ValueError('Development rows required')
    correct = sum(r['prediction'] == r['answer'] for r in rows)
    return dict(**{k: v/len(rows) for k, v in totals.items()}, count=len(rows), correct=correct,
                accuracy=correct/len(rows),
                scoring='singleton BF16 unfused Torch; full model, original tied readout',
                by_family={f: dict(count=sum(r['family'] == f for r in rows),
                    correct=sum(r['family'] == f and r['prediction'] == r['answer'] for r in rows))
                    for f in sorted({r['family'] for r in rows})}), rows
