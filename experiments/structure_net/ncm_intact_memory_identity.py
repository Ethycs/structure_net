"""Exact native tensor identity for the zero-output inference state."""
import torch
from experiments.structure_net.ncm_intact_memory import NativeMemoryCorrection


class IdentityMemoryCorrection(NativeMemoryCorrection):
    def forward(self, hidden, memory):
        if not torch.is_grad_enabled() and torch.count_nonzero(self.output.weight).item() == 0:
            return hidden
        return super().forward(hidden, memory)
