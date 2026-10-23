"""Inference-only removal of repeated GPU zero tests; same word operations."""
from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F

from experiments.structure_net.ncm_native_query_cache import QueryLoraLinear


def _static_forward(self, x):
    if torch.is_grad_enabled():
        raise ValueError("static word-bank runtime is inference-only")
    if self.bank == "coarse":
        return QueryLoraLinear.forward(self, x)
    if self.bank != "word":
        raise ValueError("unknown word bank")
    if (self.word_a._version, self.word_b._version) != self._static_word_versions:
        raise ValueError("word parameters changed after static runtime installation")
    if not self.enabled or self._static_word_zero:
        return self.base(x)
    value = self.base(x)
    return value + F.linear(F.linear(x.float(), self.word_a), self.word_b).to(value.dtype)


@contextmanager
def static_word_runtime(backbone):
    if torch.is_grad_enabled():
        raise ValueError("install static word runtime under no_grad/inference_mode")
    previous = []
    try:
        for module in backbone.adapters:
            if hasattr(module, "_static_word_versions"):
                raise ValueError("nested static runtime forbidden")
            own = "forward" in module.__dict__
            previous.append((module, own, module.forward))
            module._static_word_versions = (module.word_a._version, module.word_b._version)
            module._static_word_zero = torch.count_nonzero(module.word_b).item() == 0
            module.forward = MethodType(_static_forward, module)
        yield
    finally:
        for module, own, forward in reversed(previous):
            if own:
                module.forward = forward
            else:
                del module.__dict__["forward"]
            del module._static_word_versions
            del module._static_word_zero
