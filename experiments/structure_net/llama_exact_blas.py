"""Prefer cuBLASLt for projections; test exactness rather than assume it."""
import torch
from experiments.structure_net.llama_tensorized_model import TensorizedLlama

CANDIDATES = ("cublaslt_bmm", "cublaslt_mm")


class BlasLlama(TensorizedLlama):
    def __init__(self, reference, variant, **_):
        if variant not in CANDIDATES or reference.replicas != 1:
            raise ValueError("known backend variant and one model required")
        if any(v["kind"] != "dense" for v in reference.matrices.values()):
            raise ValueError("original dense weights required")
        self.config, self.replicas = reference.config, reference.replicas
        self.rope, self.attention = reference.rope, reference.attention
        self.matrices, self.vectors = reference.matrices, reference.vectors
        self.variant, self.packing_verified = variant, True

    def multiply(self, x, weight):
        previous = torch.backends.cuda.preferred_blas_library()
        try:
            torch.backends.cuda.preferred_blas_library("cublaslt")
            if self.variant == "cublaslt_mm":
                return torch.mm(x[0], weight[0].T)[None, :]
            return torch.bmm(x, weight.transpose(1, 2))
        finally:
            torch.backends.cuda.preferred_blas_library(previous)

    def linear(self, x, name):
        return self.multiply(x, self.matrices[name + ".weight"]["cores"][0])

    def output(self, x):
        name = "model.embed_tokens.weight" if self.config.tie_word_embeddings else "lm_head.weight"
        return self.multiply(x, self.matrices[name]["cores"][0])
