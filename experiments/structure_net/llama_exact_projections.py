"""Falsifiable exact projection optimizations; acceptance requires bitwise tests."""
import torch

from experiments.structure_net.llama_tensorized_model import TensorizedLlama
from experiments.structure_net.llama_cuda_graph import bitwise_tree

CANDIDATES = ("mm", "packed_bmm", "packed_mm", "parallel2", "parallel3")


class ProjectionLlama(TensorizedLlama):
    """Share unchanged weights, and optionally pack independent output rows.

    Single request at a time, like the graph wrapper. Floating-point equality
    of a different GEMM shape is deliberately *not* assumed by this class.
    """

    def __init__(self, reference, variant, share_packed_reference=False):
        if variant not in CANDIDATES or reference.replicas != 1:
            raise ValueError("known variant and exactly one model required")
        if any(s["kind"] != "dense" for s in reference.matrices.values()):
            raise ValueError("original dense weights required")
        self.config, self.replicas = reference.config, 1
        self.rope, self.attention = reference.rope, reference.attention
        self.vectors = reference.vectors.copy()
        self.matrices = reference.matrices.copy()
        self.variant = variant
        self.pending = {}
        self.groups = {}
        self.packing_verified = True
        self.streams = []
        if variant.startswith("parallel"):
            if next(iter(self.matrices.values()))["cores"][0].device.type != "cuda":
                raise ValueError("parallel variants require CUDA")
            self.streams = [torch.cuda.Stream() for _ in range(int(variant[-1]) - 1)]
            for index in range(self.config.num_hidden_layers):
                prefix = f"model.layers.{index}"
                for suffixes in (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                                 ("mlp.gate_proj", "mlp.up_proj")):
                    names = [prefix + "." + s for s in suffixes]
                    self.groups[names[0]] = (None, names, None)
        if variant.startswith("packed"):
            for index in range(self.config.num_hidden_layers):
                prefix = f"model.layers.{index}"
                for suffixes in (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                                 ("mlp.gate_proj", "mlp.up_proj")):
                    names = [prefix + "." + s for s in suffixes]
                    values = [self.matrices.pop(n + ".weight")["cores"][0] for n in names]
                    packed = torch.cat(values, dim=1)
                    sizes = [v.shape[1] for v in values]
                    self.packing_verified &= all(bitwise_tree(a, b) for a, b in zip(packed.split(sizes, dim=1), values))
                    key = names[0] + ".packed.weight"
                    self.matrices[key] = {"kind": "dense", "shape": list(packed.shape[1:]), "cores": [packed]}
                    self.groups[names[0]] = (key, names, sizes)
                    if share_packed_reference:
                        for name, part, original in zip(names, packed.split(sizes, dim=1), values):
                            view = part.as_strided(original.shape, original.stride())
                            source = reference.matrices[name + ".weight"]
                            reference.matrices[name + ".weight"] = {**source, "cores": [view]}
        if not self.packing_verified or self.storage_bytes() != reference.storage_bytes():
            raise ValueError("packing changed original parameter bits or byte count")

    def multiply(self, x, weight):
        if self.variant in ("mm", "packed_mm"):
            return torch.mm(x[0], weight[0].T)[None, :]
        return torch.bmm(x, weight.transpose(1, 2))

    def linear(self, x, name):
        if name in self.pending:
            return self.pending.pop(name)
        if name in self.groups:
            key, names, sizes = self.groups[name]
            if key is None:
                current = torch.cuda.current_stream()
                for stream in self.streams:
                    stream.wait_stream(current)
                for index, other in enumerate(names[1:]):
                    with torch.cuda.stream(self.streams[index % len(self.streams)]):
                        self.pending[other] = self.multiply(x, self.matrices[other + ".weight"]["cores"][0])
                first = self.multiply(x, self.matrices[name + ".weight"]["cores"][0])
                for stream in self.streams:
                    current.wait_stream(stream)
                return first
            parts = self.multiply(x, self.matrices[key]["cores"][0]).split(sizes, dim=-1)
            self.pending.update(zip(names[1:], parts[1:]))
            return parts[0]
        return self.multiply(x, self.matrices[name + ".weight"]["cores"][0])

    def output(self, x):
        key = "model.embed_tokens.weight" if self.config.tie_word_embeddings else "lm_head.weight"
        return self.multiply(x, self.matrices[key]["cores"][0])

    def __call__(self, *args, **kwargs):
        if self.pending:
            raise RuntimeError("unconsumed projections from an earlier call")
        try:
            return super().__call__(*args, **kwargs)
        finally:
            self.pending.clear()
