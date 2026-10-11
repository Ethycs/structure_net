"""Exact residual correction controls; explicitly retain the dense oracle cost."""
import torch
from structure_net.components.layers import model_major_tensor as tensor
from experiments.structure_net.llama_tensorized_model import TensorizedLlama


def bits(value):
    integer = {torch.float16: torch.int16, torch.bfloat16: torch.int16,
               torch.float32: torch.int32, torch.float64: torch.int64}[value.dtype]
    return value.contiguous().view(integer)


def exact_output_correction(predicted, original):
    """A residual network with a dense oracle, including exact rounding repair.

Keeping the original result is deliberate: this is an expensive lossless
control, not an assertion that a smaller exact correction network exists.
"""
    residual = original.float()-predicted.float()
    corrected = (predicted.float()+residual).to(original.dtype)
    return torch.where(bits(corrected) == bits(original), corrected, original)


def matrix_residual(original, prediction):
    residual = original.float()-prediction
    recovered = (prediction+residual).to(original.dtype)
    indices = (bits(recovered).flatten() != bits(original).flatten()).nonzero().flatten()
    repairs = original.flatten()[indices]
    recovered.flatten()[indices] = repairs
    if not torch.equal(bits(recovered), bits(original)):
        raise RuntimeError("lossless weight reconstruction failed")
    return residual, indices.to(torch.int32), repairs, recovered


class ExactResidualLlama(TensorizedLlama):
    def __init__(self, original, factors, replicas=1, device="cuda", attention="flash"):
        super().__init__(original, replicas, device, attention)
        self.factors = {name: tensor.pack([spec]*replicas, device=device) for name, spec in factors["matrices"].items()}

    def reference_view(self):
        """Alternative execution of this same model, not a co-resident replica."""
        result = object.__new__(TensorizedLlama)
        for name in ("config", "replicas", "matrices", "vectors", "rope", "attention"):
            setattr(result, name, getattr(self, name))
        return result

    def slice(self, index):
        result = super().slice(index)
        result.factors = {name: {**spec, "cores": [c[index:index+1] for c in spec["cores"]]} for name, spec in self.factors.items()}
        return result

    def storage_bytes(self):
        return super().storage_bytes() + sum(c.numel()*c.element_size() for s in self.factors.values() for c in s["cores"])

    def linear(self, x, name):
        prediction = tensor.apply(x, self.factors[name+".weight"])
        original = super().linear(x, name)
        return exact_output_correction(prediction, original)

    def embed(self, ids):
        prediction = tensor.embedding(ids, self.factors["model.embed_tokens.weight"])
        return exact_output_correction(prediction, super().embed(ids))

    def output(self, x):
        name = "model.embed_tokens.weight" if self.config.tie_word_embeddings else "lm_head.weight"
        prediction = tensor.apply(x, self.factors[name])
        return exact_output_correction(prediction, super().output(x))
