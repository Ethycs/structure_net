"""Enable native NCM fine-tuning while retaining the frozen target-code geometry."""
from types import MethodType

import torch


def teacher_logits(self, features, mask, codes):
    if codes.dtype != torch.long or codes.shape != (len(features), sum(self.generator.levels)):
        raise ValueError("complete native target-code hierarchy required")
    with torch.no_grad():
        ids = torch.full((len(features), 1), 101, device=features.device, dtype=torch.long)
        prefix = self.codec.encoder.wte(ids)
        earlier = self.codec.vq.idx_to_ncm_input(codes).to(prefix.dtype)
    # The old attention-only head built this learned native query under no_grad.
    # Native fine-tuning must retain its gradient, while codec inputs stay frozen.
    root = self.generator.build_l0_query_input(len(features), dtype=prefix.dtype)
    inputs = torch.cat((prefix, root, earlier), dim=1)
    with self.source_context(features, mask):
        return self.generator(inputs)


def enable(model):
    """Change this fresh instance's gradient/mode routing; never change weights."""
    if getattr(model, "_native_finetuning_enabled", False):
        raise ValueError("native fine-tuning already enabled on this instance")
    protected = {id(p) for module in (model.codec.encoder, model.codec.vq) for p in module.parameters()}
    for module in (model.concepts.generator, model.codec.decoder):
        for p in module.parameters():
            p.requires_grad_(id(p) not in protected)
    for module in (model.codec.encoder, model.codec.vq):
        module.requires_grad_(False)
    old_train = model.train

    def train(instance, mode=True):
        old_train(mode)
        instance.concepts.generator.train(mode)
        instance.codec.decoder.train(mode)
        instance.codec.encoder.eval()
        instance.codec.vq.eval()
        return instance

    model.concepts.teacher_logits = MethodType(teacher_logits, model.concepts)
    model.train = MethodType(train, model)
    model._native_finetuning_enabled = True
    model.train(model.training)
    if any(p.requires_grad for module in (model.codec.encoder, model.codec.vq) for p in module.parameters()):
        raise ValueError("native encoder and VQ geometry must remain frozen")
    return model
