"""Detached source-only histories and residual targets for their visited states."""
import torch

from experiments.structure_net import ncm_sequential_levels as sequential


@torch.no_grad()
def targets_for_history(vq, target_latent, history):
    """Oracle TRAIN labels conditional on the actual previous-code history.

    This differs from encoding a clean suffix: every state advances with the
    historical prediction, never the new oracle target. No gradient reaches
    the frozen target encoder or quantizer through this operation.
    """
    levels = vq.levels
    if (target_latent.ndim != 3 or target_latent.shape[1:] != (levels[-1], vq.embed_dim)
            or target_latent.dtype != torch.float32 or not torch.isfinite(target_latent).all()
            or history.dtype != torch.long or history.device != target_latent.device
            or history.shape != (len(target_latent), sum(levels))):
        raise ValueError('complete finite FP32 target latent and native history required')
    residual = target_latent.detach().clone()
    labels, offset = [], 0
    for level, length in enumerate(levels):
        emb = vq.embeddings[level]
        part = history[:, offset:offset+length]
        if part.min() < 0 or part.max() >= emb.num_embeddings:
            raise ValueError('history index outside its level codebook')
        value = vq._down_for_quant(residual, level, length) if level < len(levels)-1 else residual
        distances = vq._distances(value.reshape(-1, vq.embed_dim), emb.weight)
        if not torch.isfinite(distances).all():
            raise ValueError('nonfinite residual distances')
        labels.append(distances.argmin(-1).view(len(history), length))
        contribution = emb(part)
        if level < len(levels)-1:
            contribution = vq._up_to_full(contribution, level, levels[-1])
        residual -= vq._apply_quant_resi(contribution, level)
        offset += length
    return torch.cat(labels, dim=1)


def rollout_history(model, features, mask):
    """Run the unmodified source-only policy, then restore all module flags."""
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        with torch.no_grad(), torch.autocast(features.device.type, enabled=False):
            return sequential.generate(model, features, mask).detach()
    finally:
        # Restore mixed modes (trainable generator, frozen eval codec) exactly;
        # model.train(True) alone would also enable dropout in frozen modules.
        for module, mode in modes.items():
            module.training = mode
