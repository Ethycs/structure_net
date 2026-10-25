"""Task-word gradients through frozen decoding into the code corrector only."""
import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_attention_code_repair as repair
from experiments.structure_net import ncm_lm1b_word_gradient as geometry


def loss(renderer, scores, target_codes, target_words, features, mask,
        *, latent_weight=.05, word_weight=.5):
    if renderer.training or any(p.requires_grad for p in renderer.parameters()):
        raise ValueError('the complete original renderer must stay frozen and in eval mode')
    levels = tuple(renderer.codec.vq.levels)
    logits = torch.cat(scores, dim=1)
    code = repair.level_loss(scores, target_codes, levels)
    soft = geometry.soft_latent(renderer.codec, logits)
    hard_codes = logits.detach().argmax(-1)
    with torch.no_grad():
        hard = renderer.codec.vq.idx_to_z_q(hard_codes)
        desired = renderer.codec.vq.idx_to_z_q(target_codes)
    latent = hard.detach() + (soft - soft.detach()).to(hard.dtype)
    mse = (soft - desired).square().mean()
    with renderer.word_context(features.detach(), mask):
        words = renderer.codec.decoder(latent)
    word = F.cross_entropy(words.float().flatten(0, 1), target_words.flatten(), ignore_index=0)
    total = code + latent_weight * mse + word_weight * word
    if not torch.isfinite(total):
        raise ValueError('finite correction code/latent/word objective required')
    return total, {'code_CE': float(code.detach()), 'soft_latent_MSE': float(mse.detach()),
        'hard_forward_word_CE': float(word.detach()), 'latent_weight': latent_weight, 'word_weight': word_weight}, hard_codes
