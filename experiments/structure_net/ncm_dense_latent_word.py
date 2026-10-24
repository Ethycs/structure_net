"""Dense word-latent supervision with the original NCM decoder kept frozen.

Prediction receives source features and prior predicted concepts only. Refined
target latents and words enter the separate training objective, never prediction.
This module supplies mechanisms, not a trained summary model or quality claim.
"""
import torch
import torch.nn.functional as F


def prepare_branch(branch):
    """Restore the original decoder and remove dropout from the learned branch."""
    codec = branch.generator.vqvae_local[0]
    original = getattr(codec, "original", codec)
    if original.training or any(p.requires_grad for p in original.parameters()):
        raise ValueError("original codec must stay explicitly frozen and eval")
    if {id(p) for p in branch.word_decoder.parameters()} & {id(p) for p in original.parameters()}:
        raise ValueError("word decoder must be a separate copy of the original")
    branch.word_decoder.load_state_dict(original.decoder.state_dict(), strict=True)
    branch.word_decoder.requires_grad_(False).eval()
    for net in (branch.generator.transformer, branch.word_decoder):
        for module in net.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
            value = getattr(module, "dropout", None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                module.dropout = 0.
    if any(p.requires_grad for p in branch.generator.lm_heads.parameters()) or branch.generator.l0_queries.requires_grad:
        raise ValueError("unused coarse heads and root queries must stay frozen")
    return branch


def train_mode(branch, training=True):
    branch.train(training)
    branch.word_decoder.eval()


def word_latent(branch, features, source_mask, word_conditioning, source_ids):
    """Predict the complete latent frame without receiving target information."""
    g = branch.generator
    codec = g.vqvae_local[0]
    if word_conditioning.shape != (len(features), g.levels[-1], codec.embed_dim):
        raise ValueError("complete predicted word-level conditioning required")
    source = branch.source_head.source_latents(features, source_mask, source_ids)
    conditioning = word_conditioning.detach().to(source.dtype)
    hidden = branch._level_hidden(g.transformer.wte(source), source_mask, conditioning,
        len(g.levels) - 1, sum(g.levels[:-1]))
    return conditioning.float() + branch.word_projection(hidden).float()


def decoder_logits(branch, latent):
    """Keep the decoder in FP32 while allowing input gradients to the head."""
    if branch.word_decoder.training or any(p.requires_grad for p in branch.word_decoder.parameters()):
        raise ValueError("original word decoder must stay frozen and eval")
    if latent.ndim != 3 or latent.shape[1] != branch.width:
        raise ValueError("complete latent frame required")
    batch, width, dim = latent.shape
    with torch.autocast(latent.device.type, enabled=False):
        logits = branch.word_decoder(latent.float().reshape(batch * branch.blocks, branch.block_length, dim))
    logits = logits.reshape(batch, width, logits.shape[-1])
    if branch.word_vocab_limit is not None:
        logits = logits[..., :branch.word_vocab_limit]
    if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
        raise ValueError("finite FP32 decoder logits required")
    return logits


def prediction(branch, features, source_mask, word_conditioning, source_ids):
    latent = word_latent(branch, features, source_mask, word_conditioning, source_ids)
    return latent, decoder_logits(branch, latent)


def objective(latent, logits, teacher_latent, target_words, *, latent_weight=1., word_weight=.1, pad=50257):
    """Supervise full latent geometry and nonpadding words with detached labels."""
    if (latent.shape != teacher_latent.shape or latent.shape[:2] != target_words.shape
            or logits.shape[:2] != target_words.shape or latent.device != teacher_latent.device
            or latent_weight <= 0 or word_weight <= 0 or not target_words.ne(pad).any(-1).all()
            or not torch.isfinite(teacher_latent).all()):
        raise ValueError("complete finite detached-label frames and positive loss weights required")
    dense = F.mse_loss(latent.float(), teacher_latent.detach().float())
    word = F.cross_entropy(logits.float().flatten(0, 1), target_words.flatten(), ignore_index=pad)
    loss = latent_weight * dense + word_weight * word
    if not torch.isfinite(loss):
        raise ValueError("finite dense latent and word objective required")
    return loss, {"latent_MSE": float(dense.detach()), "word_CE": float(word.detach())}
