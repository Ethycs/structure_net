"""Hard native latents in forward; soft codebook surrogate in backward only."""
import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_lm1b_neighborhood_loss as original


def soft_latent(codec, logits, temperature=1.):
    levels = tuple(codec.vq.levels)
    if (temperature <= 0 or logits.ndim != 3 or logits.shape[1] != sum(levels)
            or len(codec.vq.embeddings) != len(levels)):
        raise ValueError("positive temperature and complete native hierarchy required")
    offset, total = 0, None
    for i, width in enumerate(levels):
        book = codec.vq.embeddings[i].weight
        if logits.shape[-1] < len(book):
            raise ValueError("all native book classes required")
        probabilities = F.softmax(logits[:, offset:offset + width, :len(book)].float() / temperature, dim=-1)
        value = probabilities @ book
        if i != len(levels) - 1:
            value = codec.vq._up_to_full(value, i, levels[-1])
        value = codec.vq._apply_quant_resi(value, i)
        total = value.float() if total is None else total + value.float()
        offset += width
    return total


def straight_through_latent(codec, codes, logits, temperature=1.):
    if codes.dtype != torch.long or codes.shape != logits.shape[:2]:
        raise ValueError("complete hard history and matching code logits required")
    with torch.no_grad():
        hard = codec.vq.idx_to_z_q(codes)
    soft = soft_latent(codec, logits, temperature)
    if hard.shape != soft.shape or not torch.isfinite(soft).all():
        raise ValueError("finite matching native latent shape required")
    # Parentheses make the forward correction exactly zero, not a rounded
    # hard+soft-soft cancellation. The backward derivative follows soft.
    return hard.detach() + (soft - soft.detach()).to(hard.dtype)


def word_logits(model, features, mask, history, concept_logits=None, temperature=1.):
    if concept_logits is None:
        concept_logits = model.concepts.teacher_logits(features, mask, history)
    latent = straight_through_latent(model.codec, history, concept_logits, temperature)
    with model.word_context(features, mask):
        words = model.codec.decoder(latent)
    return words, concept_logits


def backward_microbatch(model, features, mask, codes, words, neighbor, step, *, accumulation=1, temperature=1.):
    """Keep the clean/noisy losses and add source-only history with word gradients."""
    result = original.backward_microbatch(model, features, mask, codes, words, neighbor, step,
        accumulation=accumulation, self_history=False)
    was_training = model.training
    try:
        model.eval()
        precision = torch.autocast("cuda", enabled=False) if features.device.type == "cuda" else original.nullcontext()
        with torch.no_grad(), precision:
            history = model.concepts.generate_codes(features, mask)
    finally:
        model.train(was_training)
    with original.autocast(features):
        predicted_words, concept_logits = word_logits(model, features, mask, history, temperature=temperature)
    self_concept = original.level_cross_entropy(concept_logits, codes, model.concepts.generator.levels)
    self_word = F.cross_entropy(predicted_words.float().flatten(0, 1), words.flatten(), ignore_index=0)
    auxiliary = .25 * self_concept + self_word
    if not torch.isfinite(auxiliary):
        raise ValueError("nonfinite word-gradient auxiliary objective")
    auxiliary.backward()
    result.update(self_concept_loss=float(self_concept.detach()), self_word_loss=float(self_word.detach()),
        source_only_history_codes=history.detach().cpu().tolist(), temperature=temperature,
        auxiliary_concept_weight=.25, auxiliary_word_weight=1., word_gradient_into_concept_attention=True)
    return result
