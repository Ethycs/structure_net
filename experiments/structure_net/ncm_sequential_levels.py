"""Explicit native next-scale training with a frozen text codec.

Each forward sees one level and the cumulative reconstruction of earlier
levels. No target-code argument exists on the source-only generation path.
"""
import torch
import torch.nn.functional as F


def configure(model, training):
    model.requires_grad_(False)
    for module in (model.concepts.generator, model.concepts.source_memory, model.concepts.adapters):
        module.requires_grad_(True)
    model.train(training)
    model.concepts.generator.train(training)
    model.codec.eval().requires_grad_(False)
    model.word_memory.eval()
    model.word_adapters.eval()
    return model


def level_logits(model, features, mask, previous_reconstruction, level):
    g = model.concepts.generator
    if (not g.query_level0 or g.l0_ar or g.prefix_len != 1 or g.use_ctx_encoder
            or g.config.attn_pattern != 'block_diagonal' or not 0 <= level < len(g.levels)):
        raise ValueError('native query-root block-diagonal LM1B geometry required')
    batch, length = len(features), g.levels[level]
    with torch.no_grad():
        ids = torch.full((batch, 1), 101, dtype=torch.long, device=features.device)
        prefix = model.codec.encoder.wte(ids)
    prefix = g.transformer.wte(prefix)
    if level == 0:
        if previous_reconstruction is not None:
            raise ValueError('root must be predicted without target codes')
        latent = g.build_l0_query_input(batch)
    else:
        if previous_reconstruction is None or previous_reconstruction.shape != (batch, length, model.codec.embed_dim):
            raise ValueError('only the previous cumulative reconstruction may condition a level')
        latent = previous_reconstruction.detach()
    if g.use_post_upsample_conv:
        latent = g._apply_post_upsample_conv_single(latent, level)
    x = g.transformer.wte(latent)
    offset = sum(g.levels[:level])
    if not g.use_rope:
        prefix = prefix + g.wpe(g.wp_prefix)
        x = x + g.wpe(g.wp_1L[:, offset:offset + length])
    if g.config.use_level_encoding:
        x = x + g.transformer.wle(g.lvl_1L[:, offset:offset + length])
    x = torch.cat((prefix, x), dim=1)
    attention = torch.zeros(1, 1, 1 + length, 1 + length, device=x.device)
    attention[:, :, :1, 1:] = -torch.inf
    rope = (getattr(g, f'rope_cos_gen_{level}'), getattr(g, f'rope_sin_gen_{level}')) if g.use_rope else None
    with model.concepts.source_context(features, mask):
        for block in g.transformer.blocks:
            x = block(x, attn_mask=attention, rope_cos_sin=rope)
    return g.lm_heads[level](g.transformer.ln_f(x)[:, 1:])


@torch.no_grad()
def advance(codec, cumulative, codes, level):
    return codec.vq.get_next_autoregressive_input(level, cumulative,
        codec.vq.embeddings[level](codes))


@torch.no_grad()
def generate(model, features, mask):
    if model.training:
        raise ValueError('source-only generation requires eval mode')
    levels = model.concepts.generator.levels
    cumulative = features.new_zeros(len(features), levels[-1], model.codec.embed_dim)
    previous, output = None, []
    for level in range(len(levels)):
        codes = level_logits(model, features, mask, previous, level).argmax(-1)
        output.append(codes)
        cumulative, previous = advance(model.codec, cumulative, codes, level)
    return torch.cat(output, dim=1)


def corrupt(part, level, levels, vocabulary, rng, enabled):
    if not enabled:
        return part.detach().clone(), 0
    probability = .25 - .20 * level / max(1, levels - 1)
    changed = torch.rand(part.shape, generator=rng) < probability
    replacement = torch.randint(vocabulary, part.shape, generator=rng)
    output = torch.where(changed, replacement, part.detach().cpu()).to(part.device)
    return output, int((output != part).sum())


def backward_example(model, features, mask, gold, rng, scaler, accumulation, noisy):
    levels = model.concepts.generator.levels
    if (not model.training or accumulation < 1 or gold.dtype != torch.long
            or gold.shape != (len(features), sum(levels))):
        raise ValueError('complete TRAIN hierarchy, training mode and positive accumulation required')
    cumulative = features.new_zeros(len(features), levels[-1], model.codec.embed_dim)
    previous, offset, losses, changes = None, 0, [], 0
    for level, length in enumerate(levels):
        with torch.autocast('cuda', dtype=torch.float16, enabled=features.is_cuda):
            logits = level_logits(model, features, mask, previous, level)
        target = gold[:, offset:offset + length]
        loss = F.cross_entropy(logits.float().flatten(0, 1), target.flatten())
        if not torch.isfinite(loss):
            raise ValueError('nonfinite sequential level loss')
        scaled = loss / (len(levels) * accumulation)
        (scaler.scale(scaled) if scaler is not None else scaled).backward()
        losses.append(float(loss.detach()))
        history, changed = corrupt(target, level, len(levels), logits.shape[-1], rng, noisy)
        changes += changed
        cumulative, previous = advance(model.codec, cumulative, history, level)
        offset += length
    return {'level_CE': losses, 'mean_CE': sum(losses) / len(losses),
        'history_corruptions': changes, 'noisy_history': noisy}
