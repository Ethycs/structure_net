"""Batch independent teacher-forced levels; retain native earlier-level context."""
import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_sequential_levels as sequential


def history(gold, levels, vocabulary, rng, noisy):
    if gold.dtype != torch.long or gold.ndim != 2 or gold.shape[1] != sum(levels):
        raise ValueError('complete native hierarchy required')
    parts, offset, changes = [], 0, 0
    for level, length in enumerate(levels):
        part, count = sequential.corrupt(gold[:, offset:offset+length], level, len(levels), vocabulary, rng, noisy)
        parts.append(part); changes += count; offset += length
    return torch.cat(parts, dim=1), changes


def losses(logits, gold, levels):
    if logits.shape[:2] != gold.shape or gold.shape[1] != sum(levels):
        raise ValueError('complete aligned prediction and target hierarchy required')
    values, offset = [], 0
    for length in levels:
        values.append(F.cross_entropy(logits[:, offset:offset+length].float().flatten(0, 1), gold[:, offset:offset+length].flatten()))
        offset += length
    return torch.stack(values)


def backward_example(model, features, mask, gold, rng, scaler, accumulation, noisy):
    if not model.training or accumulation < 1:
        raise ValueError('training mode and positive accumulation required')
    levels = model.concepts.generator.levels
    prior_codes, changes = history(gold, levels, 8192, rng, noisy)
    # Native idx_to_ncm_input includes only strictly earlier cumulative levels.
    # Native block-diagonal attention isolates level queries and the CLS prefix.
    with torch.autocast('cuda', dtype=torch.float16, enabled=features.is_cuda):
        logits = model.concepts.teacher_logits(features, mask, prior_codes)
    terms = losses(logits, gold, levels)
    loss = terms.mean()
    if not torch.isfinite(loss):
        raise ValueError('nonfinite packed next-scale loss')
    scaled = loss / accumulation
    (scaler.scale(scaled) if scaler is not None else scaled).backward()
    return {'level_CE': terms.detach().cpu().tolist(), 'mean_CE': float(loss.detach()),
        'history_corruptions': changes, 'noisy_history': noisy}


def gradient_parity(model, features, mask, gold, noisy=False):
    """Compare every native trainable gradient with explicit FP32 level steps."""
    sequential.configure(model, False)
    levels = model.concepts.generator.levels
    prior_codes, _ = history(gold, levels, 8192, torch.Generator().manual_seed(160142), noisy)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    previous, offset, expected_logits = None, 0, []
    cumulative = features.new_zeros(len(features), levels[-1], model.codec.embed_dim)
    for level, length in enumerate(levels):
        output = sequential.level_logits(model, features, mask, previous, level)
        loss = F.cross_entropy(output.flatten(0, 1), gold[:, offset:offset+length].flatten()) / len(levels)
        loss.backward()
        expected_logits.append(output.detach().cpu())
        cumulative, previous = sequential.advance(model.codec, cumulative, prior_codes[:, offset:offset+length], level)
        offset += length
    expected_gradients = {n: p.grad.detach().cpu().clone() if p.grad is not None else None for n, p in params.items()}
    model.zero_grad(set_to_none=True)
    actual_logits = model.concepts.teacher_logits(features, mask, prior_codes)
    losses(actual_logits, gold, levels).mean().backward()
    squared_error = squared_norm = 0.
    max_difference = 0.
    for name, param in params.items():
        expected = expected_gradients[name]
        if expected is None:
            if param.grad is not None:
                raise ValueError('gradient presence differs:' + name)
            continue
        if param.grad is None:
            raise ValueError('missing packed gradient:' + name)
        actual = param.grad.detach().cpu()
        delta = actual - expected
        squared_error += float(delta.double().square().sum())
        squared_norm += float(expected.double().square().sum())
        max_difference = max(max_difference, float(delta.abs().max()))
    expected = torch.cat(expected_logits, dim=1)
    actual = actual_logits.detach().cpu()
    relative = (squared_error / max(squared_norm, 1e-30)) ** .5
    result = {'noisy_history': noisy, 'relative_gradient_L2_error': relative,
        'max_abs_gradient_difference': max_difference,
        'logits_allclose': torch.allclose(actual, expected, rtol=1e-4, atol=1e-4),
        'argmax_disagreements': int((actual.argmax(-1) != expected.argmax(-1)).sum()),
        'trainable_parameters_compared': sum(p.numel() for p in params.values())}
    result['passed'] = relative <= 5e-5 and max_difference <= 1e-4 and result['logits_allclose'] and result['argmax_disagreements'] == 0
    model.zero_grad(set_to_none=True)
    if not result['passed']:
        raise ValueError('packed/explicit native objective differs:' + str(result))
    return result
