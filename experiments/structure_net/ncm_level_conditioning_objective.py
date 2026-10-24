"""New objective instance for an explicitly scoped future native NCM run."""
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_level_conditioning_corruption as corruption
from experiments.structure_net import ncm_lm1b_word_gradient as words


def autocast(features, dtype):
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('explicit FP16 or BF16 training autocast required')
    return torch.autocast('cuda', dtype=dtype) if features.device.type == 'cuda' else nullcontext()


def backward_microbatch(model, features, mask, codes, targets, neighbor, *, accumulation=1,
        self_history=True, dtype=torch.bfloat16, scaler=None, noise=corruption.Config()):
    if (not model.training or accumulation < 1 or features.dtype != torch.float32 or targets.dtype != torch.long
            or targets.ndim != 2 or len(targets) != len(codes)):
        raise ValueError('training mode, original FP32 source and complete integer targets required')
    levels = tuple(model.concepts.generator.levels)
    vocabularies = tuple(len(book.weight) for book in model.codec.vq.embeddings)
    history, proof = corruption.corrupt(codes.detach().cpu(), levels, vocabularies, neighbor, noise)

    def backward(value):
        if not torch.isfinite(value):
            raise ValueError('nonfinite level-conditioning objective')
        (value if scaler is None else scaler.scale(value)).backward()

    result = {'corruption': proof, 'accumulation': accumulation, 'self_history_used': self_history,
        'autocast_dtype': str(dtype), 'clean_word_weight': .7, 'noisy_word_weight': .3,
        'clean_and_noisy_concept_weight': .5}
    for name, conditioning, cw, ww in (('clean', codes, .5, .7), ('noisy', history.to(codes.device), .5, .3)):
        with autocast(features, dtype):
            output = model(features, mask, conditioning)
        concept = words.original.level_cross_entropy(output['concept_logits'], codes, levels)
        word = F.cross_entropy(output['word_logits'].float().flatten(0, 1), targets.flatten(), ignore_index=0)
        backward((cw * concept + ww * word) / accumulation)
        result[name + '_concept_loss'] = float(concept.detach())
        result[name + '_word_loss'] = float(word.detach())
        del output, concept, word
    if self_history:
        mode = model.training
        try:
            model.eval()
            precision = torch.autocast('cuda', enabled=False) if features.device.type == 'cuda' else nullcontext()
            with torch.no_grad(), precision:
                generated = model.concepts.generate_codes(features, mask)
        finally:
            model.train(mode)
        with autocast(features, dtype):
            predicted_words, logits = words.word_logits(model, features, mask, generated, temperature=1.)
        concept = words.original.level_cross_entropy(logits, codes, levels)
        word = F.cross_entropy(predicted_words.float().flatten(0, 1), targets.flatten(), ignore_index=0)
        # Same source-only-history auxiliary weight as native655, once per update.
        backward(.25 * concept + word)
        result.update(self_concept_loss=float(concept.detach()), self_word_loss=float(word.detach()),
            source_only_history_codes=generated.detach().cpu().tolist(), word_gradient_into_concepts=True)
    if corruption.digest(codes.detach().cpu()) != proof['clean_target_sha256']:
        raise ValueError('clean target changed during backward')
    return result
