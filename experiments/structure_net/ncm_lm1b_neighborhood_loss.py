"""Clean, corrupted-history and source-only self-history losses for native NCM."""
from contextlib import nullcontext
import hashlib

import torch
import torch.nn.functional as F


def level_cross_entropy(logits, targets, levels):
    if (logits.ndim!=3 or targets.shape!=logits.shape[:2] or targets.dtype!=torch.long
            or not levels or any(v<1 for v in levels) or sum(levels)!=targets.shape[1]):
        raise ValueError("full target hierarchy with positive native levels required")
    terms=[];start=0
    for width in levels:
        terms.append(F.cross_entropy(logits[:,start:start+width].float().flatten(0,1),targets[:,start:start+width].flatten()))
        start+=width
    # Equal level means prevent the one-position root being drowned by fine levels.
    return torch.stack(terms).mean()


def corrupt_history(codes, step, generator, *, vocabulary=8192, root_width=1, maximum_rate=.25, ramp_updates=1024):
    if (codes.device.type!="cpu" or codes.dtype!=torch.long or codes.ndim!=2 or step<1
            or not 0<=maximum_rate<=1 or ramp_updates<1 or not 0<root_width<codes.shape[1]
            or (codes<0).any() or (codes>=vocabulary).any()):
        raise ValueError("complete CPU-native concept codes and valid corruption schedule required")
    maximum=maximum_rate*min(step/ramp_updates,1.)
    rates=torch.rand((len(codes),1),generator=generator)*maximum
    selected=torch.rand(codes.shape,generator=generator)<rates
    selected[:,:root_width]=False
    replacements=torch.randint(vocabulary,codes.shape,generator=generator)
    noisy=torch.where(selected,replacements,codes)
    return noisy,{"maximum_rate":maximum,"per_example_rates":rates.flatten().tolist(),
        "selected_positions":int(selected.sum()),"changed_positions":int((noisy!=codes).sum()),
        "root_unchanged":True,"corrupted_codes_sha256":hashlib.sha256(noisy.contiguous().numpy().tobytes()).hexdigest()}


def autocast(features):
    return torch.autocast("cuda",dtype=torch.bfloat16) if features.device.type=="cuda" else nullcontext()


def backward_microbatch(model,features,mask,codes,words,generator,step,*,accumulation=4,self_history=False):
    if not model.training or accumulation<1 or words.ndim!=2 or len(words)!=len(codes) or features.dtype!=torch.float32:
        raise ValueError("training mode and complete word/concept batch required")
    levels=model.concepts.generator.levels
    scale=1/accumulation
    with autocast(features):out=model(features,mask,codes)
    clean_concept=level_cross_entropy(out["concept_logits"],codes,levels)
    clean_word=F.cross_entropy(out["word_logits"].float().flatten(0,1),words.flatten(),ignore_index=0)
    clean=.5*clean_concept+.7*clean_word
    if not torch.isfinite(clean):raise ValueError("nonfinite clean neighborhood loss")
    (scale*clean).backward()
    result={"clean_concept_loss":float(clean_concept.detach()),"clean_word_loss":float(clean_word.detach())}
    del out,clean,clean_concept,clean_word
    noisy,proof=corrupt_history(codes.detach().cpu(),step,generator,root_width=levels[0])
    with autocast(features):out=model(features,mask,noisy.to(codes.device))
    noisy_concept=level_cross_entropy(out["concept_logits"],codes,levels)
    noisy_word=F.cross_entropy(out["word_logits"].float().flatten(0,1),words.flatten(),ignore_index=0)
    noisy_loss=.5*noisy_concept+.3*noisy_word
    if not torch.isfinite(noisy_loss):raise ValueError("nonfinite corrupted-history loss")
    (scale*noisy_loss).backward()
    result.update(noisy_concept_loss=float(noisy_concept.detach()),noisy_word_loss=float(noisy_word.detach()),corruption=proof)
    del out,noisy_loss,noisy_concept,noisy_word
    if self_history:
        training=model.training
        try:
            model.eval()
            # This API receives source features/mask only. No teacher target enters
            # generation; FP32 history matches the later deterministic evaluator.
            history_precision=torch.autocast("cuda",enabled=False) if features.device.type=="cuda" else nullcontext()
            with torch.no_grad(),history_precision:predicted=model.concepts.generate_codes(features,mask)
        finally:
            model.train(training)
        with autocast(features):out=model(features,mask,predicted)
        self_concept=level_cross_entropy(out["concept_logits"],codes,levels)
        self_word=F.cross_entropy(out["word_logits"].float().flatten(0,1),words.flatten(),ignore_index=0)
        self_loss=.25*(self_concept+self_word)
        if not torch.isfinite(self_loss):raise ValueError("nonfinite source-only-history loss")
        # Once on the first microbatch of every fourth update, after update256.
        # This auxiliary term is explicit and is not divided by accumulation.
        self_loss.backward()
        result.update(self_concept_loss=float(self_concept.detach()),self_word_loss=float(self_word.detach()),
            source_only_history_codes=predicted.detach().cpu().tolist())
    return result
