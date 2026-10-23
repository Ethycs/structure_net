"""Faithful port of the released NCM text-logit decoding policy."""
import torch


def pick_tokens(logits, *, frequency=0., deficit=0., adjacency=None, allowance=None):
    if frequency <= 0 and not adjacency:
        return logits.argmax(-1)
    batch, length, vocab = logits.shape
    lg = logits.double()
    allow = (allowance[:vocab].to(device=lg.device, dtype=torch.float64)*length)[None, :] if allowance is not None else None
    amplitude = adjacency[0] if adjacency else 0.
    window = int(adjacency[1]) if adjacency and len(adjacency) > 1 else 1
    decay = adjacency[2] if adjacency and len(adjacency) > 2 else 1.
    counts = torch.zeros(batch, vocab, dtype=torch.float64, device=lg.device)
    output = torch.empty(batch, length, dtype=torch.long, device=lg.device)
    history = []
    for t in range(length):
        step = lg[:, t, :].clone()
        if frequency > 0:
            excess = (counts-allow).clamp_min(0.) if allow is not None else counts
            step -= frequency*excess
            if deficit > 0 and allow is not None:
                step += deficit*((length-t)/length)*(allow-counts).clamp_min(0.)
        for j, token in enumerate(reversed(history[-window:])):
            step.scatter_add_(1, token[:, None], torch.full((batch, 1), -amplitude*decay**j,
                dtype=torch.float64, device=lg.device))
        chosen = step.argmax(-1)
        output[:, t] = chosen
        history.append(chosen)
        counts.scatter_add_(1, chosen[:, None], torch.ones(batch, 1, dtype=torch.float64, device=lg.device))
    return output


def render_author(ids, gpt):
    ids = list(ids)
    if 50256 in ids:
        ids = ids[:ids.index(50256)]
    return gpt.decode([token for token in ids if token not in (50256, 50257)], skip_special_tokens=True).strip()
