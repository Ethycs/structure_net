"""Prospective, paper-inspired same-codebook corruption for every native level.

No trained model uses this helper yet. The linear direction follows NCM F.3;
the explicit endpoints below are our experimental choices, not released author
training settings. The first *generated code* is not the BOS/null prefix.
"""
from dataclasses import asdict, dataclass
import hashlib

import torch


@dataclass(frozen=True)
class Config:
    first_level_rate: float = .75
    last_level_rate: float = .25

    def __post_init__(self):
        if not 0 <= self.last_level_rate <= self.first_level_rate <= 1:
            raise ValueError('decreasing finite corruption rates in[0,1] required')


def digest(codes):
    return hashlib.sha256(codes.contiguous().numpy().tobytes()).hexdigest()


def corrupt(codes, levels, vocabularies, generator, config=Config()):
    """Return corrupted history and replay metadata, leaving clean targets intact."""
    if (codes.device.type != 'cpu' or codes.dtype != torch.long or codes.ndim != 2
            or len(levels) < 2 or len(vocabularies) != len(levels)
            or any(type(v) is not int or v < 1 for v in levels)
            or any(type(v) is not int or v < 2 for v in vocabularies)
            or sum(levels) != codes.shape[1] or codes.shape[0] < 1):
        raise ValueError('complete CPU target hierarchy and native per-level vocabularies required')
    rates = torch.linspace(config.first_level_rate, config.last_level_rate, len(levels))
    before = digest(codes)
    history = codes.clone()
    details, offset = [], 0
    for index, (width, vocabulary) in enumerate(zip(levels, vocabularies)):
        clean = codes[:, offset:offset + width]
        if (clean < 0).any() or (clean >= vocabulary).any():
            raise ValueError('code outside its own native level codebook')
        selected = torch.rand(clean.shape, generator=generator) < rates[index]
        replacements = torch.randint(vocabulary, clean.shape, generator=generator)
        noisy = torch.where(selected, replacements, clean)
        history[:, offset:offset + width] = noisy
        details.append({'level': index, 'width': width, 'vocabulary': vocabulary, 'rate': float(rates[index]),
            'selected_positions': int(selected.sum()), 'changed_positions': int((noisy != clean).sum())})
        offset += width
    if digest(codes) != before:
        raise RuntimeError('clean target codes changed')
    return history, {'config': asdict(config), 'levels': details, 'clean_target_sha256': before,
        'history_sha256': digest(history), 'clean_targets_unchanged': True,
        'first_generated_code_protected': False, 'BOS_or_null_prefix_in_this_tensor': False,
        'author_exact_rates_available': False}
