#!/usr/bin/env python3
"""Temporal-phase language task: the circle task's latent geometry in English.

The model reads templated English reporting an event time in a local frame,
plus a calibration clause stating that frame's UTC offset, and must answer
with the event's UTC time-of-day quantized into 16 ordered answer bins.

Latent geometry is identical to the circle task: the UTC minute-of-day is a
phase on the 24-hour circle, the frame offset is an exact rotation (gauge),
and the two time-expression sheets ("m minutes past H" / "60-m minutes to
H+1") form an exact C2 double cover with task-irrelevant branch. Text is
tokenized with a BPE tokenizer trained on the BabyLM strict-small corpus;
answer tokens are pinned at ids 32000..32015 and can never be emitted by
text tokenization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


MINUTES_PER_DAY = 1_440
SPECIAL_TOKENS = ("<pad>", "<bos>", "<query>", "<unk>")
PAD_ID, BOS_ID, QUERY_ID, UNK_ID = 0, 1, 2, 3
MODES = ("calibrated", "uncalibrated", "utc_oracle")
REGIMES = ("train", "interpolation", "composition", "extrapolation")

MINUTE_WORDS = {
    5: "five", 10: "ten", 15: "fifteen", 20: "twenty", 25: "twenty five",
    30: "thirty", 35: "thirty five", 40: "forty", 45: "forty five",
    50: "fifty", 55: "fifty five",
}
HOUR_WORDS = (
    "twelve", "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "eleven",
)
_UNITS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"}


def _number_word(value: int) -> str:
    """English words for 0..59 (compositional, for arbitrary offsets)."""
    value = int(value)
    if not 0 <= value < 60:
        raise ValueError(f"number word out of range: {value}")
    if value < 20:
        return _UNITS[value]
    tens, units = divmod(value, 10)
    word = _TENS[tens * 10]
    return word if units == 0 else f"{word} {_UNITS[units]}"

REPORT_TEMPLATES = (
    "{person} recorded the start of the {event} as {time}",
    "the {event} began when the clock of {person} showed {time}",
    "according to {person} , the {event} started at {time}",
    "{person} wrote that the {event} started at {time}",
    "when the {event} began , the watch of {person} read {time}",
    "{person} said the {event} got going at {time}",
    # Held-out templates (composition and extrapolation only).
    "the {event} was already starting when {person} saw the clock at {time}",
    "as the {event} opened , {person} noted the time as {time}",
)
TRAIN_TEMPLATE_COUNT = 6

FILLERS = (
    "",
    "despite the rain ,",
    "after much discussion ,",
    "to the surprise of everyone ,",
    "without any warning ,",
)

TRAIN_PERSONS = (
    "ada", "ben", "carla", "devi", "emil", "farah",
    "gus", "hana", "ivan", "june", "kofi", "lena",
)
HELDOUT_PERSONS = ("mira", "noel", "oksana", "pavel", "queenie", "ravi")
TRAIN_EVENTS = (
    "rehearsal", "meeting", "lecture", "match", "parade", "auction",
    "concert", "ceremony", "workshop", "seminar", "banquet", "recital",
)
HELDOUT_EVENTS = ("regatta", "assembly", "audit", "festival", "briefing", "tournament")

CALIBRATION_TEMPLATES = (
    "local clocks run {offset} coordinated universal time",
    "in this town the clocks sit {offset} coordinated universal time",
)


def _offset_phrase(offset_minutes: int) -> str:
    if offset_minutes == 0:
        return "exactly on"
    magnitude = abs(int(offset_minutes))
    hours, minutes = divmod(magnitude, 60)
    parts: List[str] = []
    if hours:
        parts.append(f"{_number_word(hours)} {'hour' if hours == 1 else 'hours'}")
    if minutes:
        parts.append(
            f"{_number_word(minutes)} {'minute' if minutes == 1 else 'minutes'}"
        )
    direction = "ahead of" if offset_minutes > 0 else "behind"
    return f"{' and '.join(parts)} {direction}"


def _day_part(hour: int) -> str:
    if hour < 5:
        return "at night"
    if hour < 12:
        return "in the morning"
    if hour < 17:
        return "in the afternoon"
    if hour < 21:
        return "in the evening"
    return "at night"


def time_phrase(local_minute_of_day: int, branch: int) -> str:
    """Render a local time on one C2 sheet.

    branch +1: "m minutes past H"; branch -1: "60-m minutes to H+1".
    Requires the minute part to lie in {5,...,55} so both sheets are
    nondegenerate.
    """
    minute_of_day = int(local_minute_of_day) % MINUTES_PER_DAY
    hour, minute = divmod(minute_of_day, 60)
    if minute not in MINUTE_WORDS:
        raise ValueError(f"minute part {minute} is outside the two-sheet domain")
    if branch == 1:
        return (
            f"{MINUTE_WORDS[minute]} minutes past "
            f"{HOUR_WORDS[hour % 12]} {_day_part(hour)}"
        )
    if branch == -1:
        anchor = (hour + 1) % 24
        return (
            f"{MINUTE_WORDS[60 - minute]} minutes to "
            f"{HOUR_WORDS[anchor % 12]} {_day_part(anchor)}"
        )
    raise ValueError(f"branch must be +1 or -1, got {branch}")


@dataclass(frozen=True)
class TemporalLanguageTaskConfig:
    phase_bins: int = 16
    vocab_size: int = 50_257
    answer_token_start: int = 32_000
    text_vocab_size: int = 16_000
    sequence_length: int = 96
    target_noise_radians: float = 0.42
    minute_step: int = 5
    train_offset_hours: Tuple[int, ...] = tuple(
        hour for hour in range(-11, 12) if hour not in (-7, 3)
    )
    composition_offset_hours: Tuple[int, ...] = (-7, 3)
    extrapolation_offset_minutes: Tuple[int, ...] = (-570, -210, 270, 630)
    tokenizer_path: str = "data/corpora/babylm_10M_bpe16k.tokenizer.json"

    def __post_init__(self) -> None:
        if self.phase_bins != 16:
            raise ValueError("the answer interface is fixed at 16 ordered bins")
        if self.answer_token_start + self.phase_bins > self.vocab_size:
            raise ValueError("answer tokens exceed the vocabulary")
        if self.text_vocab_size > self.answer_token_start:
            raise ValueError("text vocabulary would collide with answer tokens")
        if self.minute_step != 5 or self.sequence_length < 48:
            raise ValueError("the primary time grid and sequence budget are fixed")
        overlap = set(self.train_offset_hours) & set(self.composition_offset_hours)
        if overlap:
            raise ValueError(f"offset regimes overlap: {sorted(overlap)}")
        if any(minutes % 60 == 0 for minutes in self.extrapolation_offset_minutes):
            raise ValueError("extrapolation offsets must be off the whole-hour grid")

    @property
    def answer_token_ids(self) -> List[int]:
        return list(
            range(self.answer_token_start, self.answer_token_start + self.phase_bins)
        )


def train_bpe_tokenizer(
    corpus_dir: Path, output_path: Path, *, text_vocab_size: int = 16_000
) -> str:
    """Train a byte-level BPE tokenizer on the BabyLM text files."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

    files = sorted(str(path) for path in Path(corpus_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no corpus files under {corpus_dir}")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=int(text_vocab_size),
        special_tokens=list(SPECIAL_TOKENS),
        show_progress=False,
    )
    tokenizer.train(files, trainer)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def load_tokenizer(config: TemporalLanguageTaskConfig):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(config.tokenizer_path)))
    for index, token in enumerate(SPECIAL_TOKENS):
        if tokenizer.token_to_id(token) != index:
            raise ValueError(f"special token {token} is not at id {index}")
    if tokenizer.get_vocab_size() > config.text_vocab_size:
        raise ValueError("tokenizer vocabulary exceeds the configured text budget")
    return tokenizer


def wrapped_target_posteriors(
    phases: np.ndarray, config: TemporalLanguageTaskConfig
) -> np.ndarray:
    """Same wrapped soft targets as the circle task."""
    bin_angles = 2.0 * math.pi * np.arange(config.phase_bins) / config.phase_bins
    differences = (phases[:, None] - bin_angles[None, :] + math.pi) % (
        2.0 * math.pi
    ) - math.pi
    probabilities = np.exp(-0.5 * (differences / config.target_noise_radians) ** 2)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class TemporalLanguageDataset:
    texts: Tuple[str, ...]
    input_ids: torch.Tensor
    target_posteriors: torch.Tensor
    target_bins: torch.Tensor
    phases: torch.Tensor
    cosines: torch.Tensor
    fiber_id: torch.Tensor
    branch: torch.Tensor
    offset_minutes_true: torch.Tensor
    offset_minutes_stated: torch.Tensor
    nuisance: Dict[str, torch.Tensor] = field(repr=False, default_factory=dict)


def _regime_pools(
    config: TemporalLanguageTaskConfig, regime: str
) -> Tuple[List[int], List[int], Sequence[str], Sequence[str]]:
    """Offsets (minutes), template ids, person pool, event pool per regime."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime}")
    if regime in ("train", "interpolation"):
        offsets = [hour * 60 for hour in config.train_offset_hours]
        templates = list(range(TRAIN_TEMPLATE_COUNT))
        return offsets, templates, TRAIN_PERSONS, TRAIN_EVENTS
    if regime == "composition":
        offsets = [hour * 60 for hour in config.composition_offset_hours]
        templates = list(range(TRAIN_TEMPLATE_COUNT, len(REPORT_TEMPLATES)))
        return offsets, templates, HELDOUT_PERSONS, HELDOUT_EVENTS
    offsets = list(config.extrapolation_offset_minutes)
    templates = list(range(len(REPORT_TEMPLATES)))
    return offsets, templates, HELDOUT_PERSONS, HELDOUT_EVENTS


def render_example(
    *,
    local_minute: int,
    branch: int,
    stated_offset_minutes: int,
    template_id: int,
    person: str,
    event: str,
    filler_id: int,
    calibration_template_id: int,
    calibration_first: bool,
    mode: str,
) -> str:
    """Render one example's text (without specials)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    report = REPORT_TEMPLATES[template_id].format(
        person=person, event=event, time=time_phrase(local_minute, branch)
    )
    filler = FILLERS[filler_id]
    if filler:
        report = f"{filler} {report}"
    if mode in ("uncalibrated", "utc_oracle"):
        return f"{report} ."
    calibration = CALIBRATION_TEMPLATES[calibration_template_id].format(
        offset=_offset_phrase(stated_offset_minutes)
    )
    if calibration_first:
        return f"{calibration} . {report} ."
    return f"{report} . {calibration} ."


def generate_paired_temporal_dataset(
    config: TemporalLanguageTaskConfig,
    tokenizer,
    *,
    sample_count: int,
    seed: int,
    regime: str = "train",
    mode: str = "calibrated",
    stated_offset_noise_minutes: float = 0.0,
) -> TemporalLanguageDataset:
    """Generate exact C2 fiber pairs of templated temporal reports.

    Every fiber contributes two examples with identical latent and nuisance
    draws that differ only in the time-expression sheet (branch +1/-1).
    """
    if sample_count % 2:
        raise ValueError("sample_count must be even (two sheets per fiber)")
    generator = np.random.default_rng(seed)
    fibers = sample_count // 2
    offsets, template_pool, persons, events = _regime_pools(config, regime)

    minute_parts = np.array(sorted(MINUTE_WORDS), dtype=np.int64)
    local_minutes = (
        generator.integers(0, 24, fibers) * 60
        + minute_parts[generator.integers(0, len(minute_parts), fibers)]
    )
    true_offsets = np.array(offsets, dtype=np.int64)[
        generator.integers(0, len(offsets), fibers)
    ]
    if mode == "utc_oracle":
        true_offsets = np.zeros(fibers, dtype=np.int64)
    # Stated-offset noise uses a dedicated stream so clean and noisy datasets
    # at the same seed share identical latent and nuisance draws (the
    # common-random-numbers property the calibration titration relies on).
    stated_offsets = true_offsets.astype(np.float64)
    if stated_offset_noise_minutes > 0.0:
        noise_generator = np.random.default_rng((seed, 977_003))
        stated_offsets = stated_offsets + noise_generator.normal(
            0.0, stated_offset_noise_minutes, fibers
        )
    stated_offsets = np.rint(stated_offsets).astype(np.int64)

    template_ids = np.array(template_pool, dtype=np.int64)[
        generator.integers(0, len(template_pool), fibers)
    ]
    person_ids = generator.integers(0, len(persons), fibers)
    event_ids = generator.integers(0, len(events), fibers)
    filler_ids = generator.integers(0, len(FILLERS), fibers)
    calibration_ids = generator.integers(0, len(CALIBRATION_TEMPLATES), fibers)
    calibration_first = generator.integers(0, 2, fibers).astype(bool)

    utc_minutes = (local_minutes - true_offsets) % MINUTES_PER_DAY
    phases = 2.0 * math.pi * utc_minutes.astype(np.float64) / MINUTES_PER_DAY
    posterior_fiber = wrapped_target_posteriors(phases, config)

    texts: List[str] = []
    rows: List[List[int]] = []
    for index in range(fibers):
        for branch in (1, -1):
            text = render_example(
                local_minute=int(local_minutes[index]),
                branch=branch,
                stated_offset_minutes=int(stated_offsets[index]),
                template_id=int(template_ids[index]),
                person=persons[int(person_ids[index])],
                event=events[int(event_ids[index])],
                filler_id=int(filler_ids[index]),
                calibration_template_id=int(calibration_ids[index]),
                calibration_first=bool(calibration_first[index]),
                mode=mode,
            )
            token_ids = tokenizer.encode(text).ids
            if len(token_ids) > config.sequence_length - 2:
                raise ValueError(
                    f"rendered text exceeds the sequence budget: {text!r}"
                )
            if any(item >= config.answer_token_start for item in token_ids):
                raise ValueError("text tokenization produced an answer-token id")
            padding = config.sequence_length - 2 - len(token_ids)
            rows.append(
                [BOS_ID] + token_ids + [PAD_ID] * padding + [QUERY_ID]
            )
            texts.append(text)

    repeat = np.repeat(np.arange(fibers), 2)
    posteriors = posterior_fiber[repeat]
    return TemporalLanguageDataset(
        texts=tuple(texts),
        input_ids=torch.tensor(rows, dtype=torch.long),
        target_posteriors=torch.tensor(posteriors, dtype=torch.float32),
        target_bins=torch.tensor(posteriors.argmax(axis=1), dtype=torch.long),
        phases=torch.tensor(phases[repeat], dtype=torch.float64),
        cosines=torch.tensor(np.cos(phases[repeat]), dtype=torch.float64),
        fiber_id=torch.tensor(repeat, dtype=torch.long),
        branch=torch.tensor([1, -1] * fibers, dtype=torch.long),
        offset_minutes_true=torch.tensor(true_offsets[repeat], dtype=torch.long),
        offset_minutes_stated=torch.tensor(stated_offsets[repeat], dtype=torch.long),
        nuisance={
            "template_id": torch.tensor(template_ids[repeat], dtype=torch.long),
            "person_id": torch.tensor(person_ids[repeat], dtype=torch.long),
            "event_id": torch.tensor(event_ids[repeat], dtype=torch.long),
            "filler_id": torch.tensor(filler_ids[repeat], dtype=torch.long),
            "calibration_template_id": torch.tensor(
                calibration_ids[repeat], dtype=torch.long
            ),
            "calibration_first": torch.tensor(
                calibration_first[repeat], dtype=torch.bool
            ),
            "local_minute": torch.tensor(local_minutes[repeat], dtype=torch.long),
        },
    )


def dataset_digest(dataset: TemporalLanguageDataset) -> str:
    material = {
        "input_ids": hashlib.sha256(
            dataset.input_ids.numpy().tobytes()
        ).hexdigest(),
        "targets": hashlib.sha256(
            dataset.target_posteriors.numpy().tobytes()
        ).hexdigest(),
        "fibers": hashlib.sha256(dataset.fiber_id.numpy().tobytes()).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True).encode()
    ).hexdigest()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tokenizer", action="store_true")
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/corpora/babylm_10M")
    )
    parser.add_argument(
        "--tokenizer-output",
        type=Path,
        default=Path("data/corpora/babylm_10M_bpe16k.tokenizer.json"),
    )
    parser.add_argument("--preview", type=int, default=0)
    args = parser.parse_args()
    if args.train_tokenizer:
        digest = train_bpe_tokenizer(args.corpus, args.tokenizer_output)
        print(f"tokenizer sha256: {digest}")
        print(args.tokenizer_output)
    if args.preview:
        config = TemporalLanguageTaskConfig(
            tokenizer_path=str(args.tokenizer_output)
        )
        tokenizer = load_tokenizer(config)
        dataset = generate_paired_temporal_dataset(
            config, tokenizer, sample_count=args.preview * 2, seed=7
        )
        for index in range(args.preview * 2):
            print(
                f"[bin {int(dataset.target_bins[index]):>2} "
                f"branch {int(dataset.branch[index]):+d}] "
                f"{dataset.texts[index]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
