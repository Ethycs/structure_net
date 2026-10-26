"""Complete public epochs, reviewed anchors, and restartable bounded disk reads."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import math

import torch


@dataclass(frozen=True)
class Config:
    public_cases: int = 162353
    anchor_cases: int = 1376
    epochs: int = 8
    milestones: tuple = (1, 2, 4, 8)
    microbatch: int = 4
    effective_examples: int = 128
    public_per_anchor: int = 3
    prefetch_batches: int = 2
    sampler_seed: int = 231142
    corruption_seed: int = 232142
    dropout_seed: int = 233142
    warmup: int = 256
    max_lr: float = 1e-4
    min_lr: float = 1e-5
    recovery_interval: int = 256
    peak_limit_bytes: int = 4_900_000_000
    deterministic_algorithms: bool = True
    CUBLAS_WORKSPACE_CONFIG: str = ':4096:8'

    @property
    def anchors_per_epoch(self):
        return math.ceil(self.public_cases / self.public_per_anchor)

    @property
    def presentations_per_epoch(self):
        return self.public_cases + self.anchors_per_epoch

    @property
    def updates_per_epoch(self):
        return math.ceil(self.presentations_per_epoch / self.effective_examples)

    @property
    def updates(self):
        return self.epochs * self.updates_per_epoch


CONFIG = Config()


def seed(base, *parts):
    message = ':'.join(map(str, (base, *parts))).encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8], 'big') % (2**63 - 1)


@dataclass(frozen=True)
class Example:
    population: str
    index: int
    epoch: int
    position: int
    noisy: bool
    corruption_seed: int


class Epoch:
    """Sampling is a pure function of epoch and position, never prefetch state."""
    def __init__(self, number, config=CONFIG):
        if (type(number) is not int or not 1 <= number <= config.epochs
                or min(config.public_cases, config.anchor_cases, config.public_per_anchor,
                       config.microbatch, config.effective_examples) < 1
                or config.effective_examples % config.microbatch):
            raise ValueError('valid complete epoch and divisible update geometry required')
        self.number, self.config = number, config
        rng = torch.Generator().manual_seed(seed(config.sampler_seed, number, 'public'))
        self.public = torch.randperm(config.public_cases, generator=rng).tolist()
        self.anchors = []
        cycle = 0
        while len(self.anchors) < config.anchors_per_epoch:
            rng.manual_seed(seed(config.sampler_seed, number, 'anchor', cycle))
            self.anchors.extend(torch.randperm(config.anchor_cases, generator=rng).tolist())
            cycle += 1
        self.anchors = self.anchors[:config.anchors_per_epoch]
        rng.manual_seed(seed(config.corruption_seed, number, 'flags'))
        order = torch.randperm(config.presentations_per_epoch, generator=rng)
        self.noisy = torch.zeros(config.presentations_per_epoch, dtype=torch.bool)
        self.noisy[order[:config.presentations_per_epoch // 2]] = True

    def example(self, position):
        c = self.config
        if type(position) is not int or not 0 <= position < c.presentations_per_epoch:
            raise IndexError('valid complete epoch position required')
        group, within = divmod(position, c.public_per_anchor + 1)
        public_start = group * c.public_per_anchor
        public_count = min(c.public_per_anchor, c.public_cases - public_start)
        if within < public_count:
            population, index = 'public', self.public[public_start + within]
        else:
            population, index = 'anchor', self.anchors[group]
        return Example(population, index, self.number, position, bool(self.noisy[position]),
                       seed(c.corruption_seed, self.number, position, 'codes'))

    def update(self, index):
        c = self.config
        if type(index) is not int or not 0 <= index < c.updates_per_epoch:
            raise IndexError('valid zero-based epoch update required')
        start = index * c.effective_examples
        return [self.example(i) for i in range(start, min(start + c.effective_examples,
                                                       c.presentations_per_epoch))]

    def batches(self, start_update=0, stop_update=None):
        stop = self.config.updates_per_epoch if stop_update is None else stop_update
        if not 0 <= start_update <= stop <= self.config.updates_per_epoch:
            raise ValueError('valid half-open update range required')
        for update in range(start_update, stop):
            examples = self.update(update)
            for start in range(0, len(examples), self.config.microbatch):
                yield update, len(examples), examples[start:start + self.config.microbatch]


def rate(step, config=CONFIG):
    if not 1 <= step <= config.updates or not 0 < config.warmup < config.updates:
        raise ValueError('step and warmup must fit the complete fixed schedule')
    if step <= config.warmup:
        return config.max_lr * step / config.warmup
    return config.min_lr + .5 * (config.max_lr - config.min_lr) * (
        1 + math.cos(math.pi * (step - config.warmup) / (config.updates - config.warmup)))


def prefetch(batches, fetch, *, capacity=2):
    """One I/O owner avoids shared seek-handle races; at most capacity futures.

    This iterator owns no training RNG or completed-update cursor. Closing it
    discards speculative reads only. The caller checkpoints completed updates.
    """
    if type(capacity) is not int or capacity < 1:
        raise ValueError('positive bounded prefetch capacity required')
    iterator, pending = iter(batches), deque()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='ncm-disk') as pool:
        def submit():
            try:
                batch = next(iterator)
            except StopIteration:
                return False
            pending.append((batch, pool.submit(fetch, batch[2])))
            return True
        for _ in range(capacity):
            if not submit():
                break
        try:
            while pending:
                batch, future = pending.popleft()
                value = future.result()
                submit()
                yield *batch, value
                del value
        finally:
            for _, future in pending:
                future.cancel()
