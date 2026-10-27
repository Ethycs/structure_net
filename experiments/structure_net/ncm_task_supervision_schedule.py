"""Fixed-budget paired sampling for existing versus expanded task supervision."""
from dataclasses import dataclass

import torch

from experiments.structure_net import ncm_public_curriculum as shared

ARMS = ('existing','expanded')
HYPOTHESIS = 'ncm-task-supervision-expansion-v1'
PARENT_CHECKPOINT_SHA = 'c59051faea7d9005c2d91b1c88113d627d7545c9ae1a9c3dd7ac0a2e8f96f620'


@dataclass(frozen=True)
class Config:
    updates: int = 2048
    effective_examples: int = 32
    microbatch: int = 4
    prefetch_batches: int = 2
    warmup: int = 128
    max_lr: float = 3e-5
    min_lr: float = 3e-6
    sampler_seed: int = 251142
    corruption_seed: int = 252142
    dropout_seed: int = 253142
    recovery_interval: int = 256
    peak_limit_bytes: int = 4_900_000_000
    cache_bytes: int = 256*1024**2
    physical_GPU: int = 0
    epochs: int = 1

    @property
    def updates_per_epoch(self):return self.updates

    @property
    def presentations_per_epoch(self):return self.updates*self.effective_examples

    def validate(self):
        if (self.epochs != 1 or min(self.updates,self.effective_examples,self.microbatch,self.prefetch_batches) < 1
                or self.effective_examples % self.microbatch or not 0 < self.warmup < self.updates
                or not 0 < self.min_lr <= self.max_lr):
            raise ValueError('fixed complete one-budget update geometry and learning-rate schedule required')


CONFIG = Config()


class Schedule:
    """Case permutations and noise depend on positions, not I/O or resume timing."""
    def __init__(self, cases, config=CONFIG):
        config.validate()
        if type(cases) is not int or cases < 1:raise ValueError('nonempty complete admitted population required')
        self.cases,self.config = cases,config
        count = config.presentations_per_epoch
        indices,cycle = [],0
        while len(indices) < count:
            rng = torch.Generator().manual_seed(shared.seed(config.sampler_seed,cycle,'task'))
            indices.extend(torch.randperm(cases,generator=rng).tolist());cycle += 1
        self.indices = indices[:count]
        rng = torch.Generator().manual_seed(shared.seed(config.corruption_seed,'flags'))
        order = torch.randperm(count,generator=rng)
        self.noisy = torch.zeros(count,dtype=torch.bool);self.noisy[order[:count//2]] = True

    def example(self, position):
        c = self.config
        if type(position) is not int or not 0 <= position < c.presentations_per_epoch:
            raise IndexError('valid completed-budget presentation position required')
        return shared.Example('anchor',self.indices[position],1,position,bool(self.noisy[position]),
            shared.seed(c.corruption_seed,1,position,'codes'))

    def update(self, index):
        c = self.config
        if type(index) is not int or not 0 <= index < c.updates:raise IndexError('valid optimizer update required')
        return [self.example(i) for i in range(index*c.effective_examples,(index+1)*c.effective_examples)]

    def batches(self,start_update=0,stop_update=None):
        c = self.config;stop = c.updates if stop_update is None else stop_update
        if not 0 <= start_update <= stop <= c.updates:raise ValueError('valid half-open completed-update interval required')
        for index in range(start_update,stop):
            examples = self.update(index)
            for offset in range(0,c.effective_examples,c.microbatch):
                yield index,c.effective_examples,examples[offset:offset+c.microbatch]
