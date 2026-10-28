"""Fixed four-epoch optimization contract for recovered conversational pairs."""
from dataclasses import dataclass

from experiments.structure_net import ncm_recovered_source_balancing as balancing

HYPOTHESIS = 'ncm-recovered-paraphrase-training-v1'
SCHEMA = 'nal.ncm-recovered-paraphrase-training.v1'


@dataclass(frozen=True)
class Config(balancing.Config):
    prefetch_batches: int = 2
    warmup: int = 128
    max_lr: float = 3e-5
    min_lr: float = 3e-6
    dropout_seed: int = 274142
    recovery_interval: int = 256
    peak_limit_bytes: int = 4_900_000_000
    host_limit_bytes: int = 12*1024**3
    cache_bytes: int = 256*1024**2
    physical_GPU: int = 2

    def validate(self):
        super().validate()
        if (not 0 < self.warmup < self.updates or not 0 < self.min_lr <= self.max_lr
                or min(self.prefetch_batches, self.recovery_interval, self.peak_limit_bytes,
                       self.host_limit_bytes, self.cache_bytes) < 1
                or self.physical_GPU != 2):
            raise ValueError('fixed GPU and valid full optimization schedule required')


CONFIG = Config()


def batches(groups, start_update=0, stop_update=None, config=CONFIG):
    """Global completed-update cursors with the engine's epoch-local batches."""
    config.validate()
    end = config.updates if stop_update is None else stop_update
    if not 0 <= start_update <= end <= config.updates:
        raise ValueError('valid half-open global update interval required')
    cursor = start_update
    while cursor < end:
        number, local = divmod(cursor, config.updates_per_epoch)
        stop = min(config.updates_per_epoch, end-number*config.updates_per_epoch)
        epoch = balancing.Epoch(number+1, groups, config)
        yield from epoch.batches(start_update=local, stop_update=stop)
        cursor = number*config.updates_per_epoch+stop
