from dataclasses import replace
import threading

import pytest
import torch

from experiments.structure_net import ncm_public_curriculum as curriculum


def small():
    return replace(curriculum.CONFIG, public_cases=11, anchor_cases=3, epochs=2,
                   milestones=(1, 2), effective_examples=8, warmup=1)


def test_full_population_counts_tail_and_no_anchor_starvation():
    c = curriculum.CONFIG
    assert (c.anchors_per_epoch, c.presentations_per_epoch, c.updates_per_epoch, c.updates) == (
        54118, 216471, 1692, 13536)
    epoch = curriculum.Epoch(1, c)
    examples = [epoch.example(i) for i in range(c.presentations_per_epoch)]
    assert sorted(e.index for e in examples if e.population == 'public') == list(range(c.public_cases))
    assert {e.index for e in examples if e.population == 'anchor'} == set(range(c.anchor_cases))
    assert sum(e.noisy for e in examples) == c.presentations_per_epoch // 2
    assert len(epoch.update(c.updates_per_epoch - 1)) == 23
    assert [e.population for e in examples[-3:]] == ['public', 'public', 'anchor']


def test_resume_order_is_independent_of_prefetch_and_global_rng():
    c = small()
    state = torch.get_rng_state().clone()
    epoch = curriculum.Epoch(1, c)
    assert torch.equal(state, torch.get_rng_state())
    expected = list(epoch.batches())
    owners = set()
    def fetch(refs):
        owners.add(threading.get_ident())
        return [(r.population, r.index, r.corruption_seed) for r in refs]
    iterator = curriculum.prefetch(epoch.batches(), fetch, capacity=2)
    seen = [next(iterator), next(iterator)]
    iterator.close()  # Prefetched future batches were not optimizer updates.
    resumed = list(curriculum.prefetch(curriculum.Epoch(1, c).batches(start_update=1), fetch))
    assert [row[:3] for row in seen + resumed] == expected
    assert all(row[3] == fetch(row[2]) for row in seen + resumed)
    assert len(owners - {threading.get_ident()}) <= 2  # One owner per iterator.
    assert torch.equal(state, torch.get_rng_state())
    assert curriculum.Epoch(2, c).public != epoch.public


def test_prefetch_is_bounded_and_propagates_io_failure():
    c = small()
    consumed = []
    def batches():
        for batch in curriculum.Epoch(1, c).batches():
            consumed.append(batch)
            yield batch
    iterator = curriculum.prefetch(batches(), lambda refs: refs, capacity=2)
    next(iterator)
    assert len(consumed) == 3  # Current batch plus two queued batches.
    iterator.close()
    def broken(refs):
        raise ValueError('corrupt feature payload')
    with pytest.raises(ValueError, match='corrupt feature'):
        list(curriculum.prefetch(curriculum.Epoch(1, c).batches(), broken))


def test_partial_updates_keep_true_denominator_and_schedule():
    c = small()
    batches = list(curriculum.Epoch(1, c).batches())
    assert [len(row[2]) for row in batches] == [4, 4, 4, 3]
    assert [row[1] for row in batches] == [8, 8, 7, 7]
    assert curriculum.rate(c.warmup, c) == c.max_lr
    assert curriculum.rate(c.updates, c) == c.min_lr
    with pytest.raises(IndexError): curriculum.Epoch(1, c).update(2)
    with pytest.raises(IndexError): curriculum.Epoch(1, c).example(15)
