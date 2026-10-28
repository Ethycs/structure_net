from contextlib import contextmanager

import pytest
import torch

from experiments.structure_net import ncm_recovered_source_guidance as run


def test_author_ramp_and_constant_schedules_are_distinct_at_root():
    assert run.weight(.5, 'ramp', 0, 12) == 0
    assert run.weight(.5, 'ramp', 11, 12) == .5
    assert run.weight(.5, 'constant', 0, 12) == .5
    with pytest.raises(ValueError):
        run.weight(float('nan'), 'constant', 0, 12)


def test_zero_guidance_is_exact_and_positive_guidance_removes_prior():
    conditional, unconditional = torch.tensor([1., 2.]), torch.tensor([3., -2.])
    assert run.mix(conditional, unconditional, 0) is conditional
    assert torch.equal(run.mix(conditional, unconditional, .5), torch.tensor([0., 4.]))
    with pytest.raises(ValueError):
        run.mix(conditional, unconditional*float('inf'), 1)


def test_source_bypass_restores_method_and_layout_after_exception():
    class Head:
        training = False
        _bound = None
        calls = 0

        @contextmanager
        def source_context(self, *args):
            self.calls += 1
            yield

    head = Head()
    with pytest.raises(RuntimeError, match='intentional'):
        with run.disabled_source(head):
            with head.source_context(None, None):
                assert head.calls == 0
                raise RuntimeError('intentional')
    assert 'source_context' not in head.__dict__
    with head.source_context(None, None):
        pass
    assert head.calls == 1
    head._bound = object()
    with pytest.raises(ValueError, match='active source'):
        with run.disabled_source(head):
            pass
