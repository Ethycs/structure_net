import json

import pytest

from experiments.structure_net import ncm_sequential_full_training_v2 as training


def test_memory_failure_keeps_exact_measurement_and_accepts_boundary(tmp_path, monkeypatch):
    cuda = training.torch.cuda
    cap = training.CONFIG['peak_limit_bytes']
    monkeypatch.setattr(cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(cuda, 'mem_get_info', lambda: (200_000_000, 6*1024**3))
    for name in ('memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(cuda, name, lambda: cap)
    assert training.memory_record(tmp_path, 'boundary', 1)['peak_allocated_bytes'] == cap
    monkeypatch.setattr(cuda, 'max_memory_allocated', lambda: cap + 1)
    with pytest.raises(ValueError, match=str(cap+1)):
        training.memory_record(tmp_path, 'over_budget', 2)
    records = [json.loads(line) for line in (tmp_path / 'memory.jsonl').read_text().splitlines()]
    assert [r['phase'] for r in records] == ['boundary', 'over_budget']
    assert records[-1]['peak_allocated_bytes'] == cap + 1
    assert records[-1]['device_free_bytes'] == 200_000_000
