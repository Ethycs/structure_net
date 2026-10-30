import time

import pytest

from experiments.structure_net import ncm_latest_semantic_worker_v3 as worker


def test_real_legacy_verifier_reuses_measured_bytes_and_rejects_changed_input(tmp_path, monkeypatch):
    worker.base.SHARED_HASHES.clear()
    from experiments.structure_net import ncm_recovered_training_inputs as original
    clock = time.time_ns(); monkeypatch.setattr(original.time, 'time_ns', lambda: clock+10_000_000_000)
    path = tmp_path/'large'; path.write_bytes(b'a'*(1024**2))
    expected = worker.paper.file_hash(path)
    proof = {'source_hashes': {}, 'input_hashes': {str(path): expected}}
    reads = []
    original_open = type(path).open
    def observe(p, *args, **kwargs):
        if p == path and args and args[0] == 'rb': reads.append(p)
        return original_open(p, *args, **kwargs)
    monkeypatch.setattr(type(path), 'open', observe)
    with worker.guarded_legacy_verifier():
        worker.paper.verify(proof); worker.paper.verify(proof)
        assert len(reads) == 1
        path.write_bytes(b'b'*(1024**2))
        with pytest.raises(ValueError, match='producer changed'): worker.paper.verify(proof)
        assert len(reads) == 2
    worker.base.SHARED_HASHES.clear()


def test_legacy_helper_is_restored_even_on_failure():
    original = worker.paper.file_hash
    with pytest.raises(RuntimeError):
        with worker.guarded_legacy_verifier():
            assert worker.paper.file_hash is not original
            raise RuntimeError('stage failed')
    assert worker.paper.file_hash is original


def test_only_the_legacy_proof_helper_changes():
    from experiments.structure_net import ncm_memory_data as memory
    original = memory.file_hash
    with worker.guarded_legacy_verifier():
        assert memory.file_hash is original
    assert worker.paper.file_hash is original


def test_binding_preserves_metric_and_original_inputs(monkeypatch):
    monkeypatch.setattr(worker.latest.data, 'sha', lambda p: 'measured')
    original = {'config': {'metric': 'complete_summary_semantic_pass'}, 'input_hashes': {'model': 'fixed'}}
    bound = worker.bind_legacy(original)
    assert bound['config']['metric'] == original['config']['metric']
    assert bound['input_hashes']['model'] == 'fixed'
    assert bound['config']['legacy_producer_hash_validation']['same_live_worker_measurements_only']
    assert original == {'config': {'metric': 'complete_summary_semantic_pass'}, 'input_hashes': {'model': 'fixed'}}
