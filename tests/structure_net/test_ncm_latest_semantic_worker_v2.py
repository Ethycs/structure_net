import pytest

from experiments.structure_net import ncm_latest_semantic_worker_v2 as worker


def test_request_adapters_restore_on_success_and_failure_without_clearing_measured_hashes(monkeypatch):
    latest = worker.latest
    write, configure, digest, request = (lambda *args: None for _ in range(4))
    monkeypatch.setattr(latest.previous, 'write_proof', write)
    monkeypatch.setattr(latest.semantic, 'configure', configure)
    monkeypatch.setattr(latest.semantic, 'PROOFS', None)
    monkeypatch.setattr(latest.data, 'sha', digest)
    monkeypatch.setattr(latest.repaired.qualification.q, 'request', request)
    hashes = {}; monkeypatch.setattr(worker.base, 'SHARED_HASHES', hashes)
    calls = []
    def body(value):
        assert latest.previous.write_proof is write
        assert latest.semantic.configure is configure and latest.semantic.PROOFS is None
        assert latest.data.sha is digest and latest.repaired.qualification.q.request is request
        calls.append(value)
        latest.previous.write_proof = lambda *args: 'semantic-only wrapper'
        latest.semantic.configure = lambda *args: 'temporary qualification'
        latest.semantic.PROOFS = {'temporary': value}
        latest.data.sha = lambda *args: 'temporary proof instance'
        latest.repaired.qualification.q.request = lambda *args: 'temporary route'
        hashes[value] = 'measured digest sentinel'
        if value == 'failure':
            raise RuntimeError('preserved failure')
        return 'completed'
    monkeypatch.setattr(worker, 'ORIGINAL_RUN', body)
    assert worker.scoped_request('grade') == 'completed'
    assert worker.scoped_request('export_outputs') == 'completed'
    with pytest.raises(RuntimeError, match='preserved failure'):
        worker.scoped_request('failure')
    assert worker.scoped_request('comparison') == 'completed'
    assert calls == ['grade', 'export_outputs', 'failure', 'comparison']
    assert set(hashes) == set(calls)
    assert latest.previous.write_proof is write and latest.semantic.configure is configure
    assert latest.semantic.PROOFS is None and latest.data.sha is digest
    assert latest.repaired.qualification.q.request is request
