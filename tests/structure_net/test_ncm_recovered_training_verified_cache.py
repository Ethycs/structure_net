import os
from pathlib import Path

import pytest

from experiments.structure_net import ncm_recovered_training_verified_cache as run


def fixture(tmp_path, monkeypatch):
    path = tmp_path/'payload'; path.write_bytes(b'a'*1024**2)
    clock = run.study.inputs.time.time_ns
    monkeypatch.setattr(run.study.inputs.time, 'time_ns', lambda: clock()+3_000_000_000)
    original_open = Path.open; reads = []
    def opening(self, mode='r', *args, **kwargs):
        if self == path and mode == 'rb': reads.append(self)
        return original_open(self, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', opening)
    return path, reads


def test_only_unchanged_measured_hashes_are_reused_within_process(tmp_path, monkeypatch):
    path, reads = fixture(tmp_path, monkeypatch)
    proof = run.ProcessProofs(); digest = proof.bind(path); bindings = {str(path): digest}
    proof.verify(bindings); proof.verify(bindings)
    assert len(reads) == 1
    run.ProcessProofs().verify(bindings)
    assert len(reads) == 2  # Independent consumer/audit must hash bytes again.
    with pytest.raises(ValueError): proof.verify({str(path): 'unmeasured-declared-digest'})


def test_mutation_and_replacement_invalidate_cache_even_with_restored_mtime(tmp_path, monkeypatch):
    path, reads = fixture(tmp_path, monkeypatch)
    proof = run.ProcessProofs(); before = path.stat(); digest = proof.bind(path)
    path.write_bytes(b'b'*1024**2); os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError): proof.verify({str(path): digest})
    assert len(reads) == 2
    proof = run.ProcessProofs(); current = proof.bind(path)
    replacement = tmp_path/'replacement'; replacement.write_bytes(b'c'*1024**2); replacement.replace(path)
    with pytest.raises(ValueError): proof.verify({str(path): current})
    assert len(reads) == 4


def test_small_or_recent_files_are_not_cache_hits(tmp_path, monkeypatch):
    path, reads = fixture(tmp_path, monkeypatch)
    proof = run.ProcessProofs(); digest = proof.bind(path)
    monkeypatch.setattr(run.study.inputs.time, 'time_ns', lambda: path.stat().st_ctime_ns+1)
    proof.verify({str(path): digest}); proof.verify({str(path): digest})
    assert len(reads) == 3
    small = tmp_path/'small'; small.write_bytes(b'old')
    first = proof.bind(small); small.write_bytes(b'new')
    with pytest.raises(ValueError): proof.verify({str(small): first})
