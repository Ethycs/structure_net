import hashlib
import json

import pytest

from experiments.structure_net.ncm_recovered_training_inputs import Proofs


def test_cached_hash_detects_mutation_and_replacement(tmp_path):
    p = tmp_path/'payload'; p.write_bytes(b'first')
    proof = Proofs(); initial = proof.bind(p)
    assert proof.sha(p) == hashlib.sha256(b'first').hexdigest()
    p.write_bytes(b'other')
    with pytest.raises(ValueError): proof.bind(p)
    fresh = Proofs(); fresh.bind(p)
    replacement = tmp_path/'replacement'; replacement.write_bytes(b'third'); replacement.replace(p)
    with pytest.raises(ValueError): fresh.verify({str(p): hashlib.sha256(b'other').hexdigest()})
    assert initial != proof.sha(p)


def test_result_requires_observed_exit_and_exact_receipt_binding(tmp_path):
    result = tmp_path/'result.json'; result.write_text(json.dumps({'status': 'complete', 'artifacts': {}}))
    receipt = tmp_path/'actual_tool_receipt.json'
    def write(code, sha):
        receipt.write_text(json.dumps({'actual_tool_exit_code': code, 'artifacts': {str(result): sha}}))
    sha = hashlib.sha256(result.read_bytes()).hexdigest()
    write(1, sha)
    with pytest.raises(ValueError, match='successful'): Proofs().completed(tmp_path)
    write(0, 'wrong')
    with pytest.raises(ValueError, match='exact result'): Proofs().completed(tmp_path)
    write(0, sha)
    assert Proofs().completed(tmp_path)['status'] == 'complete'
