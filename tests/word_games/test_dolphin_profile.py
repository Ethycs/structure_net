"""Derived weights must not be accepted from stale receipts or wrong contents."""
import hashlib
import json
import sys

import pytest

from experiments.word_games_triton import prepare_dolphin


def test_stale_receipt_cannot_skip_weight_verification(tmp_path):
    weights = tmp_path / 'model.gguf'
    weights.write_bytes(b'original')
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    weights.with_suffix('.verified.json').write_text(json.dumps({
        'sha256': digest, 'identity': prepare_dolphin.identity(weights)}))
    spec = {'path': str(weights), 'sha256': digest}
    assert prepare_dolphin.verified(spec)
    weights.write_bytes(b'changed content')
    assert not prepare_dolphin.verified(spec)


def test_wrong_existing_weights_are_rejected_without_publication(tmp_path, monkeypatch):
    weights = tmp_path / 'model.gguf'
    weights.write_bytes(b'wrong model')
    recipe = tmp_path / 'recipe.json'
    recipe.write_text(json.dumps({'output': {
        'path': str(weights), 'sha256': hashlib.sha256(b'correct model').hexdigest(),
        'size': len(b'correct model')}}))
    receipt = tmp_path / 'result.json'
    monkeypatch.setattr(prepare_dolphin, 'RECIPE', recipe)
    monkeypatch.setattr(sys, 'argv', ['prepare_dolphin', '--receipt', str(receipt)])
    with pytest.raises(ValueError, match='checksum'):
        prepare_dolphin.main()
    assert not receipt.exists()
    assert not weights.with_suffix('.verified.json').exists()
    assert weights.read_bytes() == b'wrong model'
