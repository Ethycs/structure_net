import json

import pytest
import torch

from experiments.structure_net import ncm_streaming_source_features as run


def fixture(tmp_path):
    index, manifest, rows, tensors = [], {}, [], {}
    for identity in ('a', 'b', 'c'):
        source = 'complete public unit fixture ' + identity
        row = {'id': identity, 'source': source, 'source_sha256': run.text_hash(source),
            'source_component': identity, 'source_truncated': False, 'source_ids': [4, 7],
            'alignment_sha256': identity * 64, 'native_prompt_sha256': identity * 64}
        tensors[identity] = torch.arange(2 * 3072, dtype=torch.float32).reshape(2, 3072) / 128
        meta = tmp_path / (identity + '.json')
        meta.write_text(json.dumps(row))
        header, payload = run.original.codec.encode(tensors[identity], 'byte_shuffled')
        header['binding'] = run.original.binding(row)
        hp, pp = tmp_path / (identity + '.header.json'), tmp_path / (identity + '.zlib')
        hp.write_text(json.dumps(header))
        pp.write_bytes(payload)
        index.append({'id': identity, 'path': str(meta), 'sha256': run.original.file_hash(meta),
            'source_sha256': row['source_sha256'], 'source_component': identity, 'source_positions': 2})
        manifest[identity] = {'header_path': str(hp), 'header_sha256': run.original.file_hash(hp),
            'payload_path': str(pp), 'payload_sha256': run.original.file_hash(pp), 'source_tokens': 2}
        rows.append(row)
    return index, manifest, rows, tensors


def test_real_lossless_payloads_LRU_eviction_and_complete_FP32(tmp_path):
    index, manifest, rows, tensors = fixture(tmp_path)
    size = tensors['a'].numel() * 4
    cache = run.FeatureCache(index, manifest, max_bytes=2 * size)
    for i in (0, 1, 0, 2):
        actual = cache.get(rows[i])
        assert torch.equal(actual.view(torch.uint8), tensors[rows[i]['id']].view(torch.uint8))
    assert list(cache.values) == ['a', 'c']
    assert cache.metrics()['loads'] == 3 and cache.metrics()['hits'] == 1
    assert cache.metrics()['peak_resident_bytes'] == 2 * size
    assert cache.get(rows[1]).shape == (2, 3072)
    assert cache.metrics()['resident_bytes'] <= 2 * size


def test_changed_source_on_cache_hit_and_damaged_payload_rejected(tmp_path):
    index, manifest, rows, _ = fixture(tmp_path)
    cache = run.FeatureCache(index, manifest)
    cache.get(rows[0])
    with pytest.raises(ValueError, match='identity'):
        cache.get({**rows[0], 'source': rows[0]['source'] + ' changed'})
    from pathlib import Path
    Path(manifest['b']['payload_path']).write_bytes(b'broken')
    with pytest.raises(ValueError, match='changed'):
        cache.get(rows[1])


def test_oversize_complete_source_refused_and_index_not_silently_deduplicated(tmp_path):
    index, manifest, rows, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='clipping prohibited'):
        run.FeatureCache(index, manifest, max_bytes=1).get(rows[0])
    with pytest.raises(ValueError, match='unique'):
        run.FeatureCache([*index, index[0]], manifest)
