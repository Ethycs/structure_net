import copy

import pytest
import torch

from experiments.structure_net import ncm_packed_feature_store as store


def encoded(identity, values):
    header, payload = store.codec.encode(values)
    header['binding'] = {'id': identity, 'source_sha256': 's', 'native_prompt_sha256': 'p',
                         'alignment_sha256': 'a', 'original_body_sha256': 'b'}
    return header, payload


def test_shards_roundtrip_random_order_and_duplicates(tmp_path):
    values = [torch.arange(6144, dtype=torch.float32).reshape(2, 3072)+i for i in range(3)]
    pairs = [encoded(str(i), value) for i, value in enumerate(values)]
    writer = store.Writer(tmp_path/'features', max_shard_bytes=max(len(p[1]) for p in pairs))
    entries = [writer.append(*p, metadata_record_sha256=str(i)) for i, p in enumerate(pairs)]
    assert len(writer.close()) == 3
    for i in (2, 0, 1, 2):
        actual = store.load(entries[i], pairs[i][0]['binding'], source_positions=2, metadata_record_sha256=str(i))
        assert torch.equal(actual.view(torch.uint8), values[i].view(torch.uint8))
    assert not list((tmp_path/'features').glob('*.partial'))


def test_binding_and_payload_tampering_rejected(tmp_path):
    pair = encoded('one', torch.ones(1, 3072))
    writer = store.Writer(tmp_path/'features')
    entry = writer.append(*pair, metadata_record_sha256='m')
    with pytest.raises(ValueError):
        writer.append(*pair, metadata_record_sha256='m')
    writer.close()
    for bad in ({**pair[0]['binding'], 'original_body_sha256': 'wrong'}, {'id': 'one'}):
        with pytest.raises(ValueError):
            store.load(entry, bad, source_positions=1, metadata_record_sha256='m')
    with pytest.raises(ValueError):
        store.load(entry, pair[0]['binding'], source_positions=1, metadata_record_sha256='wrong')
    changed = copy.deepcopy(entry); changed['offset_bytes'] += 1
    with pytest.raises(ValueError):
        store.load(changed, pair[0]['binding'], source_positions=1, metadata_record_sha256='m')


def test_large_payload_cannot_exceed_bound(tmp_path):
    writer = store.Writer(tmp_path/'features', max_shard_bytes=1)
    with pytest.raises(ValueError):
        writer.append(*encoded('one', torch.ones(1, 3072)), metadata_record_sha256='m')
    assert writer.close() == []
