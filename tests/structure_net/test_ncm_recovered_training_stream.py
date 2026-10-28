import copy
from dataclasses import dataclass
import hashlib
import json
import struct

import pytest
import torch

from experiments.structure_net import ncm_recovered_training_stream as stream
from experiments.structure_net.ncm_indexed_jsonl import IndexedJSONL, write_record


def fixture(tmp_path):
    metadata, packed = [], []
    writer = stream.features.Writer(tmp_path/'features')
    for i in range(2):
        meta = dict(id=f'context{i}', source_sha256='same-text', source_component='component',
            record_sha256=f'record{i}', alignment_sha256=f'alignment{i}', native_prompt_sha256=f'prompt{i}',
            source_positions=i+1)
        matrix = torch.full((i+1, 3072), float(i), dtype=torch.float32)
        header, raw = stream.features.codec.encode(matrix)
        header['binding'] = {**{k: meta[k] for k in ('id', 'source_sha256', 'alignment_sha256', 'native_prompt_sha256')},
                             'original_body_sha256': 'body'}
        packed.append(writer.append(header, raw, metadata_record_sha256=meta['record_sha256']))
        metadata.append(meta)
    writer.close()
    code_entries = []
    with (tmp_path/'pairs.jsonl').open('wb') as pairs, (tmp_path/'pairs_index.jsonl').open('w') as idx, (tmp_path/'codes.bin').open('wb') as codes:
        for i, context in enumerate((0, 0, 1)):
            words = [101, 17+i, 101] + [0]*125
            frame = hashlib.sha256(struct.pack('<128H', *words)).hexdigest()
            raw = struct.pack('<641H', *([i]*641))
            row = dict(id=f'pair{i}', split='TRAIN_noisy_candidate', source_record_id=f'context{context}',
                source_sha256='same-text', source_component='component', target_sha256=f'target{i}',
                native_frame_sha256=frame, native_word_ids=words, native_lexical_tokens=1)
            write_record(pairs, idx, row)
            code_entries.append(dict(position=i, offset_bytes=codes.tell(), length_bytes=len(raw),
                frame_sha256=frame, pair_ids=[row['id']], code_sha256=hashlib.sha256(raw).hexdigest()))
            codes.write(raw)
    entries = [json.loads(s) for s in (tmp_path/'pairs_index.jsonl').read_text().splitlines()]
    pairs = IndexedJSONL(tmp_path/'pairs.jsonl', entries, stream.file_hash(tmp_path/'pairs.jsonl'))
    codes = stream.CodeStore(tmp_path/'codes.bin', code_entries,
        expected_sha256=stream.file_hash(tmp_path/'codes.bin'), codec_sha256='codec')
    return stream.RecoveredTrainingDataset(pairs, codes, metadata, packed, body_sha256='body', codec_sha256='codec')


def test_variants_share_source_reads_without_conflating_context(tmp_path, monkeypatch):
    dataset = fixture(tmp_path)
    original, calls = stream.features.load, []
    def tracked(entry, *args, **kwargs):
        calls.append(entry['id']); return original(entry, *args, **kwargs)
    monkeypatch.setattr(stream.features, 'load', tracked)
    rows = dataset.__getitems__([2, 0, 1, 2])
    assert [r['id'] for r in rows] == ['pair2', 'pair0', 'pair1', 'pair2']
    assert len(calls) == 2
    assert rows[1]['features'] is rows[2]['features']
    assert [r['features'].shape[0] for r in rows] == [2, 1, 1, 2]
    assert [r['oracle_codes'][0] for r in rows] == [2, 0, 1, 2]
    assert all(r['features'].dtype == torch.float32 and len(r['native_word_ids']) == 128 for r in rows)


def test_wrong_context_metadata_and_missing_source_rejected(tmp_path):
    dataset = fixture(tmp_path)
    metadata, entries = list(dataset.metadata.values()), list(dataset.features.values())
    bad = copy.deepcopy(entries); bad[0]['header']['binding']['native_prompt_sha256'] = 'wrong-role-or-date'
    for values in (bad, entries[:1]):
        with pytest.raises(ValueError):
            stream.RecoveredTrainingDataset(dataset.pairs, dataset.codes, metadata, values,
                body_sha256='body', codec_sha256='codec')


def test_changed_binary_and_cross_target_code_binding_rejected(tmp_path):
    dataset = fixture(tmp_path)
    pair = dataset.pairs[0]
    changed = {**pair, 'native_frame_sha256': dataset.pairs[1]['native_frame_sha256']}
    with pytest.raises(ValueError, match='binding'): dataset.codes.get(changed)
    with dataset.codes.path.open('r+b') as f: f.write(b'\xff\xff')
    with pytest.raises(ValueError, match='binary changed'): dataset.codes.get(pair)


def test_explicit_pilot_subset_keeps_catalog_mapping(tmp_path):
    dataset = fixture(tmp_path)
    subset = stream.RecoveredTrainingDataset(dataset.pairs, dataset.codes, list(dataset.metadata.values()),
        [dataset.features['context1']], body_sha256='body', codec_sha256='codec', pair_positions=[2])
    assert subset[0]['id'] == 'pair2'
    with pytest.raises(IndexError): subset[-1]


def test_mixed_sources_preserve_schedule_and_reviewed_anchors(tmp_path):
    dataset = fixture(tmp_path)
    @dataclass
    class Example:
        population: str
        index: int
        noisy: bool = False
    class Cache:
        def get(self, row): return torch.ones(1, 3072)
    anchors = [dict(id='anchor', split='TRAIN', source_sha256='anchor-source',
                    native_word_ids=[101]*128, oracle_codes=[3]*641)]
    provider = stream.MixedSources(dataset, anchors, Cache())
    rows = provider.fetch([Example('recovered', 2), Example('anchor', 0, True), Example('recovered', 0)])
    assert [r['id'] for r in rows] == ['pair2', 'anchor', 'pair0']
    assert rows[1]['presentation']['noisy'] is True
    with pytest.raises(ValueError): provider.fetch([Example('public', 0)])
