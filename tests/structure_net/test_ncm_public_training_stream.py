import copy

import pytest
import torch
from torch.utils.data import DataLoader

from experiments.structure_net import ncm_public_training_stream as stream


class Records:
    def __init__(self, rows, entries):
        self.rows, self.entries = rows, entries

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def fixture(tmp_path):
    pairs, labels, meta, packed = [], [], [], []
    writer = stream.features.Writer(tmp_path/'features')
    for i in range(2):
        pair = {'id': str(i), 'source_sha256': 's'+str(i), 'target_sha256': 't'+str(i),
                'native_frame_sha256': 'n'+str(i), 'split': 'TRAIN', 'native_word_ids': [101]*128}
        code = {**{k: pair[k] for k in ('id', 'source_sha256', 'target_sha256', 'native_frame_sha256')},
                'codec_sha256': 'codec', 'codes': [i]*641}
        metadata = {'id': str(i), 'source_sha256': pair['source_sha256'], 'catalog_position': i,
                    'record_sha256': 'record'+str(i), 'alignment_sha256': 'a'+str(i),
                    'native_prompt_sha256': 'p'+str(i), 'source_positions': i+1}
        values = torch.arange((i+1)*3072, dtype=torch.float32).reshape(i+1, 3072)
        header, payload = stream.features.codec.encode(values)
        header['binding'] = {**{k: metadata[k] for k in ('id', 'source_sha256', 'alignment_sha256', 'native_prompt_sha256')},
                             'original_body_sha256': 'body'}
        packed.append(writer.append(header, payload, metadata_record_sha256=metadata['record_sha256']))
        pairs.append(pair); labels.append(code); meta.append(metadata)
    writer.close()
    pair_records = Records(pairs, copy.deepcopy(pairs))
    code_records = Records(labels, [{**r, 'catalog_position': i} for i, r in enumerate(labels)])
    return stream.PublicTrainingDataset(pair_records, code_records, meta, packed,
                                        body_sha256='body', codec_sha256='codec')


def test_batched_disk_reads_preserve_order_duplicates_and_full_matrices(tmp_path):
    dataset = fixture(tmp_path)
    rows = next(iter(DataLoader(dataset, batch_sampler=[[1, 0, 1]], collate_fn=stream.collate_records)))
    assert [r['id'] for r in rows] == ['1', '0', '1']
    assert [len(r['features']) for r in rows] == [2, 1, 2]
    assert torch.equal(rows[0]['features'], rows[2]['features'])
    assert all(r['features'].dtype == torch.float32 for r in rows)
    assert 'source' not in rows[0] and len(rows[0]['oracle_codes']) == 641


def test_wrong_source_or_target_cannot_be_joined(tmp_path):
    dataset = fixture(tmp_path)
    dataset.codes.rows[0]['target_sha256'] = 'wrong'
    with pytest.raises(ValueError): dataset[0]
    dataset.features.reverse()
    with pytest.raises(ValueError):
        stream.PublicTrainingDataset(dataset.pairs, dataset.codes, dataset.metadata, dataset.features,
                                     body_sha256='body', codec_sha256='codec')


def test_partial_population_and_wrong_metadata_are_rejected(tmp_path):
    dataset = fixture(tmp_path)
    with pytest.raises(ValueError):
        stream.PublicTrainingDataset(dataset.pairs, dataset.codes, dataset.metadata[:-1], dataset.features,
                                     body_sha256='body', codec_sha256='codec')
    dataset.features[0]['metadata_record_sha256'] = 'different'
    with pytest.raises(ValueError): dataset[0]
