"""Join immutable public examples, code labels and disk-backed source features."""
from torch.utils.data import Dataset

from experiments.structure_net import ncm_packed_feature_store as features


class PublicTrainingDataset(Dataset):
    """Keep indexes in memory and load only requested complete examples.

    Construction expects already admitted artifacts. Their receipt/hash checks
    belong to the campaign loader; each access still checks record identities,
    code bindings and exact serialized feature hashes.
    """
    def __init__(self, pairs, codes, metadata, feature_entries, *, body_sha256, codec_sha256):
        self.pairs, self.codes = pairs, codes
        self.metadata, self.features = list(metadata), list(feature_entries)
        self.body_sha256, self.codec_sha256 = body_sha256, codec_sha256
        sizes = {len(pairs), len(codes), len(metadata), len(feature_entries)}
        if len(sizes) != 1 or not len(pairs):
            raise ValueError('complete equally sized public input populations required')
        ids = set()
        for i, (pair, code, meta, entry) in enumerate(zip(pairs.entries, codes.entries, metadata, feature_entries)):
            identity = pair['id']
            if (identity in ids or any(row['id'] != identity for row in (code, meta, entry))
                    or meta['catalog_position'] != i or code['catalog_position'] != i
                    or pair['source_sha256'] != meta['source_sha256']
                    or any(pair[k] != code[k] for k in ('target_sha256', 'native_frame_sha256'))
                    or entry['metadata_record_sha256'] != meta['record_sha256']
                    or entry['header']['binding'] != self.binding(meta)
                    or entry['header']['shape'] != [meta['source_positions'], 3072]):
                raise ValueError('complete ordered source/target/feature index binding required')
            ids.add(identity)

    def binding(self, meta):
        return {**{k: meta[k] for k in ('id', 'source_sha256', 'alignment_sha256', 'native_prompt_sha256')},
                'original_body_sha256': self.body_sha256}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError('valid complete public example index required')
        pair, code = self.pairs[index], self.codes[index]
        meta, entry = self.metadata[index], self.features[index]
        if (pair['id'] != meta['id'] or pair['split'] != 'TRAIN'
                or pair['source_sha256'] != meta['source_sha256']
                or any(pair[k] != code[k] for k in ('id', 'source_sha256', 'target_sha256', 'native_frame_sha256'))
                or code['codec_sha256'] != self.codec_sha256 or len(code['codes']) != 641
                or any(type(v) is not int or not 0 <= v < 8192 for v in code['codes'])
                or len(pair['native_word_ids']) != 128):
            raise ValueError('complete bound TRAIN source and native target/code labels required')
        value = features.load(entry, self.binding(meta), source_positions=meta['source_positions'],
                              metadata_record_sha256=meta['record_sha256'])
        return {'id': pair['id'], 'source_sha256': pair['source_sha256'],
                'target_sha256': pair['target_sha256'], 'features': value,
                'native_word_ids': pair['native_word_ids'], 'oracle_codes': code['codes']}

    def __getitems__(self, indices):
        if any(type(i) is not int or not 0 <= i < len(self) for i in indices):
            raise IndexError('valid public example indices required')
        order = sorted(set(indices), key=lambda i: (self.features[i]['path'], self.features[i]['offset_bytes']))
        rows = {i: self[i] for i in order}
        return [rows[i] for i in indices]


def collate_records(rows):
    """Leave full CPU matrices unpadded; the measured GPU collation handles masks."""
    return rows
