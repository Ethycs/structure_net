"""Bounded source/context/target joins for the recovered noisy paraphrase pool.

This consumer does not admit a training run. Campaigns must qualify the complete
producer receipts, split guards, sampler, model and optimizer before fitting.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
import struct

from torch.utils.data import Dataset

from experiments.structure_net import ncm_packed_feature_store as features
from experiments.structure_net.ncm_memory_data import file_hash


class CodeStore:
    """Read one singleton-produced code vector; retain only byte indexes in RAM."""
    def __init__(self, binary, entries, *, expected_sha256, codec_sha256):
        self.path = Path(binary)
        if file_hash(self.path) != expected_sha256:
            raise ValueError('complete immutable target-code file required')
        self.codec_sha256 = codec_sha256
        self.entries, self.pairs = {}, {}
        offset = 0
        for position, entry in enumerate(entries):
            key = entry['frame_sha256']
            if (key in self.entries or entry['position'] != position or entry['offset_bytes'] != offset
                    or entry['length_bytes'] != 1282 or not entry['pair_ids']
                    or len(set(entry['pair_ids'])) != len(entry['pair_ids'])):
                raise ValueError('complete contiguous singleton code index required')
            self.entries[key] = entry
            for identity in entry['pair_ids']:
                if identity in self.pairs: raise ValueError('each target pair must bind to exactly one code vector')
                self.pairs[identity] = key
            offset += 1282
        self.signature = self._signature()
        if not self.entries or offset != self.signature[2]:
            raise ValueError('code index must cover the complete file')

    def _signature(self):
        s = self.path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    def get(self, pair):
        key = pair['native_frame_sha256']
        if self.pairs.get(pair['id']) != key or self._signature() != self.signature:
            raise ValueError('target/code binding or immutable binary changed')
        ids, n = pair['native_word_ids'], pair['native_lexical_tokens']
        if (len(ids) != 128 or not 1 <= n <= 126 or ids[0] != 101 or ids[n+1] != 101
                or any(ids[n+2:]) or any(type(v) is not int or not 0 <= v < 30522 for v in ids)
                or set(ids[1:n+1]) & {0, 100, 101, 102, 103}
                or hashlib.sha256(struct.pack('<128H', *ids)).hexdigest() != key):
            raise ValueError('complete exact native target frame required')
        entry = self.entries[key]
        with self.path.open('rb') as f:
            f.seek(entry['offset_bytes']); raw = f.read(1282)
        if len(raw) != 1282 or hashlib.sha256(raw).hexdigest() != entry['code_sha256']:
            raise ValueError('singleton code bytes differ')
        result = list(struct.unpack('<641H', raw))
        if max(result) >= 8192: raise ValueError('target code vocabulary differs')
        return result


class RecoveredTrainingDataset(Dataset):
    def __init__(self, pairs, codes, metadata, feature_entries, *, body_sha256, codec_sha256,
                 pair_positions=None):
        self.pairs, self.codes = pairs, codes
        self.metadata = {r['id']: r for r in metadata}
        self.features = {r['id']: r for r in feature_entries}
        self.body_sha256 = body_sha256
        if (codes.codec_sha256 != codec_sha256 or len(self.metadata) != len(metadata)
                or len(self.features) != len(feature_entries)):
            raise ValueError('unique source contexts and exact frozen codec identity required')
        self.positions = tuple(range(len(pairs))) if pair_positions is None else tuple(pair_positions)
        if (not self.positions or len(set(self.positions)) != len(self.positions)
                or any(type(i) is not int or not 0 <= i < len(pairs) for i in self.positions)):
            raise ValueError('unique valid catalog pair positions required')
        self.joins = []
        used = set()
        for position in self.positions:
            pair = pairs[position]
            sid = pair['source_record_id']
            if sid not in self.metadata or sid not in self.features:
                raise ValueError('every target must have its exact conditioned source feature')
            meta, entry = self.metadata[sid], self.features[sid]
            if (pair['split'] != 'TRAIN_noisy_candidate' or codes.pairs.get(pair['id']) != pair['native_frame_sha256']
                    or pair['source_sha256'] != meta['source_sha256']
                    or pair['source_component'] != meta['source_component']
                    or entry['metadata_record_sha256'] != meta['record_sha256']
                    or entry['header']['binding'] != self.binding(meta)
                    or entry['header']['shape'] != [meta['source_positions'], 3072]):
                raise ValueError('source/context/target/feature index binding differs')
            self.joins.append({k: pair[k] for k in ('id', 'source_record_id', 'source_sha256',
                'source_component', 'target_sha256', 'native_frame_sha256')})
            used.add(sid)
        if used != set(self.features):
            raise ValueError('all supplied feature entries must belong to selected target pairs')

    def binding(self, meta):
        return {**{k: meta[k] for k in ('id', 'source_sha256', 'alignment_sha256', 'native_prompt_sha256')},
                'original_body_sha256': self.body_sha256}

    def __len__(self): return len(self.positions)

    def __getitem__(self, index): return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        if any(type(i) is not int or not 0 <= i < len(self) for i in indices):
            raise IndexError('valid recovered example positions required')
        # Read each source once per batch, even when several targets share it.
        source_ids = {self.joins[i]['source_record_id'] for i in indices}
        order = sorted(source_ids, key=lambda sid: (self.features[sid]['path'], self.features[sid]['offset_bytes']))
        matrices = {}
        for sid in order:
            meta = self.metadata[sid]
            matrices[sid] = features.load(self.features[sid], self.binding(meta),
                source_positions=meta['source_positions'], metadata_record_sha256=meta['record_sha256'])
        records = {}
        for i in sorted(set(indices), key=lambda i: self.pairs.entries[self.positions[i]]['offset_bytes']):
            pair, join = self.pairs[self.positions[i]], self.joins[i]
            if pair['split'] != 'TRAIN_noisy_candidate' or any(pair[k] != v for k, v in join.items()):
                raise ValueError('target pair changed after source join')
            records[i] = {**{k: pair[k] for k in ('id', 'source_record_id', 'source_sha256', 'target_sha256', 'native_word_ids')},
                'features': matrices[pair['source_record_id']], 'oracle_codes': self.codes.get(pair)}
        return [records[i] for i in indices]


class MixedSources:
    """One prefetch worker owns these mutable indexes and the bounded anchor cache."""
    def __init__(self, recovered, anchors, anchor_cache):
        self.recovered, self.anchors, self.anchor_cache = recovered, anchors, anchor_cache
        recovered_ids, anchor_ids = {r['id'] for r in recovered.joins}, {r['id'] for r in anchors}
        if (recovered_ids & anchor_ids or len(anchor_ids) != len(anchors)
                or any(r['split'] != 'TRAIN' or len(r['oracle_codes']) != 641
                       or len(r['native_word_ids']) != 128 for r in anchors)):
            raise ValueError('distinct complete recovered and reviewed TRAIN populations required')

    def fetch(self, examples):
        if not examples or any(e.population not in ('recovered', 'anchor') for e in examples):
            raise ValueError('nonempty bound mixed-population request required')
        values = iter(self.recovered.__getitems__([e.index for e in examples if e.population == 'recovered']))
        result = []
        for e in examples:
            if e.population == 'recovered': row = next(values)
            else:
                if type(e.index) is not int or not 0 <= e.index < len(self.anchors):
                    raise IndexError('valid reviewed anchor index required')
                original = self.anchors[e.index]
                row = {**{k: original[k] for k in ('id', 'source_sha256', 'native_word_ids', 'oracle_codes')},
                    'features': self.anchor_cache.get(original)}
            result.append({**row, 'presentation': asdict(e)})
        return result
