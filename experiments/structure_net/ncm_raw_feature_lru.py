"""Bounded mapping over immutable historical full-source FP32 TRAIN tensors."""
from collections import OrderedDict

import torch

from experiments.structure_net.ncm_memory_data import file_hash, text_hash


class RawFeatureCache:
    def __init__(self, rows, manifest, max_bytes=512 * 1024**2):
        self.rows = {r['id']: r for r in rows}
        if (len(self.rows) != len(rows) or set(self.rows) != set(manifest)
                or type(max_bytes) is not int or max_bytes < 1
                or any(text_hash(r['source']) != r['source_sha256'] for r in rows)):
            raise ValueError('unique complete original sources, manifest and positive budget required')
        self.manifest, self.max_bytes = manifest, max_bytes
        self.values = OrderedDict()
        self.resident = self.peak = self.loads = self.hits = 0

    def __getitem__(self, identity):
        if identity in self.values:
            self.hits += 1
            self.values.move_to_end(identity)
            return self.values[identity]
        row, entry = self.rows[identity], self.manifest[identity]
        size = entry['source_tokens'] * 3072 * 4
        if size > self.max_bytes:
            raise ValueError('one full source exceeds cache budget; truncation prohibited')
        while self.values and self.resident + size > self.max_bytes:
            _, evicted = self.values.popitem(last=False)
            self.resident -= evicted.numel() * 4
        if file_hash(entry['path']) != entry['sha256']:
            raise ValueError('original source feature changed')
        saved = torch.load(entry['path'], map_location='cpu', weights_only=True, mmap=True)
        value = saved['features']
        if (any(saved[k] != row[k] for k in ('id', 'source_sha256', 'alignment_sha256', 'native_prompt_sha256'))
                or value.shape != (entry['source_tokens'], 3072) or value.dtype != torch.float32
                or not torch.isfinite(value).all()):
            raise ValueError('full FP32 source feature binding or geometry differs')
        self.values[identity] = value.detach()
        self.resident += size
        self.peak = max(self.peak, self.resident)
        self.loads += 1
        return self.values[identity]

    def metrics(self):
        return {'budget_bytes': self.max_bytes, 'resident_bytes': self.resident, 'peak_resident_bytes': self.peak,
            'loads': self.loads, 'hits': self.hits, 'source_clipping': False, 'feature_dtype': 'FP32'}
