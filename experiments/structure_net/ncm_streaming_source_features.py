"""Bounded TRAIN-only access to the audited lossless FP32 source feature cache."""
from collections import OrderedDict

from experiments.structure_net import ncm_additional_source_features as original
from experiments.structure_net.ncm_memory_data import text_hash


class FeatureCache:
    def __init__(self, index, manifest, *, max_bytes=2 * 1024**3):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('positive integer feature cache byte budget required')
        self.index = {r['id']: r for r in index}
        if len(self.index) != len(index) or set(self.index) != set(manifest):
            raise ValueError('complete unique source metadata and feature identities required')
        self.manifest, self.max_bytes = manifest, max_bytes
        self.values = OrderedDict()
        self.resident_bytes = self.peak_resident_bytes = self.loads = self.hits = 0

    def get(self, source):
        identity = source['id']
        entry = self.index[identity]
        if (source['source_sha256'] != entry['source_sha256']
                or source['source_component'] != entry['source_component']
                or text_hash(source['source']) != source['source_sha256']):
            raise ValueError('TRAIN label/source feature identity or family differs')
        if identity in self.values:
            self.hits += 1
            self.values.move_to_end(identity)
            return self.values[identity]
        if original.file_hash(entry['path']) != entry['sha256']:
            raise ValueError('prepared complete source metadata changed')
        size = entry['source_positions'] * 3072 * 4
        if size > self.max_bytes:
            raise ValueError('one complete source exceeds declared cache budget; clipping prohibited')
        while self.values and self.resident_bytes + size > self.max_bytes:
            _, evicted = self.values.popitem(last=False)
            self.resident_bytes -= evicted.numel() * evicted.element_size()
        row = original.q.read(entry['path'])
        if (row['id'] != identity or row['source_sha256'] != source['source_sha256']
                or row['source_component'] != source['source_component'] or row['source'] != source['source']
                or row['source_truncated']):
            raise ValueError('exact complete TRAIN source metadata differs')
        value = original.load_feature(self.manifest[identity], row).detach()
        if value.numel() * value.element_size() != size:
            raise ValueError('complete source byte count differs from original metadata')
        self.values[identity] = value
        self.resident_bytes += size
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
        self.loads += 1
        return value

    def metrics(self):
        return {'budget_bytes': self.max_bytes, 'resident_bytes': self.resident_bytes,
            'peak_resident_bytes': self.peak_resident_bytes, 'resident_cases': len(self.values),
            'loads': self.loads, 'hits': self.hits, 'source_positions_clipped': False,
            'feature_precision': 'original FP32', 'application_inference_reuse': False}
