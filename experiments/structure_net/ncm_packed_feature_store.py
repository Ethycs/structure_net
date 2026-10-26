"""Bounded disk shards of individually hash-bound, lossless FP32 features."""
import hashlib
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_fp32_feature_codec as codec


class Writer:
    def __init__(self, root, *, max_shard_bytes=256*1024**2):
        self.root = Path(root)
        if type(max_shard_bytes) is not int or max_shard_bytes < 1:
            raise ValueError('positive shard bound required')
        self.root.mkdir()
        self.limit = max_shard_bytes
        self.index = (self.root/'index.jsonl').open('x')
        self.stream = None
        self.number = 0
        self.ids = set()
        self.shards = []

    def seal(self):
        if self.stream is None:
            return
        self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close()
        self.partial.rename(self.path)
        self.shards.append({'path': str(self.path), 'bytes': self.path.stat().st_size,
                            'sha256': self.digest.hexdigest()})
        self.stream = None

    def append(self, header, payload, *, metadata_record_sha256):
        identity = header['binding']['id']
        if identity in self.ids or not payload or len(payload) > self.limit:
            raise ValueError('unique identity and bounded nonempty payload required')
        if self.stream is not None and self.stream.tell()+len(payload) > self.limit:
            self.seal()
        if self.stream is None:
            self.path = self.root/f'shard_{self.number:06d}.bin'
            self.partial = self.path.with_suffix('.partial')
            self.stream = self.partial.open('xb'); self.digest = hashlib.sha256()
            self.number += 1
        entry = {'id': identity, 'path': str(self.path), 'offset_bytes': self.stream.tell(),
                 'length_bytes': len(payload), 'payload_sha256': hashlib.sha256(payload).hexdigest(),
                 'metadata_record_sha256': metadata_record_sha256, 'header': header}
        self.stream.write(payload); self.digest.update(payload)
        self.index.write(json.dumps(entry, allow_nan=False)+'\n'); self.index.flush()
        self.ids.add(identity)
        return entry

    def close(self):
        self.seal(); self.index.close()
        return self.shards


def load(entry, binding, *, source_positions, metadata_record_sha256):
    if (entry['id'] != binding['id'] or entry['header']['binding'] != binding
            or entry['metadata_record_sha256'] != metadata_record_sha256
            or type(entry['offset_bytes']) is not int or entry['offset_bytes'] < 0
            or type(entry['length_bytes']) is not int or entry['length_bytes'] < 1):
        raise ValueError('source/prompt/body/metadata feature binding differs')
    with Path(entry['path']).open('rb') as stream:
        stream.seek(entry['offset_bytes'])
        payload = stream.read(entry['length_bytes'])
    if len(payload) != entry['length_bytes'] or hashlib.sha256(payload).hexdigest() != entry['payload_sha256']:
        raise ValueError('packed feature bytes differ')
    value = codec.decode(entry['header'], payload)
    if value.shape != (source_positions, 3072) or value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise ValueError('complete finite FP32 source features required')
    return value
