"""Bounded random access to immutable, hash-bound UTF-8 JSONL records."""
import hashlib
import json
import os
from pathlib import Path

from torch.utils.data import Dataset

from experiments.structure_net.ncm_memory_data import file_hash


class IndexedJSONL(Dataset):
    def __init__(self, path, entries, expected_sha256):
        self.path = Path(path)
        if file_hash(self.path) != expected_sha256:
            raise ValueError('immutable JSONL payload hash differs')
        self.entries = list(entries)
        if len({r['id'] for r in self.entries}) != len(self.entries):
            raise ValueError('unique indexed record identities required')
        self.signature = self._signature()
        previous = 0
        for row in self.entries:
            if (type(row['offset_bytes']) is not int or type(row['length_bytes']) is not int
                    or row['offset_bytes'] != previous or row['length_bytes'] < 1):
                raise ValueError('complete contiguous byte index required')
            previous += row['length_bytes']
        if previous != self.signature[2]:
            raise ValueError('byte index does not cover the complete payload')
        self._stream = self._pid = None

    def _signature(self):
        stat = self.path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def __len__(self):
        return len(self.entries)

    def __getstate__(self):
        return {**self.__dict__, '_stream': None, '_pid': None}

    def close(self):
        if self._stream is not None:
            self._stream.close()
        self._stream = self._pid = None

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError('valid nonnegative record index required')
        if self._signature() != self.signature:
            raise ValueError('immutable JSONL file changed after opening')
        if self._pid != os.getpid() or self._stream is None:
            self.close()
            self._stream, self._pid = self.path.open('rb'), os.getpid()
        entry = self.entries[index]
        self._stream.seek(entry['offset_bytes'])
        raw = self._stream.read(entry['length_bytes'])
        if len(raw) != entry['length_bytes'] or hashlib.sha256(raw).hexdigest() != entry['record_sha256']:
            raise ValueError('indexed record bytes changed')
        value = json.loads(raw)
        if value['id'] != entry['id']:
            raise ValueError('indexed record identity differs')
        return value

    def __getitems__(self, indices):
        # PyTorch DataLoader uses this batched path when available. Reading in
        # file order reduces seeks; restore the requested order, including repeats.
        records = {i: self[i] for i in sorted(set(indices), key=lambda j: self.entries[j]['offset_bytes'])}
        return [records[i] for i in indices]


def write_record(stream, index, row, **metadata):
    if 'id' not in row:
        raise ValueError('record identity required')
    payload = (json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode()
    entry = {**metadata, 'id': row['id'], 'offset_bytes': stream.tell(), 'length_bytes': len(payload),
        'record_sha256': hashlib.sha256(payload).hexdigest()}
    stream.write(payload)
    index.write(json.dumps(entry, allow_nan=False)+'\n')
    return entry
