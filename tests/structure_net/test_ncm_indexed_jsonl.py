import json
import pickle

import pytest
from torch.utils.data import DataLoader

from experiments.structure_net import ncm_indexed_jsonl as storage


def dataset(tmp_path):
    path, idx = tmp_path/'records.jsonl', tmp_path/'index.jsonl'
    rows = [{'id': str(i), 'text': 'café — '+str(i)} for i in range(5)]
    with path.open('xb') as out, idx.open('x') as index:
        for row in rows: storage.write_record(out, index, row)
    entries = [json.loads(line) for line in idx.read_text().splitlines()]
    return storage.IndexedJSONL(path, entries, storage.file_hash(path)), rows


def test_batched_disk_reads_preserve_requested_order_and_repeated_draws(tmp_path):
    ds, rows = dataset(tmp_path)
    assert ds.__getitems__([4, 0, 4, 2]) == [rows[i] for i in (4, 0, 4, 2)]
    batches = list(DataLoader(ds, batch_size=2, shuffle=False, collate_fn=list, num_workers=0))
    assert [row for batch in batches for row in batch] == rows
    assert all('text' not in entry for entry in ds.entries)
    ds.close()


def test_reopened_worker_handle_does_not_pickle_a_live_file(tmp_path):
    ds, rows = dataset(tmp_path)
    assert ds[1] == rows[1]
    worker = pickle.loads(pickle.dumps(ds))
    assert worker._stream is None and worker[4] == rows[4]
    ds.close(); worker.close()


def test_modified_file_and_bad_identity_are_rejected(tmp_path):
    ds, _ = dataset(tmp_path)
    ds.entries[0]['id'] = 'wrong'
    with pytest.raises(ValueError, match='identity'): ds[0]
    ds.path.write_bytes(ds.path.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='changed'): ds[1]
    ds.close()


def test_incomplete_or_noncontiguous_index_rejected(tmp_path):
    ds, _ = dataset(tmp_path)
    for entries in (ds.entries[:-1], [{**ds.entries[0], 'offset_bytes': 1}, *ds.entries[1:]]):
        with pytest.raises(ValueError):
            storage.IndexedJSONL(ds.path, entries, storage.file_hash(ds.path))
    ds.close()
