import copy
import hashlib
import io
import json

import pytest

from experiments.structure_net import audit_ncm_public_source_metadata as audit


def case():
    pair = {key: 'value' for key in audit.SOURCE_FIELDS}
    pair.update(source_truncated=False)
    row = {**pair, 'source_ids': [4], 'context_ids': [5, 6], 'source_frame': 128,
        'alignment': [[[1, 1.0]]], 'alignment_sha256': 'a', 'native_prompt_sha256': 'b'}
    entry = {key: row[key] for key in ('id', 'source_sha256', 'source_component', 'source_frame',
                                     'alignment_sha256', 'native_prompt_sha256')}
    entry.update(catalog_position=2, source_positions=1, native_prompt_positions=2)
    return pair, row, entry


def test_full_source_and_target_boundary():
    pair, row, entry = case()
    audit.verify_record(row, entry, pair, 2, copy.deepcopy(row), {'target_text'})
    for change in ({'source': 'clipped'}, {'target_text': 'leaked'}):
        changed = {**row, **change}
        with pytest.raises(ValueError):
            audit.verify_record(changed, entry, pair, 2, changed, {'target_text'})


@pytest.mark.parametrize('values', [[], [[2, 1.0]], [[1, float('nan')]], [[1, -1.0]], [[1, 0.5]]])
def test_invalid_alignment_rejected_even_when_replayed(values):
    pair, row, entry = case()
    row['alignment'] = [values]
    with pytest.raises(ValueError):
        audit.verify_record(row, entry, pair, 2, row, set())


def test_disk_binding_and_identity():
    payload = json.dumps({'id': 'first'}).encode()
    entry = {'id': 'first', 'offset_bytes': 0, 'length_bytes': len(payload),
             'record_sha256': hashlib.sha256(payload).hexdigest()}
    assert audit.read_record(io.BytesIO(payload), entry) == {'id': 'first'}
    with pytest.raises(ValueError):
        audit.read_record(io.BytesIO(payload), {**entry, 'id': 'second'})
    with pytest.raises(ValueError):
        audit.read_record(io.BytesIO(payload[:-1]), entry)


def test_complete_order_and_payload_coverage(tmp_path):
    index = tmp_path/'index.jsonl'
    payload = tmp_path/'metadata.jsonl'
    payload.write_bytes(b'ab')
    entries = [{'id': str(i)} for i in range(2)]
    saved = [{'id': str(i), 'catalog_position': i, 'offset_bytes': i, 'length_bytes': 1} for i in range(2)]
    index.write_text(''.join(json.dumps(row)+'\n' for row in saved))
    assert len(list(audit.tasks(index, entries, range(2), batch_size=1))) == 2
    with pytest.raises(ValueError):
        list(audit.tasks(index, entries, range(1)))
    payload.write_bytes(b'abc')
    with pytest.raises(ValueError):
        list(audit.tasks(index, entries, range(2)))
