import hashlib

import pytest

from experiments.structure_net.audit_ncm_public_training_catalog import accounting, verify_index


def test_accounting_requires_both_disjoint_partitions_and_full_population():
    counts = {'original_rows': 204045, 'original_candidates': 163051, 'prior_screen_exclusions': 40994,
              'retained_pairs': 162000, 'additional_excluded_rows': 1051}
    accounting(counts)
    for key in counts:
        with pytest.raises(ValueError, match='accounting'):
            accounting({**counts, key: counts[key]+1})


def test_index_audit_detects_modified_utf8_offsets_payloads_and_pairing():
    payload = 'café\n'.encode()
    row = {k: 'fixed' for k in ('id', 'source_sha256', 'target_sha256', 'native_frame_sha256',
            'source_cl100k_tokens', 'stored_tokens', 'native_lexical_tokens', 'public_provenance')}
    entry = {**row, 'offset_bytes': 50, 'length_bytes': len(payload),
             'record_sha256': hashlib.sha256(payload).hexdigest()}
    verify_index(entry, row, payload, 50)
    for changed in ({**entry, 'offset_bytes': 49}, {**entry, 'length_bytes': len('café\n')},
                    {**entry, 'id': 'swapped'}, {**entry, 'record_sha256': 'changed'}):
        with pytest.raises(ValueError, match='binding'):
            verify_index(changed, row, payload, 50)
