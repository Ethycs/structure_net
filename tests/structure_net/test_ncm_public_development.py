import json
from types import SimpleNamespace

import pytest

from experiments.structure_net import ncm_public_development as development


def test_public_guard_binds_exact_screened_dev_and_rejects_source_overlap(tmp_path, monkeypatch):
    catalog = SimpleNamespace(AUDIT=tmp_path/'audit', INSPECTION_AUDIT=tmp_path/'inspection_audit',
                              INSPECT=tmp_path/'inspection', DEST=tmp_path/'catalog', REVISION='revision')
    source = tmp_path/'sources'
    for path in (catalog.INSPECT, catalog.DEST, source): path.mkdir()
    (source/'cases.json').write_text('fixed source population')
    def write(path, value): path.write_text(json.dumps(value))
    write(catalog.INSPECT/'producer.json', {'input_hashes': {
        str(source/'cases.json'): development.data.sha(source/'cases.json')}})
    write(catalog.INSPECT/'result.json', {'open_source_and_protected_exact_hash_checks_completed': True,
                                      'counts': {'rows': 204045}})
    entry = {'id': 'public', 'source_sha256': 'public_hash', 'public_provenance': {'article_id': '123'}}
    write(catalog.DEST/'training_index.jsonl', entry)
    monkeypatch.setattr(development.study.pipeline.bundle, 'catalog', catalog)
    monkeypatch.setattr(development.data, 'completed', lambda root: {})
    monkeypatch.setattr(development.study, 'CONFIG', SimpleNamespace(public_cases=1))
    evaluation = SimpleNamespace(old=SimpleNamespace(SOURCE=source, COHORTS=('dev',)))
    cohorts = {'dev': [{'id': 'held', 'source_id': 'held', 'group_id': 'held',
                        'source_component': 'held', 'source_sha256': 'held_hash'}]}
    assert development.public_guard(evaluation, cohorts)['identity_overlaps'] == 0
    cohorts['dev'][0]['source_sha256'] = 'public_hash'
    with pytest.raises(ValueError, match='overlap'): development.public_guard(evaluation, cohorts)
    (source/'cases.json').write_text('different source population')
    with pytest.raises(ValueError, match='exact DEV'): development.public_guard(evaluation, cohorts)
