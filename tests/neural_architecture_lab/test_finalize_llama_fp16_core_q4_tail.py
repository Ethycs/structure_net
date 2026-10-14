import json
import pytest
from experiments.neural_architecture_lab.finalize_llama_fp16_core_q4_tail import audit,verify_producer,HYPOTHESIS_ID,digest,sha256,write_json


def producer(root):
    snap=root/'producer_source';snap.mkdir()
    source=snap/'producer.py';source.write_text('frozen producer')
    p={'hypothesis_id':HYPOTHESIS_ID,'source_hashes':{'producer.py':sha256(source)},'config':{'repeats':3}}
    p['fingerprint']=digest(p);write_json(root/'producer.json',p)
    return p


def test_producer_requires_both_fingerprint_and_frozen_source(tmp_path):
    p=producer(tmp_path)
    assert verify_producer(tmp_path)==p
    p['config']['repeats']=4
    write_json(tmp_path/'producer.json',p)
    with pytest.raises(ValueError,match='fingerprint'):verify_producer(tmp_path)
    p['config']['repeats']=3;write_json(tmp_path/'producer.json',p)
    (tmp_path/'producer_source'/'producer.py').write_text('mutated')
    with pytest.raises(ValueError,match='snapshot'):verify_producer(tmp_path)


def test_duplicate_cells_cannot_count_as_completed(tmp_path):
    p=producer(tmp_path)
    row={'id':'request','repeat':0,'arm':'core64_q4'}
    write_json(tmp_path/'rows.json',{'status':'completed','rows':[row,row,row]})
    write_json(tmp_path/'result.json',{'status':'completed_pending_source_review','producer_fingerprint':p['fingerprint'],
        'rows_sha256':sha256(tmp_path/'rows.json'),'arm':'core64_q4'})
    with pytest.raises(ValueError,match='duplicated'):audit(tmp_path,{'request':{}},'core64_q4',3)
