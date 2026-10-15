import pytest
from experiments.neural_architecture_lab.finalize_llama_q4_kernel import audit,verify_producer,HYPOTHESIS_ID,MODEL_SHA,digest,sha256,write_json


def producer(root):
    copies=root/'producer_source';copies.mkdir()
    source=copies/'source.py';source.write_text('frozen kernel')
    p={'hypothesis_id':HYPOTHESIS_ID,'model_sha256':MODEL_SHA,
       'source_hashes':{'source.py':sha256(source)},'config':{'family':'packed','repeats':3}}
    p['fingerprint']=digest(p);write_json(root/'producer.json',p)
    return p


def test_fingerprint_and_weight_identity_are_bound(tmp_path):
    p=producer(tmp_path);assert verify_producer(tmp_path)==p
    p['model_sha256']='0'*64;write_json(tmp_path/'producer.json',p)
    with pytest.raises(ValueError,match='fingerprint/model'):verify_producer(tmp_path)


def test_duplicate_timing_cells_cannot_pass_the_audit(tmp_path):
    p=producer(tmp_path)
    row={'id':'request','arm':'packed','repeat':0}
    write_json(tmp_path/'rows.json',{'status':'completed','rows':[row,row,row]})
    write_json(tmp_path/'result.json',{'status':'completed_pending_source_review','producer_fingerprint':p['fingerprint'],
        'rows_sha256':sha256(tmp_path/'rows.json'),'arm':'packed'})
    with pytest.raises(ValueError,match='duplicate'):audit(tmp_path,{'request':{}},'packed',3)
