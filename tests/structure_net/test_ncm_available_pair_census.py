import json
import sqlite3

from experiments.structure_net import ncm_available_pair_census as census


def test_global_pair_count_deduplicates_source_names_and_preserves_alternative_targets(tmp_path,monkeypatch):
    monkeypatch.setattr(census,'STORAGE',tmp_path)
    db=census.open_census()
    a=census.pair(db,'store1','a','raw','session1','raw','the same source','summary')
    b=census.pair(db,'store2','b','raw','session2','raw','the same source','summary')
    c=census.pair(db,'store2','c','raw','session2','raw','the same source','alternative')
    assert a==b and a!=c
    assert db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]==2
    assert db.execute('SELECT COUNT(*) FROM occurrences').fetchone()[0]==3
    assert db.execute('SELECT COUNT(DISTINCT input_sha) FROM pairs').fetchone()[0]==1
    db.close()


def test_invalid_atomic_span_is_not_a_pair_but_complete_child_input_is_counted(tmp_path,monkeypatch):
    monkeypatch.setattr(census,'STORAGE',tmp_path)
    raw=tmp_path/'memory.db';summary=tmp_path/'summary.sqlite'
    with sqlite3.connect(raw) as db:
        db.execute('CREATE TABLE turns(turn_id TEXT,text TEXT,role TEXT,source_id TEXT)')
        db.execute('INSERT INTO turns VALUES(?,?,?,?)',('t','source text','user','s'))
    span={'turn_id':'t','start_char':0,'end_char':11,'role':'user','source_id':'s',
          'turn_text_sha256':census.digest('source text'),'span_text_sha256':census.digest('source text')}
    rows=[('atomic',{'section_id':'a','summary':'short','source_id':'s','spans':[span]}),
          ('atomic',{'section_id':'b','summary':'bad','source_id':'s','spans':[{**span,'span_text_sha256':'wrong'}]}),
          ('hierarchy',{'section_id':'h','summary':'merged','source_id':'s','spans':[{**span,'turn_id':'missing'}],'child_section_ids':['a','b']}),
          ('parents',{'section_id':'p','summary':'no children','source_id':'s','spans':[]})]
    with sqlite3.connect(summary) as db:
        db.execute('CREATE TABLE sections(kind TEXT,sid TEXT,payload TEXT)')
        db.executemany('INSERT INTO sections VALUES(?,?,?)',[(k,r['section_id'],json.dumps(r)) for k,r in rows])
    db=census.open_census();result=census.scan_store(db,'test',summary,raw)
    assert result['counts']['raw_summary_candidate']==1
    assert result['counts']['hierarchy_merge_candidate']==1
    assert result['counts']['merge_raw_evidence_unverified']==1
    assert sum(result['errors'].values())==2
    assert db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]==2
    assert census.scan_store(db,'test',summary,raw)==result
    db.close()
