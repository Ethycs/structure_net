from collections import Counter
import json
import sqlite3

from experiments.structure_net import ncm_available_pair_census as census
from experiments.structure_net.ncm_available_pair_deep_join import remaining_joins


def test_targets_are_not_source_evidence_and_ambiguous_children_are_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(census,'STORAGE',tmp_path)
    db=census.open_census()
    db.executescript('''
      CREATE TABLE source_texts(sha TEXT PRIMARY KEY,text TEXT,origin TEXT);
      CREATE TABLE hash_labels(id TEXT,source_sha TEXT,target TEXT,metadata TEXT);
      CREATE TABLE section_targets(sid TEXT,target_sha TEXT,target TEXT);
      CREATE TABLE structural_labels(id TEXT,sid TEXT,children TEXT,target TEXT);
    ''')
    base=sqlite3.connect(':memory:')
    base.executescript('CREATE TABLE texts(sha TEXT,text TEXT); CREATE TABLE verified_inputs(sha TEXT);')
    hidden='a target-only string'
    base.execute('INSERT INTO texts VALUES(?,?)',(census.digest(hidden),hidden))
    db.execute('INSERT INTO hash_labels VALUES(?,?,?,?)',('missing',census.digest(hidden),'bad target','{}'))
    actual='actual raw source';sha=census.digest(actual)
    base.execute('INSERT INTO texts VALUES(?,?)',(sha,actual));base.execute('INSERT INTO verified_inputs VALUES(?)',(sha,))
    db.execute('INSERT INTO hash_labels VALUES(?,?,?,?)',('valid',sha,'valid target','{}'))
    db.executemany('INSERT INTO section_targets VALUES(?,?,?)',[('a','1','one'),('a','2','alternative'),('b','3','two')])
    db.executemany('INSERT INTO structural_labels VALUES(?,?,?,?)',[
        ('ambiguous','p',json.dumps(['a','b']),'bad merge'),('valid','q',json.dumps(['b']),'good merge')])
    counts=Counter();remaining_joins(db,base,counts)
    assert counts['hash_label_missing_source']==1
    assert counts['structural_labels_missing_or_ambiguous_children']==1
    assert counts['hash_labels_joined']==counts['structural_labels_joined']==1
    assert db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]==2
    db.close();base.close()
