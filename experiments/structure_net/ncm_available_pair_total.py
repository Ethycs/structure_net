"""Reconcile all completed census stages into one content-deduplicated supply."""
import argparse
import json
from pathlib import Path
import sqlite3
import subprocess

from experiments.structure_net import ncm_available_pair_census as census
from experiments.structure_net import ncm_available_pair_supplement as supplement
from experiments.structure_net import ncm_available_pair_deep_join as deep

archive=census.archive
ROOT=Path('data/experiments/ncm_paper_regime/20261027_available_pair_total_v1')
FIT="exact_copy=0 AND target_tokens BETWEEN 1 AND 128 AND input_tokens<=2048 AND (input_kind='children' OR input_tokens>=192)"


def summarize(db,where='1'):
    result=[]
    for kind, in db.execute('SELECT DISTINCT input_kind FROM pairs WHERE '+where+' ORDER BY input_kind'):
        row=db.execute('''SELECT COUNT(*),COUNT(DISTINCT input_sha),SUM(exact_copy),
          SUM(exact_copy=0),SUM(CASE WHEN '''+FIT+''' THEN 1 ELSE 0 END),
          COUNT(DISTINCT CASE WHEN '''+FIT+''' THEN input_sha END),
          SUM(CASE WHEN '''+FIT+''' THEN input_tokens ELSE 0 END),
          SUM(CASE WHEN '''+FIT+''' THEN target_tokens ELSE 0 END)
          FROM pairs WHERE ('''+where+''') AND input_kind=?''',(kind,)).fetchone()
        item=dict(zip(('kind','distinct_pairs','distinct_inputs','exact_copies','noncopy_pairs',
            'application_length_fit_pairs','application_length_fit_distinct_inputs','input_tokens_across_fit_pairs',
            'target_tokens_across_fit_pairs'),(kind,*row)))
        # Disjoint buckets provide an independent partition of each population.
        parts=dict(db.execute('''SELECT CASE WHEN exact_copy=1 THEN 'exact_copy'
          WHEN input_kind!='children' AND input_tokens<192 THEN 'below_raw_summarization_threshold'
          WHEN input_tokens>2048 THEN 'input_over_2048'
          WHEN target_tokens>128 THEN 'target_over_128'
          ELSE 'application_length_fit' END,COUNT(*) FROM pairs
          WHERE ('''+where+''') AND input_kind=? GROUP BY 1''',(kind,)))
        if sum(parts.values())!=item['distinct_pairs'] or parts.get('application_length_fit',0)!=item['application_length_fit_pairs']:
            raise ValueError('length partition does not reconcile')
        item['disjoint_length_partition']=parts;result.append(item)
    return result


def main():
    if ROOT.exists():raise ValueError('fresh total reconciliation required')
    prerequisites=[(deep.BASE,'finish_result.json'),(supplement.ROOT,'result.json'),(deep.ROOT,'result.json'),(deep.SCAN,'result.json')]
    for root,name in prerequisites:
        if not (root/name).exists():raise ValueError('completed census stage required')
    ROOT.mkdir();census.STORAGE=ROOT;db=census.open_census()
    db.execute('CREATE TABLE pair_stages(id TEXT,stage TEXT,PRIMARY KEY(id,stage))')
    stages=[]
    for name,root in [('indexed',deep.BASE),('body_and_pointer',supplement.ROOT),('deep',deep.ROOT)]:
        db.execute('ATTACH DATABASE ? AS incoming',((root/'pairs.sqlite').resolve().as_uri()+'?mode=ro',))
        inconsistent=db.execute('''SELECT COUNT(*) FROM incoming.pairs p JOIN pairs q ON q.id=p.id
          WHERE p.input_kind!=q.input_kind OR p.input_sha!=q.input_sha OR p.target_sha!=q.target_sha
          OR p.input_tokens!=q.input_tokens OR p.target_tokens!=q.target_tokens OR p.exact_copy!=q.exact_copy''').fetchone()[0]
        if inconsistent:raise ValueError('same pair identity has inconsistent content or length')
        before=db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]
        db.execute('INSERT OR IGNORE INTO pairs SELECT * FROM incoming.pairs')
        added=db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]-before
        db.execute('INSERT INTO pair_stages SELECT id,? FROM incoming.pairs',(name,));db.commit()
        stage={'stage':name,'pairs_in_stage':db.execute('SELECT COUNT(*) FROM incoming.pairs').fetchone()[0],
            'new_pairs_after_prior_stages':added,'cumulative':summarize(db)}
        stages.append(stage);db.execute('DETACH DATABASE incoming')
    by_kind=summarize(db)
    ordinary=[r for r in by_kind if r['kind'] in ('raw','children')]
    # Explicitly retain unresolved cache and pointer denominators.
    db.execute('CREATE TABLE recovered_cache_origins(origin TEXT PRIMARY KEY)')
    for root in (supplement.ROOT,deep.ROOT):
        c=census.readonly(root/'pairs.sqlite')
        db.executemany('INSERT OR IGNORE INTO recovered_cache_origins VALUES(?)',c.execute("SELECT DISTINCT origin FROM occurrences WHERE origin LIKE 'cache:%'"));c.close()
    db.execute('ATTACH DATABASE ? AS original',((deep.BASE/'pairs.sqlite').resolve().as_uri()+'?mode=ro',))
    unresolved_caches=dict(db.execute('''SELECT category,COUNT(*) FROM original.cached_labels c
      WHERE c.pair_id IS NULL AND NOT EXISTS(SELECT 1 FROM recovered_cache_origins r WHERE r.origin='cache:'||c.file_sha) GROUP BY category'''))
    source_missing=db.execute("SELECT COUNT(*) FROM original.issues WHERE kind='atomic' AND reason='missing source turn'").fetchone()[0]
    db.execute('DETACH DATABASE original');db.commit()
    units=['finish-recovery','discovery','source-package','extra-labels','supplement','deep-scan','deep-join']
    states={u:subprocess.check_output(['systemctl','--user','show','structure-net-ncm-available-pair-'+u+'-v1.service',
        '-p','ActiveState,SubState,MainPID,ExecMainPID,ExecMainStatus,InvocationID,NRestarts'],text=True) for u in units}
    if any('SubState=exited' not in s or 'ExecMainStatus=0' not in s for s in states.values()):
        raise ValueError('all census workers must have an observed successful terminal')
    value={'status':'available_pair_content_census_reconciled','ordinary_source_summary_pairs':sum(r['distinct_pairs'] for r in ordinary),
        'ordinary_noncopy_source_summary_pairs':sum(r['noncopy_pairs'] for r in ordinary),
        'ordinary_application_length_fit_pairs':sum(r['application_length_fit_pairs'] for r in ordinary),
        'by_input_kind':by_kind,'stage_reconciliation':stages,'remaining_unjoined_cache_contents':unresolved_caches,
        'historical_missing_atomic_source_occurrences_before_supplements':source_missing,
        'supplement_result':archive.read(supplement.ROOT/'result.json'),'deep_result':archive.read(deep.ROOT/'result.json'),
        'archive_scan_result':{k:v for k,v in archive.read(deep.BASE/'extra_scan_result.json').items() if k!='unparsed_large_JSON'},
        'deep_scan_result':archive.read(deep.SCAN/'result.json'),
        'scope':{'core_stores':500,'archive_bundles':798,'original_source_package_files':6},
        'interpretation':{'content_identity':'input kind + exact source text hash + exact target text hash; no namespace alias',
            'length_tokenizer':'cl100k_base','raw_input_tokens':[192,2048],'merge_input_token_max':2048,'target_token_max':128,
            'source_group_split_applied':False,'semantic_label_quality_reviewed':False,'test_fixture_filter_applied':False,
            'native_codec_capacity_applied':False,'training_admitted':False,'all_theoretically_possible_pairs_claimed':False,
            'raw_structural_conversion_and_image_context_pairs_excluded_from_ordinary_total':True},
        'workers':states,'producer_sha256':archive.sha(Path(__file__)),
        'result_bindings':{str(root/name):archive.sha(root/name) for root,name in prerequisites}}
    archive.write(ROOT/'result.json',value);db.close()
    print(json.dumps({k:v for k,v in value.items() if k in ('status','ordinary_source_summary_pairs','ordinary_noncopy_source_summary_pairs','ordinary_application_length_fit_pairs','by_input_kind','remaining_unjoined_cache_contents')}),flush=True)


if __name__=='__main__':argparse.ArgumentParser(__doc__).parse_args();main()
