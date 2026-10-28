"""Reconcile all four source-pair formats after the qualified scan resumption."""
import json
from pathlib import Path
import subprocess

from experiments.structure_net import ncm_available_pair_total as counts
from experiments.structure_net import ncm_available_pair_total_v2 as exporter
from experiments.structure_net import ncm_available_pair_inventory_audit as audit

archive=counts.archive
census=counts.census
ROOT=Path('data/experiments/ncm_paper_regime/20261027_available_pair_total_v3')
SCAN=Path('data/experiments/ncm_paper_regime/20261027_available_pair_deep_scan_v2')
JOIN=Path('data/experiments/ncm_paper_regime/20261027_available_pair_deep_join_v2')
STAGES=[('indexed',counts.deep.BASE),('body_and_pointer',counts.supplement.ROOT),('deep',JOIN),('batch_merges',exporter.batch.ROOT)]


def main():
    if ROOT.exists():raise ValueError('fresh final reconciliation required')
    prerequisites=[(r,'finish_result.json' if n=='indexed' else 'result.json') for n,r in STAGES]+[(SCAN,'result.json')]
    for root,name in prerequisites:
        if not (root/name).exists():raise ValueError('complete source stage required')
    units=['finish-recovery-v1','discovery-v1','source-package-v1','extra-labels-v1','supplement-v1','batch-merges-v1','deep-scan-v2','deep-join-v2']
    states={u:subprocess.check_output(['systemctl','--user','show','structure-net-ncm-available-pair-'+u+'.service',
        '-p','ActiveState,SubState,MainPID,ExecMainPID,ExecMainStatus,InvocationID,NRestarts'],text=True) for u in units}
    if any('SubState=exited' not in s or 'ExecMainStatus=0' not in s for s in states.values()):raise ValueError('all actual contributing workers must have successful terminals')
    ROOT.mkdir();census.STORAGE=ROOT;db=census.open_census()
    db.executescript('CREATE TABLE pair_stages(id TEXT,stage TEXT,PRIMARY KEY(id,stage)); CREATE TABLE recovered_cache_origins(origin TEXT PRIMARY KEY);')
    stages=[]
    for stage,root in STAGES:
        db.execute('ATTACH DATABASE ? AS incoming',((root/'pairs.sqlite').resolve().as_uri()+'?mode=ro',))
        inconsistent=db.execute('''SELECT COUNT(*) FROM incoming.pairs p JOIN pairs q ON p.id=q.id
          WHERE p.input_kind!=q.input_kind OR p.input_sha!=q.input_sha OR p.target_sha!=q.target_sha
          OR p.input_tokens!=q.input_tokens OR p.target_tokens!=q.target_tokens OR p.exact_copy!=q.exact_copy''').fetchone()[0]
        if inconsistent:raise ValueError('same content identity has inconsistent metadata')
        before=db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]
        db.execute('INSERT OR IGNORE INTO pairs SELECT * FROM incoming.pairs')
        added=db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]-before
        db.execute('INSERT INTO pair_stages SELECT id,? FROM incoming.pairs',(stage,))
        for origin, in db.execute("SELECT DISTINCT origin FROM incoming.occurrences WHERE origin LIKE 'cache:%' OR origin LIKE 'batch_cache:%'"):
            if origin.startswith('batch_cache:'):origin='cache:'+origin.removeprefix('batch_cache:')
            db.execute('INSERT OR IGNORE INTO recovered_cache_origins VALUES(?)',(origin,))
        db.commit();stages.append({'stage':stage,'pairs_in_stage':db.execute('SELECT COUNT(*) FROM incoming.pairs').fetchone()[0],
            'new_pairs_after_prior_stages':added,'cumulative':counts.summarize(db)})
        db.execute('DETACH DATABASE incoming')
        print(json.dumps({'reconciled_stage':stage,'new_pairs':added}),flush=True)
    db.execute('ATTACH DATABASE ? AS original',((counts.deep.BASE/'pairs.sqlite').resolve().as_uri()+'?mode=ro',))
    unresolved=dict(db.execute('''SELECT category,COUNT(*) FROM original.cached_labels c WHERE c.pair_id IS NULL
      AND NOT EXISTS(SELECT 1 FROM recovered_cache_origins r WHERE r.origin='cache:'||c.file_sha) GROUP BY category'''))
    db.execute('DETACH DATABASE original')
    by_kind=counts.summarize(db);ordinary=[r for r in by_kind if r['kind'] in ('raw','children')]
    # Reuse the hash-verified exporter with explicit, recorded source roots.
    exporter.ROOT=ROOT;exporter.previous.deep.ROOT=JOIN
    exported=exporter.export(db)
    if exported['pairs']!=sum(r['application_length_fit_pairs'] for r in ordinary):raise ValueError('export population mismatch')
    value={'status':'full_shared_pair_inventory_reconciled_and_exported',
        'ordinary_source_summary_pairs':sum(r['distinct_pairs'] for r in ordinary),
        'ordinary_noncopy_source_summary_pairs':sum(r['noncopy_pairs'] for r in ordinary),
        'ordinary_application_length_fit_pairs':sum(r['application_length_fit_pairs'] for r in ordinary),
        'by_input_kind':by_kind,'stage_reconciliation':stages,'remaining_unjoined_cache_contents':unresolved,
        'source_results':{n:archive.read(r/('finish_result.json' if n=='indexed' else 'result.json')) for n,r in STAGES},
        'deep_scan_result':archive.read(SCAN/'result.json'),'scope':{'core_stores':500,'archive_bundles':798,'archive_files':2357622,'original_source_files':6},
        'interpretation':{'content_identity':'input kind + exact source text hash + exact target text hash; no namespace alias',
            'length_tokenizer':'cl100k_base','raw_input_tokens':[192,2048],'merge_input_token_max':2048,'target_token_max':128,
            'source_group_split_applied':False,'semantic_label_quality_reviewed':False,'test_fixture_filter_applied':False,
            'native_codec_capacity_applied':False,'training_admitted':False,'all_theoretically_possible_pairs_claimed':False,
            'raw_structural_conversion_and_image_context_pairs_excluded_from_ordinary_total':True,
            'normalized_pair_inventory_is_not_the_final_role_conditioned_training_prompt_format':True},
        'candidate_export':exported,'workers':states,'producer_sha256':archive.sha(Path(__file__)),
        'count_helper_sha256':archive.sha(Path(counts.__file__)),'export_helper_sha256':archive.sha(Path(exporter.__file__)),
        'result_bindings':{str(r/n):archive.sha(r/n) for r,n in prerequisites},
        'superseded_parser_receipt_sha256':archive.sha(ROOT.parent/'20261027_available_pair_parser_supersession_v1.json')}
    archive.write(ROOT/'result.json',value);db.close();counts.ROOT=ROOT;audit.main()
    print(json.dumps({k:value[k] for k in ('status','ordinary_source_summary_pairs','ordinary_noncopy_source_summary_pairs',
        'ordinary_application_length_fit_pairs','by_input_kind','remaining_unjoined_cache_contents')}),flush=True)


if __name__=='__main__':main()
