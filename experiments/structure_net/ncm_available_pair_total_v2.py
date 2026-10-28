"""Extend the complete inventory with the separately identified merge-job format."""
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

from experiments.structure_net import ncm_available_pair_total as previous
from experiments.structure_net import ncm_available_pair_batch_merges as batch
from experiments.structure_net import ncm_available_pair_inventory_audit as audit

ROOT=Path('data/experiments/ncm_paper_regime/20261027_available_pair_total_v2')
archive=previous.archive
census=previous.census


def export(db):
    roots=[('indexed',previous.deep.BASE),('body_and_pointer',previous.supplement.ROOT),
           ('deep',previous.deep.ROOT),('batch_merges',batch.ROOT)]
    readers=[(name,census.readonly(root/'pairs.sqlite')) for name,root in roots]
    path=ROOT/'candidate_pairs_unsplit.jsonl';digest=hashlib.sha256();count=0
    with path.open('xb') as out:
        for ident,kind,sh,th,st,tt in db.execute("SELECT id,input_kind,input_sha,target_sha,input_tokens,target_tokens FROM pairs WHERE input_kind IN ('raw','children') AND "+previous.FIT+' ORDER BY id'):
            for name,reader in readers:
                row=reader.execute('SELECT s.text,t.text FROM pairs p JOIN texts s ON s.sha=p.input_sha JOIN texts t ON t.sha=p.target_sha WHERE p.id=?',(ident,)).fetchone()
                if row is not None:break
            else:raise ValueError('pair text unavailable')
            source,target=row
            if census.digest(source)!=sh or census.digest(target)!=th:raise ValueError('pair content changed')
            origin=reader.execute('SELECT origin,record,kind FROM occurrences WHERE pair_id=? LIMIT 1',(ident,)).fetchone()
            value={'pair_id':ident,'input_kind':kind,'source':source,'target':target,'input_tokens':st,'target_tokens':tt,
                'representative_stage':name,'representative_origin':origin,'split':'unassigned','training_admitted':False}
            line=(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n').encode();out.write(line);digest.update(line);count+=1
    for _,reader in readers:reader.close()
    result={'path':str(path),'pairs':count,'sha256':digest.hexdigest(),'bytes':path.stat().st_size,
        'source_group_split_and_quality_review_required':True,'role_date_and_spine_conditioning_retained_in_source_artifacts':True,
        'export_is_normalized_pair_inventory_not_final_training_prompt_format':True,'training_admitted':False}
    archive.write(ROOT/'candidate_export.json',result);return result


def main():
    if ROOT.exists():raise ValueError('fresh final inventory extension required')
    old_root=previous.ROOT;old=archive.read(old_root/'result.json');br=archive.read(batch.ROOT/'result.json')
    states={u:subprocess.check_output(['systemctl','--user','show','structure-net-ncm-available-pair-'+u+'-v1.service',
        '-p','ActiveState,SubState,MainPID,ExecMainPID,ExecMainStatus,InvocationID,NRestarts'],text=True) for u in ('total','batch-merges')}
    if any('SubState=exited' not in s or 'ExecMainStatus=0' not in s for s in states.values()):raise ValueError('successful terminal workers required')
    ROOT.mkdir();source=census.readonly(old_root/'pairs.sqlite');db=sqlite3.connect(ROOT/'pairs.sqlite');source.backup(db);source.close()
    db.execute('ATTACH DATABASE ? AS incoming',((batch.ROOT/'pairs.sqlite').resolve().as_uri()+'?mode=ro',))
    bad=db.execute('''SELECT COUNT(*) FROM incoming.pairs p JOIN pairs q ON p.id=q.id
      WHERE p.input_sha!=q.input_sha OR p.target_sha!=q.target_sha OR p.input_kind!=q.input_kind
      OR p.input_tokens!=q.input_tokens OR p.target_tokens!=q.target_tokens OR p.exact_copy!=q.exact_copy''').fetchone()[0]
    if bad:raise ValueError('batch pair conflicts with prior content identity')
    before=db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0];db.execute('INSERT OR IGNORE INTO pairs SELECT * FROM incoming.pairs')
    added=db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]-before
    db.execute("INSERT INTO pair_stages SELECT id,'batch_merges' FROM incoming.pairs")
    for origin, in db.execute("SELECT DISTINCT origin FROM incoming.occurrences WHERE origin LIKE 'batch_cache:%'"):
        db.execute('INSERT OR IGNORE INTO recovered_cache_origins VALUES(?)',('cache:'+origin.removeprefix('batch_cache:'),))
    db.commit();db.execute('DETACH DATABASE incoming')
    db.execute('ATTACH DATABASE ? AS original',((previous.deep.BASE/'pairs.sqlite').resolve().as_uri()+'?mode=ro',))
    unresolved=dict(db.execute('''SELECT category,COUNT(*) FROM original.cached_labels c WHERE c.pair_id IS NULL
      AND NOT EXISTS(SELECT 1 FROM recovered_cache_origins r WHERE r.origin='cache:'||c.file_sha) GROUP BY category'''))
    db.execute('DETACH DATABASE original');by_kind=previous.summarize(db);ordinary=[r for r in by_kind if r['kind'] in ('raw','children')]
    exported=export(db)
    if exported['pairs']!=sum(r['application_length_fit_pairs'] for r in ordinary):raise ValueError('export differs from reconciled count')
    result={**old,'status':'all_completed_pair_census_formats_reconciled',
        'ordinary_source_summary_pairs':sum(r['distinct_pairs'] for r in ordinary),
        'ordinary_noncopy_source_summary_pairs':sum(r['noncopy_pairs'] for r in ordinary),
        'ordinary_application_length_fit_pairs':sum(r['application_length_fit_pairs'] for r in ordinary),
        'by_input_kind':by_kind,'candidate_export':exported,'remaining_unjoined_cache_contents':unresolved,
        'stage_reconciliation':old['stage_reconciliation']+[{'stage':'batch_merges','pairs_in_stage':br['distinct_pairs'],
            'new_pairs_after_prior_stages':added,'cumulative':by_kind}],
        'batch_merge_result':br,'workers':{**old['workers'],**states},'producer_sha256':archive.sha(Path(__file__)),
        'result_bindings':{**old['result_bindings'],str(old_root/'result.json'):archive.sha(old_root/'result.json'),
                           str(batch.ROOT/'result.json'):archive.sha(batch.ROOT/'result.json')}}
    archive.write(ROOT/'result.json',result);db.close()
    previous.ROOT=ROOT;audit.main()
    print(json.dumps({k:result[k] for k in ('status','ordinary_source_summary_pairs','ordinary_noncopy_source_summary_pairs',
        'ordinary_application_length_fit_pairs','by_input_kind','remaining_unjoined_cache_contents')}),flush=True)


if __name__=='__main__':main()
