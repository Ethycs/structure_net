"""Audit new NCM optimization evidence without changing completed parents."""
import argparse
import json
from pathlib import Path
import shutil

from experiments.structure_net.ncm_llama import FIXTURES,digest,sha256,tokenizers,decode_ids,write_json
from experiments.structure_net.run_ncm_replicas import BASE as REPLICAS,ORDER,REFERENCE,TEACHER,summarize as summarize_replicas
from experiments.structure_net.ncm_factorization import BASE as FACTORS
from experiments.neural_architecture_lab.finalize_ncm_llama import summarize
from neural_architecture_lab.ncm_optimization_hypotheses import (
    OptimizationEvidence,replica_classification,factor_classification)
from experiments.structure_net.ncm_vocabulary import CachedVocabulary


def read(p):return json.loads(Path(p).read_text())


def verify(root):
    p=read(root/'producer.json');r=read(root/'result.json')
    if digest({k:v for k,v in p.items() if k!='fingerprint'})!=p['fingerprint'] or r['producer_fingerprint']!=p['fingerprint']:
        raise ValueError('producer fingerprint differs')
    copies={sha256(x) for x in (root/'producer_source').iterdir()}
    if any(h not in copies for h in p['source_hashes'].values()):raise ValueError('source snapshot missing')
    for path,h in p['input_hashes'].items():
        if sha256(path)!=h:raise ValueError('producer input changed')
    return p,r


def finish(args,result,paths):
    if args.output.exists():raise ValueError('fresh aggregate required')
    paths=list(dict.fromkeys(paths))
    result.update(artifacts={str(p):sha256(p) for p in paths},confirmed=False,underpowered=True,
        status='completed_exploratory',failed_cells=0,retried_cells=0,
        scientific_boundary='One frozen task-specific model; technical repeats on used synthetic sources; no production/mobile/general capability claim.')
    dst=args.output.parent/'analysis_source';dst.mkdir(parents=True)
    for path in (Path(__file__),Path('src/neural_architecture_lab/ncm_optimization_hypotheses.py'),Path('experiments/structure_net/ncm_vocabulary.py')):shutil.copyfile(path,dst/path.name)
    write_json(args.output,result);print(json.dumps({k:result[k] for k in ('hypothesis_id','classification','direct_timing_cells')}))


def replicas(args):
    pilot=REPLICAS/'20261021_pilot_v1';_,pr=verify(pilot)
    if pr['status']!='passed' or pr['parity_checks']!=27:raise ValueError('nine-copy pilot differs')
    control=read(TEACHER)
    parent=Path('data/experiments/ncm_llama/20261021_registered_v1/campaign_results.json')
    teacher_rate=read(parent)['heldout_summary']['teacher']['common_tokens_per_second']
    reference={(r['id'],r['repeat']):r for r in read(REFERENCE)['rows']}
    examples={e['id']:e for e in read(FIXTURES)['examples'] if e['split']=='heldout'}
    gpt,llama=tokenizers();gpt=CachedVocabulary(gpt);results=[];paths=[TEACHER,REFERENCE,FIXTURES,parent]
    execution=REPLICAS/'20261021_campaign_v1/execution.json';jobs=read(execution)
    if jobs['status']!='completed' or [r['copies'] for r in jobs['runs']]!=list(ORDER):raise ValueError('coordinator schedule differs')
    for job in jobs['runs']:
        n=job['copies'];root=Path(job['root']);p,r=verify(root)
        if job['returncode'] or job['result_sha256']!=sha256(root/'result.json') or r['status']!='completed':raise ValueError('child terminal differs')
        raw=read(root/'rows.json');rows=raw['rows'];batches=raw['batches']
        if sha256(root/'rows.json')!=r['rows_sha256']:raise ValueError('raw rows changed')
        if {(x['id'],x['repeat']) for x in rows}!=set(reference):raise ValueError('request schedule differs')
        for x in rows:
            old=reference[x['id'],x['repeat']]
            if not x['parity'] or x['native_output_ids']!=old['native_output_ids'] or x['text']!=old['text'] or x['source_sha256']!=examples[x['id']]['source_sha256']:
                raise ValueError('literal replica output parity failed')
            if x['common_output_tokens']!=len(llama.encode(x['text'],add_special_tokens=False)) or decode_ids(x['native_output_ids'],gpt)!=(x['text'],x['termination']):
                raise ValueError('token decoding/count differs')
        summary=summarize_replicas(rows,batches,n)
        if summary!=r['summary'] or r['unique_runtime_weight_buffer_bytes']!=n*713294288:raise ValueError('summary/storage differs')
        summary.update(model_bytes=r['unique_runtime_weight_buffer_bytes'],peak_allocated_bytes=r['peak_allocated_bytes'],peak_reserved_bytes=r['peak_reserved_bytes'])
        summary.update(within_teacher_model_budget=summary['model_bytes']<=control['unique_runtime_weight_buffer_bytes'],
            within_teacher_allocated_budget=summary['peak_allocated_bytes']<=control['peak_allocated_bytes'],
            within_teacher_reserved_budget=summary['peak_reserved_bytes']<=control['peak_reserved_bytes'])
        results.append(summary)
        paths+=list(root.glob('*.json'))
    eligible=[r['aggregate_tokens_per_second'] for r in results if r['within_teacher_reserved_budget']]
    parity=sum(r['parity_cells'] for r in results)
    result={'schema':'nal.ncm-replicas-campaign.v1','hypothesis_id':'llama32-3b-ncm-replicas-v1',
        'classification':replica_classification(parity,eligible,teacher_rate),'direct_timing_cells':1800,
        'fresh_conditions':6,'parity_cells':parity,'summary':results,'teacher_reused':control,'teacher_reused_tokens_per_second':teacher_rate,
        'excluded_warmup_requests':sum(ORDER),'failed_systems_pilots':0,
        'comparison_boundary':'Independent copies on one GPU; no shared-weight batching or multi-GPU test. Llama uses reused single-request controls, not an optimal equal-memory serving comparison.'}
    finish(args,result,paths+list(pilot.glob('*.json'))+[execution,execution.parent/'producer.json',execution.parent/'producer.py'])


def review_count(rows,examples,path):
    review=read(path);items=review['items'];indexed={r['id']:r for r in items}
    if len(items)!=len(examples) or set(indexed)!=set(examples):raise ValueError('complete unique source reviews required')
    passes=0
    for x in rows:
        item=indexed[x['id']];ex=examples[x['id']]
        if item['source_sha256']!=ex['source_sha256'] or item['text_sha256']!=digest(x['text']):raise ValueError('review binding differs')
        if item['source_quote'] not in ex['source'] or not item['rationale']:raise ValueError('source support missing')
    for x in items:passes+=x['passed']
    return passes


def batching(args):
    base=Path('data/experiments/ncm_batching');pilot=base/'20261021_pilot_v1';_,pr=verify(pilot)
    if pr['status']!='passed' or pr['parity_checks']!=27:raise ValueError('batch9 pilot differs')
    reference={(r['id'],r['repeat']):r for r in read(REFERENCE)['rows']}
    examples={e['id']:e for e in read(FIXTURES)['examples'] if e['split']=='heldout'}
    gpt,llama=tokenizers();gpt=CachedVocabulary(gpt);results=[];paths=[REFERENCE,FIXTURES]+list(pilot.glob('*.json'))
    for size in (1,4,9):
        root=base/f'20261021_batch{size}_v1';p,r=verify(root)
        if r['status']!='completed' or r['batch']!=size:raise ValueError('completed batch condition required')
        raw=read(root/'rows.json');rows=raw['rows'];batches=raw['batches']
        if sha256(root/'rows.json')!=r['rows_sha256'] or {(x['id'],x['repeat']) for x in rows}!=set(reference):raise ValueError('batch cell schedule differs')
        for x in rows:
            old=reference[x['id'],x['repeat']]
            if x['source_sha256']!=examples[x['id']]['source_sha256'] or x['native_output_ids']!=old['native_output_ids'] or x['text']!=old['text']:
                raise ValueError('batch/parent output parity differs')
            if x['common_output_tokens']!=len(llama.encode(x['text'],add_special_tokens=False)) or decode_ids(x['native_output_ids'],gpt)!=(x['text'],x['termination']):raise ValueError('batch output counts differ')
        summary=summarize_replicas(rows,batches,size)
        if summary!=r['summary'] or r['model_bytes']!=713294288 or r['peak_reserved_bytes']>6794772480:raise ValueError('batch memory/summary differs')
        summary.update(model_bytes=r['model_bytes'],peak_allocated_bytes=r['peak_allocated_bytes'],peak_reserved_bytes=r['peak_reserved_bytes'],
            timed_dummy_outputs=r['timed_dummy_outputs'],graph_setup_seconds=r['graph_setup_seconds'])
        results.append(summary);paths+=list(root.glob('*.json'))
    ratio=max(x['aggregate_tokens_per_second'] for x in results)/results[0]['aggregate_tokens_per_second']
    result={'schema':'nal.ncm-batching-campaign.v1','hypothesis_id':'llama32-3b-ncm-batching-v1',
        'classification':'batching_throughput_target_met' if ratio>=2 else 'batching_throughput_target_unmet',
        'direct_timing_cells':900,'parity_cells':900,'summary':results,'maximum_token_rate_ratio_vs_fresh_batch1':ratio,
        'excluded_warmup_requests':14,'failed_systems_pilots':0,
        'comparison_boundary':'Shared-weight batches on one GPU; exact-output runtime validation on used synthetic sources. Llama batching untested.'}
    finish(args,result,paths)


def factorization(args):
    root=FACTORS;pilot=root/'20261021_pilot_v1';training=root/'20261021_training_v1';dev=root/'20261021_development_v1'
    _,pr=verify(pilot);_,tr=verify(training);_,dr=verify(dev)
    if pr['status']!='passed' or tr['status']!='trained':raise ValueError('training systems gate differs')
    decision=read(dev/'source_review.json');selected=decision['selected'];paths=[]
    for r in (pilot,training,dev):paths+=list(r.glob('*.json'))
    result={'schema':'nal.ncm-factorization-campaign.v1','hypothesis_id':'llama32-3b-ncm-factorization-v1',
        'development':dr,'development_source_review':decision,'training':tr,'pilot':pr,
        'direct_timing_cells':0,'classification':'factorized_development_quality_failed','final_stage_executed':False}
    if selected is not None:
        examples={e['id']:e for e in read(FIXTURES)['examples'] if e['split']=='heldout'}
        gpt,llama=tokenizers();gpt=CachedVocabulary(gpt);summary={};memory={};passes={}
        for condition in (selected,'dense'):
            rroot=root/f'20261021_{condition}_heldout_v1';_,r=verify(rroot);rows=read(rroot/'rows.json')['rows']
            if r['status']!='completed_pending_source_review' or sha256(rroot/'rows.json')!=r['rows_sha256']:
                raise ValueError('timing rows/terminal differ')
            keys=[(x['id'],x['repeat']) for x in rows]
            if len(keys)!=300 or set(keys)!={(i,n) for i in examples for n in range(3)}:raise ValueError('complete final timing cells required')
            for x in rows:
                if x['source_sha256']!=examples[x['id']]['source_sha256'] or x['common_output_tokens']!=len(llama.encode(x['text'],add_special_tokens=False)) or decode_ids(x['native_output_ids'],gpt)!=(x['text'],x['termination']):
                    raise ValueError('source/decoded output differs')
            passes[condition]=review_count(rows,examples,rroot/'source_review.json')
            summary[condition]=summarize(rows);memory[condition]=r
            paths+=list(rroot.glob('*.json'))
        s=summary[selected];d=summary['dense']
        byte_ratio=memory[selected]['unique_runtime_weight_buffer_bytes']/memory['dense']['unique_runtime_weight_buffer_bytes']
        latency_ratio=d['sum_median_seconds']/s['sum_median_seconds'];tps_ratio=s['common_tokens_per_second']/d['common_tokens_per_second']
        result.update(classification=factor_classification(passes[selected],passes['dense'],byte_ratio,latency_ratio,tps_ratio),
            direct_timing_cells=600,final_stage_executed=True,summary=summary,memory=memory,source_passes=passes,
            model_byte_ratio=byte_ratio,latency_speedup=latency_ratio,token_rate_ratio=tps_ratio,
            teacher_reused_token_rate_ratio=s['common_tokens_per_second']/77.2062647316689)
    else:
        diagnostic=root/'20261021_diagnostic_v1';_,r=verify(diagnostic)
        result.update(diagnostic=r,direct_timing_cells=r['requests'])
        paths+=list(diagnostic.glob('*.json'))
    paths+=[training/'initial.pt',training/'repaired.pt',FIXTURES]
    finish(args,result,paths)


def register(args):
    c=read(args.campaign)
    for path,h in c['artifacts'].items():
        if sha256(path)!=h:raise ValueError('aggregate artifact changed')
    record=OptimizationEvidence(c['hypothesis_id'],c['classification'],str(args.campaign),sha256(args.campaign),
        str(args.report),sha256(args.report),c['direct_timing_cells']).record()
    from structure_net.logging.standardized_logging import LoggingConfig,StandardizedLogger
    logger=StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',queue_dir='data/experiment_queue',sent_dir='data/experiment_sent',rejected_dir='data/experiment_rejected',enable_wandb=False,auto_upload=False,enable_chromadb=True,chromadb_path='data/chroma_db'))
    logger.log_hypothesis(record);hid=record['id'];exact=logger.hypotheses_collection.get(ids=[hid],include=['metadatas'])
    if exact['ids']!=[hid] or any(exact['metadatas'][0][k]!=record[k] for k in ('tested','confirmed','confirmation_status','direct_experiment_count')):
        raise ValueError('exact registry readback differs')
    semantic=logger.hypotheses_collection.query(query_texts=[record['description']],n_results=10)
    if hid not in semantic['ids'][0]:raise ValueError('semantic registry readback failed')
    receipt={'status':'registered_and_read_back','record':record,'exact_readback':exact['metadatas'][0],'semantic_ids':semantic['ids'][0]}
    for path in (args.campaign.parent/'registry_readback.json',Path('data/hypothesis_registry')/(hid+'.json')):
        if path.exists():raise ValueError('fresh registry receipt required')
        write_json(path,receipt)
        if read(path)['record']!=record:raise ValueError('full record readback differs')
    print(json.dumps(record))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    for name in ('replicas','factorization','batching'):
        a=s.add_parser(name);a.add_argument('--output',type=Path,required=True)
    a=s.add_parser('register');a.add_argument('--campaign',type=Path,required=True);a.add_argument('--report',type=Path,required=True)
    args=p.parse_args();globals()[args.command](args)


if __name__=='__main__':main()
