"""Retain the eager-model result and audit a separately scoped NCM graph runtime."""
import argparse
import json
import math
from pathlib import Path
import shutil

from experiments.structure_net.run_ncm_llama_graph import BASE,PARENT,HYPOTHESIS_ID,MODEL,CODEC,MODEL_SHA,CODEC_SHA
from experiments.structure_net.ncm_llama import FIXTURES,digest,sha256,write_json
from experiments.neural_architecture_lab.finalize_ncm_llama import audit,summarize
from neural_architecture_lab.ncm_llama_hypothesis import graph_classification,NcmGraphEvidence


def read(p):return json.loads(Path(p).read_text())


def verify(root):
    p=read(root/'producer.json');r=read(root/'result.json')
    if p['hypothesis_id']!=HYPOTHESIS_ID or digest({k:v for k,v in p.items() if k!='fingerprint'})!=p['fingerprint'] or r['producer_fingerprint']!=p['fingerprint']:
        raise ValueError('graph producer identity differs')
    copies={sha256(x) for x in (root/'producer_source').iterdir()}
    if any(h not in copies for h in p['source_hashes'].values()):raise ValueError('graph source missing')
    for path,h in p['input_hashes'].items():
        if sha256(path)!=h:raise ValueError('graph reference changed')
    return p,r


def finalize(args):
    if args.output.exists():raise ValueError('fresh graph aggregate required')
    parent=read(args.parent);pilot=BASE/'20261021_pilot_v3';root=BASE/'20261021_heldout_v1'
    _,pr=verify(pilot);p,r=verify(root)
    if pr['status']!='passed' or r['status']!='completed_pending_audit':raise ValueError('complete graph controls required')
    if sha256(MODEL)!=MODEL_SHA or sha256(CODEC)!=CODEC_SHA:raise ValueError('weights changed')
    raw=read(root/'rows.json');rows=raw['rows'];ex={e['id']:e for e in read(FIXTURES)['examples'] if e['split']=='heldout'}
    keys=[(x['id'],x['repeat']) for x in rows]
    if len(keys)!=300 or len(set(keys))!=300 or set(keys)!={(i,n) for i in ex for n in range(3)} or raw['status']!='completed' or sha256(root/'rows.json')!=r['rows_sha256']:
        raise ValueError('graph timing cells differ')
    if sha256(root/'excluded_warmups.json')!=r['warmups_sha256'] or [x['id'] for x in read(root/'excluded_warmups.json')['rows']]!=list(ex):raise ValueError('graph warmups differ')
    controls={};summary={};reference=None
    for condition in ('selected','teacher','fp16'):
        old,_,terminal,_=audit(PARENT/f'20261021_{condition}_heldout_v1',condition)
        controls[condition]=terminal;summary[condition]=summarize(old)
        if condition=='selected':reference={(x['id'],x['repeat']):x for x in old}
    parity=0
    for x in rows:
        old=reference[x['id'],x['repeat']]
        if x['source_sha256']!=old['source_sha256'] or x['condition']!='graph':raise ValueError('graph source differs')
        if not math.isfinite(x['wall_seconds']) or x['wall_seconds']<=0:raise ValueError('invalid graph request timing')
        parity+=x['native_output_ids']==old['native_output_ids'] and x['text']==old['text'] and x['common_output_tokens']==old['common_output_tokens']
    if parity!=r['parity_cells'] or r['unique_runtime_weight_buffer_bytes']!=controls['selected']['unique_runtime_weight_buffer_bytes']:
        raise ValueError('graph output/model footprint readback differs')
    summary['graph']=summarize(rows);g=summary['graph'];ratios={}
    for a in ('selected','teacher','fp16'):
        ratios[a]={'latency_speedup':summary[a]['sum_median_seconds']/g['sum_median_seconds'],
            'token_rate_ratio':g['common_tokens_per_second']/summary[a]['common_tokens_per_second']}
    passes=parent['source_pass_counts'];classification=graph_classification(parity,passes['teacher'],passes['selected'],ratios['teacher']['latency_speedup'],ratios['teacher']['token_rate_ratio'])
    result={'schema':'nal.ncm-graph-campaign.v1','hypothesis_id':HYPOTHESIS_ID,'status':'completed_exploratory',
        'classification':classification,'parent_classification':parent['classification'],'parent_campaign':str(args.parent),
        'parent_campaign_sha256':sha256(args.parent),'model_sha256':MODEL_SHA,'codec_sha256':CODEC_SHA,
        'parity_cells':parity,'fresh_graph_timing_cells':300,'reused_control_timing_cells':900,'training_updates':0,
        'source_pass_counts_inherited_by_exact_output_parity':passes,'summary':summary,'ratios':ratios,'memory':r,'pilot':pr,
        'used_runtime_validation_sources':True,'independent_new_model_holdout':False,'underpowered':True,'confirmed':False,
        'scientific_boundary':'one frozen model; exact-output runtime validation on used sources; nearby sequential reused timing controls, no randomized block order; no production/mobile/full-context claim'}
    failed=BASE/'20261021_pilot_v1'
    paths=[args.parent,MODEL,CODEC,failed/'producer.json',failed/'failure.json',failed/'error.log',pilot/'producer.json',pilot/'result.json',pilot/'checks.json',root/'producer.json',root/'result.json',root/'rows.json',root/'excluded_warmups.json']
    failed2=BASE/'20261021_pilot_v2';paths += [failed2/'producer.json',failed2/'failure.json',failed2/'error.log']
    result['failed_systems_pilots']=2
    result['artifacts']={str(x):sha256(x) for x in paths}
    copies=args.output.parent/'analysis_source';copies.mkdir(parents=True)
    for path in (Path(__file__),Path('src/neural_architecture_lab/ncm_llama_hypothesis.py')):shutil.copyfile(path,copies/path.name)
    write_json(args.output,result);print(json.dumps({'classification':classification,'parity_cells':parity,'ratios':ratios}))


def register(args):
    c=read(args.campaign)
    for path,h in c['artifacts'].items():
        if sha256(path)!=h:raise ValueError('graph evidence changed')
    record=NcmGraphEvidence(c['classification'],str(args.campaign),sha256(args.campaign),str(args.report),sha256(args.report),c['parent_campaign'],c['parent_campaign_sha256']).record()
    from structure_net.logging.standardized_logging import LoggingConfig,StandardizedLogger
    logger=StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',queue_dir='data/experiment_queue',sent_dir='data/experiment_sent',rejected_dir='data/experiment_rejected',enable_wandb=False,auto_upload=False,enable_chromadb=True,chromadb_path='data/chroma_db'))
    logger.log_hypothesis(record);exact=logger.hypotheses_collection.get(ids=[HYPOTHESIS_ID],include=['metadatas'])
    if exact['ids']!=[HYPOTHESIS_ID] or any(exact['metadatas'][0][k]!=record[k] for k in ('tested','confirmed','confirmation_status')):raise ValueError('exact graph registry readback differs')
    semantic=logger.hypotheses_collection.query(query_texts=[record['description']],n_results=10)
    if HYPOTHESIS_ID not in semantic['ids'][0]:raise ValueError('semantic graph readback differs')
    receipt={'status':'registered_and_read_back','record':record,'exact_readback':exact['metadatas'][0],'semantic_readback_ids':semantic['ids'][0]}
    for path in (args.campaign.parent/'registry_readback.json',Path('data/hypothesis_registry')/(HYPOTHESIS_ID+'.json')):
        if path.exists():raise ValueError('fresh graph registry receipt required')
        write_json(path,receipt)
        if read(path)['record']!=record:raise ValueError('full graph JSON readback differs')
    print(json.dumps(record))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('finalize');a.add_argument('--parent',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=s.add_parser('register');a.add_argument('--campaign',type=Path,required=True);a.add_argument('--report',type=Path,required=True)
    args=p.parse_args();globals()[args.command](args)


if __name__=='__main__':main()
