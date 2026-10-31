"""Hash-checked analysis of resident-GPU and CPU MoE tuning experiments."""
import argparse
from pathlib import Path
from statistics import mean

from experiments.word_games.common import digest, read, sha, write
from experiments.word_games.moe_placement import HYPOTHESIS_ID, SCHEMA_VERSION, quality_compare
from experiments.word_games.metaphor import Metaphor
from experiments.word_games.prepare_qwen38_moe import SHA256
from experiments.word_games.qwen_runtime import REQUESTS_SHA


def structural_diagnostic(reference, candidate):
    """Post-hoc decomposition; never replaces the locked full-output gate."""
    def state(key, row):
        value=row['parsed']
        if key.endswith('/receiver'):
            return {k:value.get(k) for k in ('decision','occupied_slots')}
        return {k:v for k,v in value.items() if k!='target_text'}
    receivers=[k for k in reference if k.endswith('/receiver')]
    senders=[k for k in reference if '/sender/' in k]
    try:Metaphor(**candidate['metaphor']['parsed']);valid=True
    except (TypeError,ValueError):valid=False
    return {'post_hoc':True,'live_workflows':False,'selector_valid':valid,
            'receiver_state_equal':sum(state(k,reference[k])==state(k,candidate[k]) for k in receivers),
            'receiver_count':len(receivers),
            'sender_structured_equal':sum(state(k,reference[k])==state(k,candidate[k]) for k in senders),
            'sender_count':len(senders)}


def campaign(root):
    c=read(root/'campaign_results.json');p=read(root/'producer.json')
    if c['hypothesis_id']!=HYPOTHESIS_ID or c['fingerprint']!=p['fingerprint']:
        raise ValueError('campaign identity mismatch')
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}):
        raise ValueError('producer fingerprint mismatch')
    if p['weights_sha256']!=SHA256 or p['requests_sha256']!=REQUESTS_SHA:
        raise ValueError('checkpoint/input identity mismatch')
    for path,h in c['artifacts'].items():
        if sha(path)!=h:raise ValueError('changed result '+path)
    for path,h in p['source_hashes'].items():
        if not any(sha(x)==h for x in (root/'producer_source').glob('*_'+Path(path).name)):
            raise ValueError('missing original source '+path)
    results={}
    for config in c['completed']:
        for repdir in sorted((root/config).glob('pass_*')):
            r=read(repdir/'result.json')
            for path,h in r['response_hashes'].items():
                if sha(path)!=h:raise ValueError('changed response '+path)
            responses={str(x.parent.relative_to(repdir/'runs')):read(x) for x in (repdir/'runs').rglob('response.json')}
            results[f'{root.name}/{config}/{repdir.name}']={'root':str(repdir),'config':config,'phase':c['phase'],
                    'totals':r['totals'],'result_sha256':sha(repdir/'result.json'),'responses':responses}
    return c,p,results


def select_cpu(root,out):
    c,p,rows=campaign(root)
    if c['phase']!='pilot':raise ValueError('pilot required')
    times={r['config']:r['totals']['request_seconds'] for r in rows.values() if r['config']=='control' or r['config'].startswith('cpu')}
    if 'control' not in times:raise ValueError('control unavailable')
    best=min(times,key=times.get)
    chosen=best if times[best]<.95*times['control'] else 'control'
    write(out,{'hypothesis_id':HYPOTHESIS_ID,'method':'lowest pilot elapsed, at least five percent improvement',
               'pilot_campaign':str(root/'campaign_results.json'),'pilot_sha256':sha(root/'campaign_results.json'),
               'times':times,'selected':chosen,'quality_evidence':False})
    print(chosen,times)


def aggregate(roots,out):
    out.mkdir(parents=True,exist_ok=False)
    rows={};inputs=[];failed=[]
    for root in roots:
        c,p,r=campaign(root);rows.update(r)
        inputs.append({'root':str(root),'sha256':sha(root/'campaign_results.json'),
                       'phase':c['phase'],'requested':c['requested'],'completed':c['completed'],'failed':c['failed']})
        for arm in c['failed']:
            failed.append({'root':str(root/arm),'error':read(root/arm/'interruption.json'),
                           'sha256':sha(root/arm/'interruption.json')})
    full={k:r for k,r in rows.items() if r['phase']=='full'}
    refs=[k for k,r in full.items() if r['config']=='control']
    if len(refs)<3:raise ValueError('two initial controls and ending control required')
    reference=full[refs[0]]['responses']
    fastest_control=min(full[k]['totals']['request_seconds'] for k in refs)
    paired=[]
    for k,r in full.items():
        q=quality_compare(reference,r['responses'])
        paired.append({'run':k,'config':r['config'],'root':r['root'],'totals':r['totals'],**q,
                       'structured_diagnostic':structural_diagnostic(reference,r['responses']),
                       'speedup_vs_fastest_control':fastest_control/r['totals']['request_seconds'],
                       'result_sha256':r['result_sha256']})
    by_config={}
    for name in dict.fromkeys(r['config'] for r in paired):
        rr=[r for r in paired if r['config']==name]
        ok=(name!='control' and len(rr)>=2 and all(r['quality_parity'] and
            r['totals']['request_seconds']<=.95*fastest_control for r in rr))
        by_config[name]={'repeats':len(rr),'mean_request_seconds':mean(r['totals']['request_seconds'] for r in rr),
             'mean_output_tokens_per_second':mean(r['totals']['output_tokens_per_second'] for r in rr),
             'mean_decode_tokens_per_second':mean(r['totals']['decode_only_tokens_per_second'] for r in rr),
             'min_parsed_equal':min(r['parsed_equal'] for r in rr),'speed_and_parity_gate':ok}
    winners=[name for name,r in by_config.items() if r['speed_and_parity_gate']]
    speed_only=[name for name in by_config if name!='control' and
                all(r['totals']['request_seconds']<=.95*fastest_control for r in paired if r['config']==name)]
    result={'schema':SCHEMA_VERSION,'hypothesis_id':HYPOTHESIS_ID,'confirmed':False,'conformance':'EXPLORATORY',
       'classification':'speedup_with_replay_parity' if winners else 'no_speedup_with_replay_parity',
       'sources':inputs,'failed':failed,'controls':refs,'fastest_control_seconds':fastest_control,
       'conditions':by_config,'passing_conditions':winners,'runs':paired,
       'speed_only_conditions':speed_only,'semantic_quality_status':'not independently validated; manually observed unsupported claims',
       'pilot_results':[{k:v for k,v in r.items() if k!='responses'} for r in rows.values() if r['phase']=='pilot'],
       'quality_boundary':'Exact parsed equality on a frozen 36-request replay; not task accuracy. Known parent selector/semantic failures remain.',
       'independent_judge':False,'trained_parameters':0,'analysis_source_sha256':sha(__file__)}
    abort=Path('data/experiments/word_games/20261031_moe_placement_pilot_v1/operator_note.json')
    result['pre_inference_abort']={'path':str(abort),'sha256':sha(abort),'note':read(abort)}
    supplements={}
    for root in roots:
        for name in ('producer.json','runtime.json','placement.json','telemetry.jsonl','server.log','gpu_after.json'):
            for path in root.glob('*/'+name):supplements[str(path)]=sha(path)
    result['runtime_evidence_hashes']=supplements
    reviews={}
    for file in sorted(Path('data/experiments/word_games/20261031_moe_placement_review_v1').glob('*.json')):
        record=read(file)
        run=next(r for r in paired if r['config']==file.stem and Path(r['root']).name=='pass_00')
        for case in record['cases']:
            raw=Path(run['root'])/'runs/cases'/case['case']/'receiver/response.json'
            if sha(raw)!=case['response_sha256'] or sha(raw.with_name('request.json'))!=case['prompt_sha256']:
                raise ValueError('manual review evidence changed')
        reviews[file.stem]={'path':str(file),'sha256':sha(file),'method':record['method'],
                           'independent_judge':False,'live_workflows':False,
                           'issue_cases':[r['case'] for r in record['cases'] if r['assessment']=='issue']}
    result['manual_receiver_reviews']=reviews
    write(out/'campaign_results.json',result)
    print(result['classification'],by_config)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--select-cpu',type=Path)
    a=p.parse_args()
    if a.select_cpu:select_cpu(a.select_cpu,a.output)
    else:aggregate(a.roots,a.output)
