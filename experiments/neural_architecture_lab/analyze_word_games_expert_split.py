"""Verify immutable replay evidence and summarize both expert partition studies."""
import argparse
from pathlib import Path
from experiments.word_games.common import read,sha,write
from experiments.word_games.native_speed import output_tokens
from experiments.neural_architecture_lab.analyze_word_games_moe_placement import structural_diagnostic


def verify_root(root):
    p=read(root/'producer.json')
    archive={sha(f) for f in (root/'producer_source').iterdir() if f.is_file()}
    if not set(p['source_hashes'].values())<=archive:raise ValueError('missing archived producer')
    for result in root.rglob('result.json'):
        obj=read(result)
        for path,h in obj.get('response_hashes',{}).items():
            if sha(path)!=h:raise ValueError('response changed: '+path)
    return {'root':str(root),'producer_sha256':sha(root/'producer.json'),
            'campaign_sha256':sha(root/'campaign_results.json')}


def rows(path):
    result={}
    for p in path.rglob('response.json'):
        obj=read(p)
        # Native artifacts retain both the HTTP response and parsed object.
        if not isinstance(obj.get('parsed'),dict):raise ValueError('invalid parsed response')
        result[str(p.parent.relative_to(path))]=obj
    return result


def memory_peaks(root):
    result={}
    import json
    for p in root.rglob('telemetry.jsonl'):
        values={}
        for line in p.read_text().splitlines():
            for row in json.loads(line)['gpus']:
                v=values.setdefault(row[0],{'peak_used_mib':0,'min_free_mib':99999})
                v['peak_used_mib']=max(v['peak_used_mib'],int(row[1]))
                v['min_free_mib']=min(v['min_free_mib'],int(row[2]))
        result[str(p.parent.relative_to(root))] = values
    return result


def analyze(hybrid,three,out,packed=None):
    out.mkdir(parents=True,exist_ok=False)
    sources=[verify_root(hybrid),verify_root(three)]
    h=read(hybrid/'campaign_results.json'); t=read(three/'campaign_results.json')
    ha={r['arm']:r for r in h['results']}
    baseline=ha['experts8']['passes'][0]['totals'];candidate=ha['pins64']['passes'][0]['totals']
    hresult={'hypothesis_id':'word-games-hybrid-experts-v1','classification':'negative_speed_pilot',
             'baseline':baseline,'candidate':candidate,'completed_requests':sum(p['totals']['calls'] for r in ha.values() for p in r['passes']),
             'full_passes':0,'stats':ha['pins64']['stats'],'throughput_ratio':candidate['output_tokens_per_second']/baseline['output_tokens_per_second'],
             'wall_time_ratio':candidate['request_seconds']/baseline['request_seconds'],'service_restored':True,
             'description':'Implemented within-layer expert partition with compact GPU pools, exact selected-expert accounting, unchanged quantized weights, persistent CPU miss graphs and asynchronous worker. 64 pins in each of 26 CPU-resident Qwen35MoE layers. Failed competitive speed pilot versus whole-bank eight-layer SUPER placement. No full quality or accuracy-parity claim.'}
    full=[r for r in t['results'] if r['phase']=='full' and r['completed']]
    tres={'hypothesis_id':'word-games-three-gpu-experts-v1','full_passes':len(full),
          'completed_requests':sum(r['totals']['calls'] for r in t['results'] if r['completed']),
          'service_restored':t['service_restored']}
    if len(full)==2:
        b=next(r for r in full if r['extra_banks']==0);c=next(r for r in full if r['extra_banks']>0)
        tres.update(baseline=b['totals'],candidate=c['totals'],extra_banks=c['extra_banks'])
        tres['classification']='full_speedup_quality_unestablished' if c['totals']['output_tokens_per_second']>b['totals']['output_tokens_per_second'] else 'full_no_speedup_quality_unestablished'
        reference=rows(three/'full_experts8/runs');changed=rows(three/f'full_experts{8+c["extra_banks"]}/runs')
        if len(reference)!=36 or set(reference)!=set(changed):raise ValueError('36 paired rows required')
        tres['quality_diagnostic']=structural_diagnostic(reference,changed)
        tres['parsed_equal']=sum(reference[k]['parsed']==changed[k]['parsed'] for k in reference)
        tres['changed_ids']=[k for k in reference if reference[k]['parsed']!=changed[k]['parsed']]
    else:
        pilot=[r for r in t['results'] if r['phase']=='pilot' and r['completed']]
        b=next(r for r in pilot if r['extra_banks']==0);c=next(r for r in pilot if r['extra_banks']>0)
        tres.update(baseline=b['totals'],candidate=c['totals'],classification='negative_speed_pilot')
    tres['throughput_ratio']=tres['candidate']['output_tokens_per_second']/tres['baseline']['output_tokens_per_second']
    tres['wall_time_ratio']=tres['candidate']['request_seconds']/tres['baseline']['request_seconds']
    tres['description']='Temporarily unloaded Triton Qwen3-8B at user request, kept attention/recurrent work on RTX 3060 and moved intact expert banks onto RTX 2060 SUPER and RTX 2060. Same Q4_K_M weights, Q8 KV and b11960 runtime. Matched fixed WORD_GAMES replay; state diagnostics and speed are separate from semantic accuracy. Qwen3-8B restored and configuration verified.'
    roots=[hybrid,three]
    if packed:
        sources.append(verify_root(packed));roots.append(packed)
        p=read(packed/'campaign_results.json')
        tres['service_restored']=tres['service_restored'] and p['service_restored']
        tres['completed_requests']+=sum(r['totals']['calls'] for r in p['results'] if r['completed'])
        pc=[r for r in p['results'] if r['phase']=='full' and r['completed']]
        tres['placements']={'experts8':tres['baseline'],'experts18':tres['candidate']}
        tres['packed_followup_completed']=bool(pc)
        if pc:
            start=next(r for r in p['results'] if r['condition']=='reference18')
            end=next(r for r in p['results'] if r['condition']=='ending_reference18')
            if not start['completed'] or not end['completed']:raise ValueError('missing drift controls')
            drift=end['totals']['request_seconds']/start['totals']['request_seconds']
            tres['reference_pilot_drift_ratio']=drift
            tres['placements']['packed38']=pc[0]['totals'];tres['full_passes']+=1
            tres['packed_vs_experts18_throughput_ratio']=pc[0]['totals']['output_tokens_per_second']/tres['candidate']['output_tokens_per_second']
            if abs(drift-1)>.05:raise ValueError('pilot drift exceeds follow-up gate')
            if pc[0]['totals']['output_tokens_per_second']>tres['candidate']['output_tokens_per_second']:
                tres['candidate']=pc[0]['totals'];tres['best_placement']='packed38'
                reference=rows(three/'full_experts8/runs');changed=rows(packed/'full_packed38/runs')
                if len(reference)!=36 or set(reference)!=set(changed):raise ValueError('packed paired corpus mismatch')
                tres['quality_diagnostic']=structural_diagnostic(reference,changed)
                tres['parsed_equal']=sum(reference[k]['parsed']==changed[k]['parsed'] for k in reference)
                tres['changed_ids']=[k for k in reference if reference[k]['parsed']!=changed[k]['parsed']]
                tres['throughput_ratio']=tres['candidate']['output_tokens_per_second']/tres['baseline']['output_tokens_per_second']
                tres['wall_time_ratio']=tres['candidate']['request_seconds']/tres['baseline']['request_seconds']
                tres['description']+=' Follow-up used spare VRAM to put 38 of 40 serving expert banks on GPUs, retaining two CPU banks. Full comparison uses preceding controls plus bracketing three-request drift checks; not independent replication.'
    evidence={str(p):sha(p) for root in roots for p in root.rglob('*')
              if p.is_file() and p.name in ('result.json','server.log','telemetry.jsonl','hybrid_stats.json','service_restored.json')}
    failure=three.parent/'20261031_three_gpu_experts_v1'
    attempts=[]
    if (failure/'campaign_results.json').exists() and failure!=three:
        attempts.append({**verify_root(failure),'reason':'GPU release lag; guard stopped before request timing',
                         'service_restored':read(failure/'campaign_results.json')['service_restored']})
    reviews={}
    review_root=three.parent/'20261031_expert_split_semantics_v1'
    for p in review_root.glob('*.json'):
        review=read(p)
        for case in review['cases']:
            for key in ('request','response'):
                if sha(case[key])!=case[key+'_sha256']:raise ValueError('semantic-review evidence changed')
        reviews[p.stem]={'path':str(p),'sha256':sha(p),'method':review['method'],
                        'issue_cases':[r['case'] for r in review['cases'] if r['assessment']=='issue']}
    write(out/'campaign_results.json',{'sources':sources,'hypotheses':[hresult,tres],'evidence_hashes':evidence,
          'infrastructure_attempts':attempts,
          'manual_receiver_reviews':reviews,
          'memory_peaks':{str(root.name):memory_peaks(root) for root in roots},'confirmed':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--hybrid',type=Path,required=True);p.add_argument('--three',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--packed',type=Path)
    a=p.parse_args();analyze(a.hybrid,a.three,a.output,a.packed)
