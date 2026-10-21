"""Independent grading with exact source turns, retaining interrupted free-quote evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import re
import shutil
import time

from experiments.structure_net import ncm_quality as q
from experiments.neural_architecture_lab.finalize_ncm_quality import verify,summarize,paired_ci,agreement
from neural_architecture_lab.ncm_quality_hypothesis import classification

BASE=q.BASE.parent/'20261021_judge_v3'
PLAN=Path('docs/07 - Status Reports/2026-10-21_ncm-independent-quality-amendment-v3.md')
RUBRIC='''You independently grade two anonymous short dialogue summaries. Do not use tools,
files, outside knowledge or network. Only source_turns are factual authority; references guide
salience but can be imperfect. Score each candidate independently; both can pass or fail.
The complete original dialogue is the ordered text of source_turns. Each has a turn_id.
Return COMPACT JSON without indentation or markdown, with exactly:
{"evaluations":[{"candidate_id":"...","faithfulness":0,"coverage":0,"coherence":0,
"verdict":"supported|contradicted|insufficient evidence","claims":[{"claim":"brief atomic claim",
"verdict":"supported|contradicted|insufficient evidence","evidence_ids":["T0"]}],
"critical_omissions":["..."],"rationale":"at most 25 words"}]}.
List ALL material factual claims, each in at most 15 words. Include implied outcomes,
numbers, attribution, dates, decisions and approvals. Be concise without omitting claims.
For supported claims cite the source turn IDs entailing them; at least one is required.
For contradicted claims cite conflicting source turns. For insufficient evidence, citations
may be empty. Never invent IDs or copy/rewrite source quotes. Code saves the original text.
Overall supported requires all material claims supported; any contradiction makes the
overall verdict contradicted; otherwise unsupported claims make it insufficient evidence.
Judge coverage for an approximately 32-word summary, not exhaustive reference reproduction.
'''+q.RUBRIC[q.RUBRIC.index('Use integer scores'):]


def source_turns(source):
    turns=[{'turn_id':f'T{i}','text':text} for i,text in enumerate(source.split('\n'))]
    if '\n'.join(x['text'] for x in turns)!=source:raise ValueError('lossless source turns required')
    return turns


def packet(case,candidates,reverse=False):
    cs=[{'candidate_id':x['candidate_id'],'text':x['text']} for x in candidates]
    if reverse:cs.reverse()
    return {'case_id':q.digest(['case',case['id']])[:20],
        'source_turns':source_turns(case['source']),'references':case['references'],'candidates':cs}


def parse(text,p):
    text=text.strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*','',text);text=re.sub(r'\s*```$','',text)
    obj=json.loads(text);ids={x['candidate_id'] for x in p['candidates']}
    if not isinstance(obj,dict) or set(obj)!={'evaluations'} or not isinstance(obj['evaluations'],list):raise ValueError('schema')
    turns={x['turn_id']:x['text'] for x in p['source_turns']};seen=set()
    for g in obj['evaluations']:
        if g.get('candidate_id') not in ids or g['candidate_id'] in seen:raise ValueError('candidate binding')
        seen.add(g['candidate_id'])
        for key in ('faithfulness','coverage','coherence'):
            if type(g.get(key)) is not int or not 0<=g[key]<=4:raise ValueError('score range')
        if g.get('verdict') not in ('supported','contradicted','insufficient evidence'):raise ValueError('verdict')
        if not isinstance(g.get('claims'),list) or not isinstance(g.get('critical_omissions'),list) or not isinstance(g.get('rationale'),str) or not g['rationale'].strip():raise ValueError('rationale/claims')
        if any(not isinstance(x,str) or not x.strip() for x in g['critical_omissions']):raise ValueError('omissions')
        for c in g['claims']:
            if not isinstance(c.get('claim'),str) or not c['claim'].strip() or c.get('verdict') not in ('supported','contradicted','insufficient evidence'):raise ValueError('claim schema')
            evidence=c.get('evidence_ids')
            if not isinstance(evidence,list) or any(not isinstance(x,str) or x not in turns for x in evidence):raise ValueError('unknown source turn')
            if c['verdict']=='supported' and not evidence:raise ValueError('supported claim without source')
            c['source_quotes']=[turns[x] for x in evidence]
            indices=[int(x[1:]) for x in evidence]
            c['source_quote']='\n'.join(p['source_turns'][i]['text'] for i in range(min(indices),max(indices)+1)) if indices else ''
        if g['verdict']=='supported' and any(c['verdict']!='supported' for c in g['claims']):raise ValueError('inconsistent supported verdict')
    if seen!=ids:raise ValueError('missing candidate')
    return obj


def freeze(root,stage,inputs,judge):
    extras=[Path(__file__),PLAN,Path('experiments/neural_architecture_lab/finalize_ncm_quality.py'),
        Path('src/neural_architecture_lab/ncm_quality_hypothesis.py'),Path('tests/structure_net/test_ncm_quality_evidence.py')]
    p=q.freeze(root,stage,list(dict.fromkeys(inputs+extras)),{'evidence_protocol':'source_turn_ids_v3',
        'rubric':RUBRIC,'judge_role':judge,'requested_judge':q.JUDGES.get(judge),
        'workers':q.C.secondary_workers if judge=='secondary' else q.C.workers})
    for i,path in enumerate(extras):shutil.copyfile(path,root/'producer_source'/f'v3_{i}_{path.name}')
    return p


def request(p,judge,root):
    import requests
    key=next(x.split('=',1)[1].strip().strip('"').strip("'") for x in Path('.env').read_text().splitlines() if x.startswith('LITELLM_KEY='))
    payload={'model':q.JUDGES[judge],'temperature':0,'max_tokens':3072,
        'messages':[{'role':'system','content':RUBRIC},{'role':'user','content':json.dumps(p,ensure_ascii=False)}]}
    paths=[]
    for attempt in range(3):
        path=root/'attempts'/f"{p['case_id']}_{attempt}.json"
        if path.exists():raise ValueError('attempt overwrite')
        start=time.monotonic();raw={'request':payload,'request_sha256':q.digest(payload),'attempt':attempt}
        try:
            r=requests.post('https://central-dev.zt:4000/v1/chat/completions',headers={'Authorization':'Bearer '+key},
                json=payload,verify='/home/rabbit/.config/lakefs/caddy-root.crt',timeout=240)
            raw.update(http_status=r.status_code,response=r.json());r.raise_for_status()
            grade=parse(raw['response']['choices'][0]['message']['content'],p)
            if raw['response'].get('model')!=q.JUDGES[judge]:raise ValueError('judge alias changed')
            raw['valid']=True
        except Exception as e:
            raw.update(valid=False,error_type=type(e).__name__)
            if isinstance(e,ValueError):raw['error']=str(e)[:200]
        raw['wall_seconds']=time.monotonic()-start;q.write(path,raw);paths.append(str(path))
        if raw['valid']:return {'case_id':p['case_id'],'packet_sha256':q.digest(p),'status':'valid',
            'judge':q.JUDGES[judge],'evaluations':grade['evaluations'],'attempts':paths}
    return {'case_id':p['case_id'],'packet_sha256':q.digest(p),'status':'invalid','judge':q.JUDGES[judge],'attempts':paths}


def execute(packets,judge,root,stop_invalid=False):
    rows=[]
    with ThreadPoolExecutor(max_workers=1 if judge=='secondary' else 4) as pool:
        jobs=[pool.submit(request,p,judge,root) for p in packets] if not stop_invalid else []
        if stop_invalid:
            for p in packets:
                row=request(p,judge,root);rows.append(row);q.write(root/'rows'/f"{row['case_id']}.json",row)
                print(json.dumps({'completed':len(rows),'total':len(packets),'status':row['status']}),flush=True)
                if row['status']=='invalid':break
        else:
            for f in as_completed(jobs):
                row=f.result();rows.append(row);q.write(root/'rows'/f"{row['case_id']}.json",row)
                print(json.dumps({'completed':len(rows),'total':len(packets),'status':row['status']}),flush=True)
    return sorted(rows,key=lambda x:x['case_id'])


def calibration_packets():
    cases=q.read(q.DATA/'calibration.json')['pairs']
    return [packet(x,x['candidates'],bool(int(q.digest(x['id']),16)%2)) for x in cases]


def calibrate(args):
    from experiments.structure_net.ncm_llama import tokenizers
    _,llama=tokenizers();p=freeze(args.output,'v3_known_answer_calibration',[q.DATA/'calibration.json'],args.judge)
    packets=calibration_packets();q.write(args.output/'packets.json',{'packets':packets})
    rows=execute(packets,args.judge,args.output,stop_invalid=args.judge=='secondary')
    expected={c['candidate_id']:c for x in q.read(q.DATA/'calibration.json')['pairs'] for c in x['candidates']}
    total=correct=factual=rejected=0
    for r in rows:
        if r['status']!='valid':continue
        for g in r['evaluations']:
            c=expected[g['candidate_id']];decision=q.passes(g,c['text'],len(llama.encode(c['text'],add_special_tokens=False)))
            total+=1;correct+=decision==c['expected_pass']
            if c['kind'] in ('contradiction','unsupported'):factual+=1;rejected+=not decision
    gate=total==48 and correct>=44 and factual==rejected
    q.write(args.output/'judgments.json',{'rows':rows})
    q.seal(args.output,p,{'status':'passed' if gate else 'calibration_failed','judge':q.JUDGES[args.judge],
        'correct':correct,'total':total,'expected':48,'factual_negatives':factual,'factual_negatives_rejected':rejected,
        'pairs':len(rows),'invalid_pairs':sum(r['status']!='valid' for r in rows),
        'retried_pairs':sum(len(r['attempts'])>1 for r in rows),'judgments_sha256':q.sha256(args.output/'judgments.json')})


def model_outputs():
    outputs={}
    for model in ('ncm','llama'):
        root=q.BASE/f'20261021_{model}_generation_v1';_,r=verify(root)
        if r['status']!='completed' or r['cases']!=240 or q.sha256(root/'outputs.json')!=r['outputs_sha256']:raise ValueError('frozen generation integrity')
        outputs[model]={x['id']:x for x in q.read(root/'outputs.json')['rows']}
    return outputs


def chosen_cases(judge):
    cases=q.read(q.DATA/'cases.json')['rows']
    if judge=='primary':return cases
    return [x for cohort,n in [('dialogsum',50),('stress',20)] for x in sorted(
        [c for c in cases if c['cohort']==cohort],key=lambda c:q.digest(['audit',c['id']]))[:n]]


def model_packets(judge):
    out=model_outputs();packets=[]
    for c in chosen_cases(judge):
        candidates=[{'candidate_id':q.digest(['candidate',c['id'],m])[:20],'text':out[m][c['id']]['text']} for m in ('ncm','llama')]
        packets.append(packet(c,candidates,bool(int(q.digest(['order',c['id']]),16)%2)^(judge=='secondary')))
    return packets


def judge(args):
    cal=BASE/f'20261021_calibration_{args.judge}_v1';_,r=verify(cal)
    if r['status']!='passed':raise ValueError('judge calibration gate')
    inputs=[q.DATA/'cases.json',cal/'result.json']+[q.BASE/f'20261021_{m}_generation_v1/outputs.json' for m in ('ncm','llama')]
    p=freeze(args.output,'v3_blinded_independent_grading',inputs,args.judge)
    packets=model_packets(args.judge);q.write(args.output/'packets.json',{'packets':packets})
    rows=execute(packets,args.judge,args.output);q.write(args.output/'judgments.json',{'rows':rows})
    q.seal(args.output,p,{'status':'completed' if all(r['status']=='valid' for r in rows) else 'invalid_judgments',
        'judge':q.JUDGES[args.judge],'pairs':len(rows),'invalid_pairs':sum(r['status']!='valid' for r in rows),
        'retried_pairs':sum(len(r['attempts'])>1 for r in rows),'judgments_sha256':q.sha256(args.output/'judgments.json')})


def audit(root,packets,role):
    _,result=verify(root);rows=q.read(root/'judgments.json')['rows'];lookup={p['case_id']:p for p in packets}
    if q.read(root/'packets.json')['packets']!=packets or q.sha256(root/'judgments.json')!=result['judgments_sha256']:raise ValueError('packet/result binding')
    if len(rows)!=len({r['case_id'] for r in rows}) or any(r['case_id'] not in lookup for r in rows):raise ValueError('duplicate/unknown rows')
    for r in rows:
        p=lookup[r['case_id']]
        if r['packet_sha256']!=q.digest(p) or r['judge']!=q.JUDGES[role]:raise ValueError('response identity')
        for i,path in enumerate(r['attempts']):
            a=q.read(path);payload={'model':q.JUDGES[role],'temperature':0,'max_tokens':3072,
                'messages':[{'role':'system','content':RUBRIC},{'role':'user','content':json.dumps(p,ensure_ascii=False)}]}
            if a['attempt']!=i or a['request']!=payload or a['request_sha256']!=q.digest(payload):raise ValueError('request changed')
            if i<len(r['attempts'])-1 and a['valid']:raise ValueError('quality decision retried')
        a=q.read(r['attempts'][-1])
        if r['status']=='valid':
            if not a['valid'] or a['http_status']!=200 or a['response']['model']!=q.JUDGES[role] or parse(a['response']['choices'][0]['message']['content'],p)['evaluations']!=r['evaluations']:raise ValueError('raw evaluation differs')
    return rows,result


def aggregate(args):
    from experiments.structure_net.ncm_llama import tokenizers,decode_ids
    from experiments.structure_net.ncm_vocabulary import CachedVocabulary
    if args.output.exists():raise ValueError('fresh aggregate required')
    gpt,llama=tokenizers();gpt=CachedVocabulary(gpt);outputs=model_outputs();cases=q.read(q.DATA/'cases.json')['rows'];case_map={c['id']:c for c in cases}
    for m,lookup in outputs.items():
        if set(lookup)!=set(case_map):raise ValueError('complete output cells')
        for i,o in lookup.items():
            if o['source_sha256']!=case_map[i]['source_sha256'] or q.digest(case_map[i]['source'])!=o['source_sha256'] or q.digest(o['text'])!=o['text_sha256']:raise ValueError('source/text changed')
            if len(llama.encode(o['text'],add_special_tokens=False))!=o['common_output_tokens']:raise ValueError('output token count')
            if m=='ncm' and decode_ids(o['native_output_ids'],gpt)!=(o['text'],o['termination']):raise ValueError('NCM native decode')
            if m=='llama' and llama.decode(o['native_output_ids'],skip_special_tokens=True).strip()!=o['text']:raise ValueError('Llama native decode')
    flat={};calibration={};judges={};roots=[]
    for role in ('primary','secondary'):
        cal=BASE/f'20261021_calibration_{role}_v1';roots.append(cal)
        rows,cr=audit(cal,calibration_packets(),role);calibration[role]=cr
        known={c['candidate_id']:c for x in q.calibration_cases() for c in x['candidates']};correct=total=neg=reject=0
        for r in rows:
            if r['status']!='valid':continue
            for g in r['evaluations']:
                c=known[g['candidate_id']];p=q.passes(g,c['text'],len(llama.encode(c['text'],add_special_tokens=False)))
                total+=1;correct+=p==c['expected_pass']
                if c['kind'] in ('contradiction','unsupported'):neg+=1;reject+=not p
        if (correct,total,neg,reject)!=(cr['correct'],cr['total'],cr['factual_negatives'],cr['factual_negatives_rejected']):raise ValueError('calibration recomputation')
        if cr['status']!='passed':
            if role=='primary':raise ValueError('primary invalid')
            continue
        if total!=48 or correct<44 or neg!=reject:raise ValueError('calibration gate inconsistent')
        root=BASE/f'20261021_judge_{role}_v1';roots.append(root);rows,result=audit(root,model_packets(role),role)
        if result['status']!='completed' or len(rows)!=(240 if role=='primary' else 70) or any(r['status']!='valid' for r in rows):raise ValueError('invalid endpoint')
        index={q.digest(['candidate',c['id'],m])[:20]:(c,m) for c in chosen_cases(role) for m in outputs};flat[role]=[]
        for row in rows:
            for g in row['evaluations']:
                c,m=index[g['candidate_id']];o=outputs[m][c['id']]
                flat[role].append({**o,'cohort':c['cohort'],'family':c['family'],'grade':g,
                    'passed':q.passes(g,o['text'],o['common_output_tokens'])})
        judges[role]=result
    summary={cohort:{m:summarize([x for x in flat['primary'] if x['model']==m and x['cohort']==cohort]) for m in outputs} for cohort in ('dialogsum','stress')}
    paired={cohort:paired_ci(*[{x['id']:x['passed'] for x in flat['primary'] if x['model']==m and x['cohort']==cohort} for m in ('ncm','llama')]) for cohort in summary}
    agree={}
    if 'secondary' in flat:
        for cohort in summary:
            agree[cohort]={}
            for m in outputs:
                second={x['id']:x['passed'] for x in flat['secondary'] if x['model']==m and x['cohort']==cohort}
                first={x['id']:x['passed'] for x in flat['primary'] if x['model']==m and x['cohort']==cohort and x['id'] in second}
                agree[cohort][m]=agreement(first,second)
    else:agree={'status':'unavailable','reason':'secondary judge calibration failed'}
    q.write(args.output.parent/'case_scores.json',flat)
    prior=[q.BASE.parent/'20261021_calibration_primary_v1',q.BASE/'20261021_judge_primary_v1']
    paths=[args.output.parent/'case_scores.json',q.DATA/'cases.json',q.DATA/'producer.json',q.DATA/'result.json',
        q.BASE/'20261021_dependency_freeze_v1/manifest.json']
    for root in roots+prior+[q.BASE/f'20261021_{m}_generation_v1' for m in outputs]:paths+=list(root.rglob('*.json'))
    primary=summary['dialogsum'];decision=classification(True,primary['ncm']['passes'],primary['llama']['passes'])
    result={'schema':'nal.ncm-quality-campaign.v3','hypothesis_id':q.HYPOTHESIS_ID,'status':'completed_exploratory',
        'classification':decision,'confirmed':False,'underpowered':True,'independent_adaptation_seeds':1,
        'primary':primary,'stress':summary['stress'],'paired':paired,'judge_agreement':agree,'judge_calibration':calibration,
        'judge_results':judges,'requested_model_outputs':480,'completed_model_outputs':480,'failed_model_outputs':0,
        'retried_model_outputs':0,'reused_model_outputs':480,'primary_judgments':480,'secondary_judgments':len(flat.get('secondary',[])),
        'invalid_judgment_pairs':sum(x['invalid_pairs'] for x in judges.values()),
        'retried_judgment_pairs':sum(x['retried_pairs'] for x in judges.values()),
        'prior_invalid_stages':[q.read(root/'result.json') for root in prior],
        'point_target_met':primary['ncm']['passes']>=190,'population_lower95_target_met':primary['ncm']['wilson95'][0]>=.95,
        'retained_teacher_point_quality':primary['ncm']['passes']>=primary['llama']['passes'],
        'application_accuracy_tested':False,'speed_tested':False,'mobile_performance_tested':False,
        'scientific_boundary':'One frozen model; public dialogue proxy; calibrated model labels, not human gold. Evidence-schema recovery after limited outcome exposure; unchanged quality rules. Original complete v2 endpoint invalid. No production/mobile/speed claim.',
        'artifacts':{str(p):q.sha256(p) for p in dict.fromkeys(paths)}}
    q.write(args.output,result);print(json.dumps({k:result[k] for k in ('classification','primary','stress','paired','judge_agreement')}))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('calibrate','judge'):
        a=sub.add_parser(name);a.add_argument('--judge',choices=['primary','secondary'],required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('aggregate');a.add_argument('--output',type=Path,required=True)
    args=p.parse_args();globals()[args.command](args)


if __name__=='__main__':main()
