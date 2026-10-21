"""Reconstruct and audit independent quality scores from raw blinded judge replies."""
import argparse
import json
from pathlib import Path
import random
import shutil

from experiments.structure_net.ncm_quality import (
    BASE,DATA,PLAN,C,JUDGES,RUBRIC,HYPOTHESIS_ID,read,write,digest,sha256,unchanged,
    parse_grade,passes,blind_packet,wilson,calibration_cases)
from neural_architecture_lab.ncm_quality_hypothesis import QualityEvidence,classification


def verify(root):
    p=read(root/'producer.json');r=read(root/'result.json')
    if digest({k:v for k,v in p.items() if k!='fingerprint'})!=p['fingerprint']:raise ValueError('producer fingerprint')
    for path,h in p['input_hashes'].items():
        if sha256(path)!=h:raise ValueError('archived producer input changed')
    if p['hypothesis_id']!=HYPOTHESIS_ID or r['producer_fingerprint']!=p['fingerprint']:raise ValueError('terminal binding')
    copies={sha256(x) for x in (root/'producer_source').iterdir()}
    if any(h not in copies for h in p['source_hashes'].values()):raise ValueError('snapshot missing')
    return p,r


def audit_judgments(root,packets,judge):
    expected={p['case_id']:p for p in packets}
    rows=[read(p) for p in sorted((root/'rows').glob('*.json'))]
    if len(rows)!=len(expected) or {x['case_id'] for x in rows}!=set(expected):raise ValueError('complete unique judge rows required')
    if read(root/'packets.json')['packets']!=packets:raise ValueError('frozen packets differ')
    for row in rows:
        packet=expected[row['case_id']]
        if row['packet_sha256']!=digest(packet) or row['judge']!=JUDGES[judge]:raise ValueError('judge binding')
        if not 1<=len(row['attempts'])<=3:raise ValueError('retry count')
        request={'model':JUDGES[judge],'temperature':0,'max_tokens':4096,
            'messages':[{'role':'system','content':RUBRIC},{'role':'user','content':json.dumps(packet,ensure_ascii=False)}]}
        for i,path in enumerate(row['attempts']):
            a=read(path)
            if a['attempt']!=i or a['request']!=request or a['request_sha256']!=digest(request):raise ValueError('attempt identity')
            if i<len(row['attempts'])-1 and a['valid']:raise ValueError('quality outcome retried')
        a=read(row['attempts'][-1])
        if row['status']=='valid':
            if not a['valid'] or a['http_status']!=200 or a['response']['model']!=JUDGES[judge]:raise ValueError('valid transport')
            rebuilt=parse_grade(a['response']['choices'][0]['message']['content'],packet)
            if rebuilt['evaluations']!=row['evaluations']:raise ValueError('parsed grade changed')
        elif a['valid']:raise ValueError('invalid result differs')
    return rows


def audit_calibration(judge,llama):
    root=BASE/f'20261021_calibration_{judge}_v1';_,r=verify(root)
    cases=read(DATA/'calibration.json')['pairs']
    if cases!=calibration_cases():raise ValueError('calibration fixtures differ')
    packets=[]
    for x in cases:
        cs=[{'candidate_id':c['candidate_id'],'text':c['text']} for c in x['candidates']]
        if int(digest(x['id']),16)%2:cs.reverse()
        packets.append({'case_id':digest(x['id'])[:20],'source':x['source'],'references':x['references'],'candidates':cs})
    rows=audit_judgments(root,packets,judge);lookup={c['candidate_id']:c for x in cases for c in x['candidates']}
    correct=total=factual=rejected=0
    for row in rows:
        if row['status']!='valid':continue
        for g in row['evaluations']:
            c=lookup[g['candidate_id']];p=passes(g,c['text'],len(llama.encode(c['text'],add_special_tokens=False)))
            correct+=p==c['expected_pass'];total+=1
            if c['kind'] in ('contradiction','unsupported'):factual+=1;rejected+=not p
    gate=total==48 and correct>=44 and factual==rejected
    if (correct,total,factual,rejected)!=(r['correct'],r['total'],r['factual_negatives'],r['factual_negatives_rejected']):raise ValueError('calibration aggregate changed')
    if not gate or r['status']!='passed':raise ValueError('calibration failed')
    return r


def summarize(rows):
    n=len(rows);k=sum(x['passed'] for x in rows)
    claims=[c for x in rows for c in x['grade']['claims']]
    fully=sum(bool(x['text'].strip() and x['grade']['claims'] and x['grade']['faithfulness']==4 and
        x['grade']['verdict']=='supported' and all(c['verdict']=='supported' for c in x['grade']['claims'])) for x in rows)
    return {'n':n,'passes':k,'pass_rate':k/n,'wilson95':wilson(k,n),
        'fully_supported_summaries':fully,'fully_supported_rate':fully/n,
        'coverage_failures':sum(x['grade']['coverage']<3 or bool(x['grade']['critical_omissions']) for x in rows),
        'coherence_failures':sum(x['grade']['coherence']<3 for x in rows),
        'empty_outputs':sum(not x['text'].strip() for x in rows),'over128_outputs':sum(x['common_output_tokens']>128 for x in rows),
        'canvas_or_token_limit_outputs':sum(x['termination'] in ('canvas_limit','token_limit') for x in rows),
        'claims':len(claims),'supported_claims':sum(c['verdict']=='supported' for c in claims),
        'claim_support_fraction':sum(c['verdict']=='supported' for c in claims)/len(claims) if claims else None,
        'mean_scores':{k:sum(x['grade'][k] for x in rows)/n for k in ('faithfulness','coverage','coherence')}}


def paired_ci(a,b):
    # Teacher minus student. Matched dialogues, not independent model training seeds.
    if set(a)!=set(b):raise ValueError('paired keys required')
    values=[int(b[k])-int(a[k]) for k in sorted(a)];n=len(values);rng=random.Random(20261021)
    samples=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
    return {'teacher_minus_student':sum(values)/n,'paired_bootstrap95':[samples[249],samples[9749]],
        'bootstrap_draws':10000,'bootstrap_seed':20261021,
        'both_pass':sum(a[k] and b[k] for k in a),'student_only_pass':sum(a[k] and not b[k] for k in a),
        'teacher_only_pass':sum(b[k] and not a[k] for k in a),'both_fail':sum(not a[k] and not b[k] for k in a)}


def agreement(a,b):
    if set(a)!=set(b) or not a:raise ValueError('complete nonempty agreement keys required')
    n=len(a);observed=sum(a[k]==b[k] for k in a)/n;p=sum(a.values())/n;q=sum(b.values())/n
    chance=p*q+(1-p)*(1-q)
    return {'n':n,'agreement':observed,'disagreements':sum(a[k]!=b[k] for k in a),
        'cohen_kappa':(observed-chance)/(1-chance) if chance<1 else None,
        'primary_passes':sum(a.values()),'secondary_passes':sum(b.values())}


def finalize(args):
    from experiments.structure_net.ncm_llama import tokenizers,decode_ids
    from experiments.structure_net.ncm_vocabulary import CachedVocabulary
    if args.output.exists():raise ValueError('fresh aggregate required')
    _,dr=verify(DATA);gpt,llama=tokenizers();gpt=CachedVocabulary(gpt)
    cases=read(DATA/'cases.json')['rows'];case_index={x['id']:x for x in cases};outputs={};all_roots=[DATA]
    if dr['cases_sha256']!=sha256(DATA/'cases.json') or dr['calibration_sha256']!=sha256(DATA/'calibration.json'):raise ValueError('frozen cases changed')
    manifest=BASE/'20261021_dependency_freeze_v1/manifest.json';mf=read(manifest)
    for path,h in mf['hashes'].items():
        if sha256(path)!=h:raise ValueError('prospective dependency changed')
    coordinator=BASE/'20261021_coordinator_v1';execution=read(coordinator/'execution.json')
    if execution['status']!='completed' or execution['source_sha256']!=sha256(coordinator/'producer.py') or execution['dependency_manifest_sha256']!=sha256(manifest):raise ValueError('coordinator identity/terminal')
    names=[x['name'] for x in execution['runs']]
    if names!=['ncm_generation','llama_generation','primary_judge','secondary_calibration','secondary_judge']:raise ValueError('coordinator schedule')
    for job in execution['runs']:
        if job.get('result_sha256') and sha256(Path(job['root'])/'result.json')!=job['result_sha256']:raise ValueError('coordinator child changed')
    for model in ('ncm','llama'):
        root=BASE/f'20261021_{model}_generation_v1';p,r=verify(root);all_roots.append(root)
        _,pilot=verify(BASE/f'20261021_{model}_pilot_v1');all_roots.append(BASE/f'20261021_{model}_pilot_v1')
        if pilot['status']!='passed' or pilot['cases']!=2:raise ValueError('pilot failed')
        if r['status']!='completed' or r['cases']!=240 or sha256(root/'outputs.json')!=r['outputs_sha256']:raise ValueError('generation coverage')
        rows=read(root/'outputs.json')['rows'];outputs[model]={x['id']:x for x in rows}
        if len(rows)!=240 or set(outputs[model])!=set(case_index):raise ValueError('model cases differ')
        for x in rows:
            c=case_index[x['id']]
            if x['source_sha256']!=c['source_sha256'] or digest(c['source'])!=c['source_sha256'] or digest(x['text'])!=x['text_sha256']:raise ValueError('source/output hash')
            if x['common_output_tokens']!=len(llama.encode(x['text'],add_special_tokens=False)):raise ValueError('common token count')
            if model=='ncm':
                if len(x['native_output_ids'])!=48 or decode_ids(x['native_output_ids'],gpt)!=(x['text'],x['termination']):raise ValueError('native NCM decode')
            elif llama.decode(x['native_output_ids'],skip_special_tokens=True).strip()!=x['text']:raise ValueError('native teacher decode')
    calibration={'primary':audit_calibration('primary',llama)};flat={};judge_results={}
    second_root=BASE/'20261021_calibration_secondary_v1'
    second_result=read(second_root/'result.json')
    calibration['secondary']=audit_calibration('secondary',llama) if second_result['status']=='passed' else second_result
    all_roots.append(second_root)
    active_judges=['primary','secondary'] if second_result['status']=='passed' else ['primary']
    for judge in active_judges:
        root=BASE/f'20261021_judge_{judge}_v1';_,result=verify(root);all_roots.extend([root,BASE/f'20261021_calibration_{judge}_v1'])
        chosen=cases
        if judge=='secondary':
            chosen=[]
            for cohort,n in [('dialogsum',50),('stress',20)]:chosen+=sorted([x for x in cases if x['cohort']==cohort],key=lambda x:digest(['audit',x['id']]))[:n]
        packets=[blind_packet(x,outputs,judge=='secondary') for x in chosen]
        rows=audit_judgments(root,packets,judge)
        if read(root/'judgments.json')['rows']!=rows or sha256(root/'judgments.json')!=result['judgments_sha256']:raise ValueError('judgment aggregate changed')
        if result['status']!='completed' or len(rows)!=(240 if judge=='primary' else 70) or any(x['status']!='valid' for x in rows):raise ValueError('incomplete endpoint')
        lookup={digest(['candidate',c['id'],model])[:20]:(c,model) for c in chosen for model in outputs};flat[judge]=[]
        for row in rows:
            for grade in row['evaluations']:
                c,model=lookup[grade['candidate_id']];o=outputs[model][c['id']]
                flat[judge].append({'id':c['id'],'cohort':c['cohort'],'family':c['family'],'model':model,'grade':grade,
                    'text':o['text'],'source_sha256':c['source_sha256'],'text_sha256':o['text_sha256'],
                    'common_output_tokens':o['common_output_tokens'],'termination':o['termination'],
                    'passed':passes(grade,o['text'],o['common_output_tokens'])})
        judge_results[judge]=result
    summary={cohort:{m:summarize([x for x in flat['primary'] if x['model']==m and x['cohort']==cohort]) for m in outputs} for cohort in ('dialogsum','stress')}
    paired={}
    for cohort in summary:
        maps={m:{x['id']:x['passed'] for x in flat['primary'] if x['model']==m and x['cohort']==cohort} for m in outputs}
        paired[cohort]=paired_ci(maps['ncm'],maps['llama'])
    agreements={} if 'secondary' in flat else {'status':'unavailable','reason':'secondary judge calibration did not pass'}
    for cohort in (('dialogsum','stress') if 'secondary' in flat else ()):
        agreements[cohort]={}
        for model in outputs:
            second={x['id']:x['passed'] for x in flat['secondary'] if x['cohort']==cohort and x['model']==model}
            first={x['id']:x['passed'] for x in flat['primary'] if x['cohort']==cohort and x['model']==model and x['id'] in second}
            agreements[cohort][model]=agreement(first,second)
    args.output.parent.mkdir(parents=True,exist_ok=True);write(args.output.parent/'case_scores.json',flat)
    paths=[manifest,args.output.parent/'case_scores.json',coordinator/'execution.json',coordinator/'producer.py',
        Path('data/experiments/ncm_quality/20261021_calibration_primary_v1/result.json')]
    for root in all_roots:paths+=list(root.rglob('*.json'))
    primary=summary['dialogsum'];decision=classification(True,primary['ncm']['passes'],primary['llama']['passes'])
    result={'schema':'nal.ncm-quality-campaign.v1','hypothesis_id':HYPOTHESIS_ID,'status':'completed_exploratory',
        'classification':decision,'confirmed':False,'underpowered':True,'independent_adaptation_seeds':1,
        'primary':primary,'stress':summary['stress'],'paired':paired,'judge_agreement':agreements,
        'judge_calibration':calibration,'judge_results':judge_results,'dataset':dr,
        'prior_transport_calibration':read('data/experiments/ncm_quality/20261021_calibration_primary_v1/result.json'),
        'requested_model_outputs':480,'completed_model_outputs':480,'failed_model_outputs':0,'retried_model_outputs':0,
        'reused_model_outputs':0,'primary_judgments':480,'secondary_judgments':len(flat.get('secondary',[])),
        'calibration_judgments':48+second_result.get('total',0),
        'invalid_judgment_pairs':sum(x['invalid_pairs'] for x in judge_results.values()),
        'retried_judgment_pairs':sum(x['retried_pairs'] for x in judge_results.values()),
        'point_target_met':primary['ncm']['passes']>=190,'population_lower95_target_met':primary['ncm']['wilson95'][0]>=.95,
        'retained_teacher_point_quality':primary['ncm']['passes']>=primary['llama']['passes'],
        'application_accuracy_tested':False,'speed_tested':False,'mobile_performance_tested':False,
        'scientific_boundary':'One frozen adaptation seed; public dialogue proxy and constructed stress cases, calibrated model judgments rather than human gold. Pretraining contamination and opaque judge aliases remain possible. No production/mobile/speed claim.',
        'artifacts':{str(p):sha256(p) for p in dict.fromkeys(paths)}}
    dst=args.output.parent/'analysis_source';dst.mkdir()
    sources=[Path(__file__),Path('src/neural_architecture_lab/ncm_quality_hypothesis.py')]
    for p in sources:shutil.copyfile(p,dst/p.name)
    result['analysis_source_hashes']={str(p):sha256(p) for p in sources};write(args.output,result)
    print(json.dumps({k:result[k] for k in ('classification','primary','stress','paired','judge_agreement')},allow_nan=False))


def register(args):
    c=read(args.campaign)
    for path,h in c['artifacts'].items():
        if sha256(path)!=h:raise ValueError('aggregate evidence changed')
    record=QualityEvidence(c['classification'],str(args.campaign),sha256(args.campaign),str(args.report),sha256(args.report),c['completed_model_outputs']).record()
    from structure_net.logging.standardized_logging import LoggingConfig,StandardizedLogger
    logger=StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',queue_dir='data/experiment_queue',
        sent_dir='data/experiment_sent',rejected_dir='data/experiment_rejected',enable_wandb=False,auto_upload=False,
        enable_chromadb=True,chromadb_path='data/chroma_db'))
    logger.log_hypothesis(record);exact=logger.hypotheses_collection.get(ids=[HYPOTHESIS_ID],include=['metadatas'])
    if exact['ids']!=[HYPOTHESIS_ID] or any(exact['metadatas'][0][k]!=record[k] for k in ('tested','confirmed','confirmation_status','direct_experiment_count')):raise ValueError('exact registry readback')
    semantic=logger.hypotheses_collection.query(query_texts=[record['description']],n_results=10)
    if HYPOTHESIS_ID not in semantic['ids'][0]:raise ValueError('semantic registry readback')
    receipt={'status':'registered_and_read_back','record':record,'exact_readback':exact['metadatas'][0],'semantic_ids':semantic['ids'][0]}
    for path in (args.campaign.parent/'registry_readback.json',Path('data/hypothesis_registry')/(HYPOTHESIS_ID+'.json')):
        if path.exists():raise ValueError('fresh receipt required')
        write(path,receipt)
        if read(path)['record']!=record:raise ValueError('full JSON registry readback')
    print(json.dumps(record))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('finalize');a.add_argument('--output',type=Path,required=True)
    a=s.add_parser('register');a.add_argument('--campaign',type=Path,required=True);a.add_argument('--report',type=Path,required=True)
    args=p.parse_args();globals()[args.command](args)


if __name__=='__main__':main()
