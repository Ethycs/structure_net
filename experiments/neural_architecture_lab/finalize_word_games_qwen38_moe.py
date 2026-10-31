"""Conservative, hash-checked standalone Qwen38 comparison and registration."""
import argparse
from pathlib import Path

from experiments.word_games.common import read, sha, write
from experiments.neural_architecture_lab.finalize_word_games_qwen_pair import checked

ROOT=Path('data/experiments/word_games')
COMPONENTS=('sender_operation_match','boundary_outcome_match','receiver_state_correct')


def replay(root):
    record=checked(root)
    responses={str(p.parent.relative_to(root/'runs')):read(p) for p in (root/'runs').rglob('response.json')}
    if len(responses)!=36: raise ValueError('complete 36-request replay required')
    count=sum(len(r['response']['content'].encode()) for r in responses.values())
    timings=[r['response']['timings'] for r in responses.values()]
    record.update(output_bytes=count,output_bytes_per_second=count/record['totals']['request_seconds'],
                  prompt_tokens=sum(t['prompt_n'] for t in timings),
                  prefill_seconds=sum(t['prompt_ms'] for t in timings)/1000,
                  decode_seconds=sum(t['predicted_ms'] for t in timings)/1000)
    return record,responses


def aggregate(out, qwen38_full, qwen38_live, qwen_full, manual_review):
    out.mkdir(parents=True,exist_ok=False)
    b,br=replay(qwen38_full)
    q,qr=replay(qwen_full)
    if br.keys()!=qr.keys(): raise ValueError('replay IDs differ')
    quality=checked(qwen38_live)
    if not quality['completed']:
        producer=read(qwen38_live/'producer.json')
        for path, expected in producer['source_hashes'].items():
            if not any(sha(x)==expected for x in (qwen38_live/'producer_source').glob('*_'+Path(path).name)):
                raise ValueError('missing frozen failed-run source')
    prior=ROOT/'20261031_qwen_baseline_live_v1'
    prior_quality=checked(prior)
    c=read(prior/'quality/campaign_results.json')
    prior_rows={p.parent.name:read(p) for p in (prior/'quality/cases').glob('*/result.json')}
    if len(prior_rows)!=12 or c['completed']!=12: raise ValueError('complete baseline required')
    for path, expected in c['artifacts'].items():
        if sha(path)!=expected: raise ValueError('changed baseline quality evidence')
    prior_quality.update(cases=12,metrics=c['metrics'],joint_successes=sum(all(r[k] for k in COMPONENTS) for r in prior_rows.values()),
                         quality_campaign_sha256=sha(prior/'quality/campaign_results.json'))
    regressions,improvements=[],[]
    if quality['completed']:
        c=read(qwen38_live/'quality/campaign_results.json')
        rows={p.parent.name:read(p) for p in (qwen38_live/'quality/cases').glob('*/result.json')}
        if len(rows)!=12 or c['completed']!=12: raise ValueError('twelve-case denominator required')
        for case,r in rows.items():
            for root in (prior,qwen38_live):
                p=root/'quality/cases'/case/'result.json'
                if sha(p)!=read(root/'quality/campaign_results.json')['artifacts'][str(p)]: raise ValueError('changed case')
            if read(prior/'quality/cases'/case/'context.json')!=read(qwen38_live/'quality/cases'/case/'context.json'):
                raise ValueError('paired authorization/context mismatch')
            for k in COMPONENTS:
                if prior_rows[case][k] and not r[k]: regressions.append([case,k])
                if not prior_rows[case][k] and r[k]: improvements.append([case,k])
        joint=sum(all(r[k] for k in COMPONENTS) for r in rows.values())
        leak=any(c['metrics'][k] for k in ('canary_in_receiver_input','canary_in_receiver_output','forbidden_source_id_in_packet'))
        classification=('disclosure_failure' if leak else 'higher_joint_task_success' if joint>prior_quality['joint_successes'] else
                        'unchanged_joint_task_success' if joint==prior_quality['joint_successes'] else 'lower_joint_task_success')
        quality.update(cases=12,joint_successes=joint,joint_accuracy=joint/12,metrics=c['metrics'],
                       quality_campaign_sha256=sha(qwen38_live/'quality/campaign_results.json'))
        review=read(manual_review)
        if review['method']!='manual_assistant_review' or review['independent_judge'] is not False:
            raise ValueError('manual review must not imply independent judging')
        if {r['id'] for r in review['cases']}!=rows.keys() or len(review['cases'])!=12:
            raise ValueError('manual review must cover twelve cases')
        for r in review['cases']:
            case_root=qwen38_live/'quality/cases'/r['id']
            if (r['packet_sha256']!=sha(case_root/'receiver_packet.json') or
                    r['response_sha256']!=sha(case_root/'receiver/response.json')):
                raise ValueError('reviewed response or packet changed')
            if r['assessment'] not in ('supported','issue') or not r['notes'].strip():
                raise ValueError('explicit manual assessment required')
        quality['manual_review']={'path':str(manual_review),'sha256':sha(manual_review),
                                  'supported':sum(r['assessment']=='supported' for r in review['cases']),
                                  'cases':12,'method':review['method']}
    else:
        quality.update(cases=0,joint_successes=0,joint_accuracy=None)
        classification='metaphor_gate_failed'
    pairs=[]
    for name in sorted(br):
        pairs.append({'id':name,'qwen38_seconds':br[name]['elapsed_seconds'],'qwen_seconds':qr[name]['elapsed_seconds'],
                      'exact_text_match':br[name]['response']['content']==qr[name]['response']['content']})
    result={'hypothesis_id':'word-games-qwen38-moe-v1','schema':'word-games.qwen38-moe.v1',
            'classification':classification,'confirmed':False,'qwen38_replay':b,'qwen_replay':q,
            'qwen38_quality':quality,'qwen_quality':prior_quality,'paired_requests':pairs,
            'component_regressions':regressions,'component_improvements':improvements,
            'paired_elapsed_ratio_qwen_over_qwen38':q['totals']['request_seconds']/b['totals']['request_seconds'],
            'independent_semantic_judge':False,'thinking':False,'speculation':False,'training':False,
            'cpu_expert_layers':26,'weight_quant':'Q4_K_M','placement':'RTX3060 plus CPU',
            'source_sha256':sha(__file__)}
    diagnostic=ROOT/'20261031_qwen38_replay_review_v1/review.json'
    if diagnostic.exists():
        review=read(diagnostic)
        expected={name for name in br if name.endswith('/receiver')}
        if len(review['cases'])!=12 or {r['id'] for r in review['cases']}!=expected:
            raise ValueError('incomplete frozen receiver diagnostic')
        for row in review['cases']:
            if row['response_sha256']!=sha(qwen38_full/'runs'/row['id']/'response.json'):
                raise ValueError('changed diagnostic response')
        result['frozen_receiver_diagnostic']={'path':str(diagnostic),'sha256':sha(diagnostic),
            'post_hoc':True,'live_workflows':False,'independent_judge':False,
            'structural_successes':review['structural_successes'],'supported_summaries':review['supported_summaries'],'cases':12}
    write(out/'campaign_results.json',result)
    print(classification,quality.get('joint_accuracy'),b['totals']['output_tokens_per_second'])


def register(campaign,report):
    from neural_architecture_lab.word_games_qwen38_moe_hypothesis import Qwen38Evidence
    c=read(campaign)
    for key in ('qwen38_replay','qwen_replay','qwen38_quality','qwen_quality'):
        r=c[key]
        name='campaign_results.json' if r['completed'] else 'quality/interruption.json'
        expected=r['campaign_sha256'] if r['completed'] else r['failure_sha256']
        if sha(Path(r['root'])/name)!=expected: raise ValueError('changed evidence')
        if not r['completed'] and any(sha(p)!=h for p,h in r['raw_hashes'].items()):
            raise ValueError('changed failed-run response')
        if r.get('quality_campaign_sha256') and sha(Path(r['root'])/'quality/campaign_results.json')!=r['quality_campaign_sha256']:
            raise ValueError('changed quality evidence')
    if c['qwen38_quality']['completed']:
        review=c['qwen38_quality']['manual_review']
        if sha(review['path'])!=review['sha256']: raise ValueError('changed review')
    if 'frozen_receiver_diagnostic' in c:
        diagnostic=c['frozen_receiver_diagnostic']
        if sha(diagnostic['path'])!=diagnostic['sha256']: raise ValueError('changed diagnostic review')
    record=Qwen38Evidence(c['classification'],c['qwen38_quality']['joint_successes'],c['qwen38_quality']['cases'],
        c['qwen38_replay']['totals']['output_tokens_per_second'],str(campaign),sha(campaign),str(report),sha(report),
        c['qwen38_quality'].get('manual_review',{}).get('supported',0)).record()
    from structure_net.logging.standardized_logging import LoggingConfig,StandardizedLogger
    logger=StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',
        queue_dir='data/experiment_queue',sent_dir='data/experiment_sent',rejected_dir='data/experiment_rejected',
        enable_wandb=False,auto_upload=False,enable_chromadb=True,chromadb_path='data/chroma_db'))
    logger.log_hypothesis(record)
    if logger.hypotheses_collection is None: raise RuntimeError('registry unavailable')
    logger.hypotheses_collection.update(ids=[record['id']],metadatas=[record])
    exact=logger.hypotheses_collection.get(ids=[record['id']],include=['metadatas'])
    if exact['ids']!=[record['id']]: raise RuntimeError('missing readback')
    for k in ('tested','confirmed','confirmation_status','joint_successes','cases_completed','output_tokens_per_second',
              'cpu_expert_layers','weight_quant','semantic_supported_summaries'):
        if exact['metadatas'][0][k]!=record[k]: raise RuntimeError('incorrect readback '+k)
    semantic=logger.hypotheses_collection.query(query_texts=[record['description']],n_results=10)
    if record['id'] not in semantic['ids'][0]: raise RuntimeError('missing semantic readback')
    receipt={'record':record,'exact_readback':exact['metadatas'][0],'semantic_ids':semantic['ids'][0]}
    for dest in (campaign.parent/'registry_readback.json',Path('data/hypothesis_registry')/(record['id']+'.json')):
        write(dest,receipt)
    print('registered_and_read_back',c['classification'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path);p.add_argument('--campaign',type=Path);p.add_argument('--report',type=Path)
    p.add_argument('--qwen38-full',type=Path,default=ROOT/'20261031_qwen38_full_v1')
    p.add_argument('--qwen38-live',type=Path,default=ROOT/'20261031_qwen38_live_v1')
    p.add_argument('--qwen-full',type=Path,default=ROOT/'20261031_qwen_control_full_v1')
    p.add_argument('--manual-review',type=Path,default=ROOT/'20261031_qwen38_manual_review_v1/review.json')
    a=p.parse_args()
    if a.output: aggregate(a.output,a.qwen38_full,a.qwen38_live,a.qwen_full,a.manual_review)
    else: register(a.campaign,a.report)
