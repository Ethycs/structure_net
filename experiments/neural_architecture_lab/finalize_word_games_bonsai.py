"""Conservative, hash-checked standalone Bonsai comparison and registration."""
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


def aggregate(out, bonsai_full, bonsai_live, qwen_full, manual_review):
    out.mkdir(parents=True,exist_ok=False)
    b,br=replay(bonsai_full)
    q,qr=replay(qwen_full)
    if br.keys()!=qr.keys(): raise ValueError('replay IDs differ')
    quality=checked(bonsai_live)
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
        c=read(bonsai_live/'quality/campaign_results.json')
        rows={p.parent.name:read(p) for p in (bonsai_live/'quality/cases').glob('*/result.json')}
        if len(rows)!=12 or c['completed']!=12: raise ValueError('twelve-case denominator required')
        for case,r in rows.items():
            for root in (prior,bonsai_live):
                p=root/'quality/cases'/case/'result.json'
                if sha(p)!=read(root/'quality/campaign_results.json')['artifacts'][str(p)]: raise ValueError('changed case')
            if read(prior/'quality/cases'/case/'context.json')!=read(bonsai_live/'quality/cases'/case/'context.json'):
                raise ValueError('paired authorization/context mismatch')
            for k in COMPONENTS:
                if prior_rows[case][k] and not r[k]: regressions.append([case,k])
                if not prior_rows[case][k] and r[k]: improvements.append([case,k])
        joint=sum(all(r[k] for k in COMPONENTS) for r in rows.values())
        leak=any(c['metrics'][k] for k in ('canary_in_receiver_input','canary_in_receiver_output','forbidden_source_id_in_packet'))
        classification=('disclosure_failure' if leak else 'higher_joint_task_success' if joint>prior_quality['joint_successes'] else
                        'unchanged_joint_task_success' if joint==prior_quality['joint_successes'] else 'lower_joint_task_success')
        quality.update(cases=12,joint_successes=joint,joint_accuracy=joint/12,metrics=c['metrics'],
                       quality_campaign_sha256=sha(bonsai_live/'quality/campaign_results.json'))
        review=read(manual_review)
        if review['method']!='manual_assistant_review' or review['independent_judge'] is not False:
            raise ValueError('manual review must not imply independent judging')
        if {r['id'] for r in review['cases']}!=rows.keys() or len(review['cases'])!=12:
            raise ValueError('manual review must cover twelve cases')
        for r in review['cases']:
            case_root=bonsai_live/'quality/cases'/r['id']
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
        pairs.append({'id':name,'bonsai_seconds':br[name]['elapsed_seconds'],'qwen_seconds':qr[name]['elapsed_seconds'],
                      'exact_text_match':br[name]['response']['content']==qr[name]['response']['content']})
    result={'hypothesis_id':'word-games-bonsai27-naked-v1','schema':'word-games.bonsai-naked.v1',
            'classification':classification,'confirmed':False,'bonsai_replay':b,'qwen_replay':q,
            'bonsai_quality':quality,'qwen_quality':prior_quality,'paired_requests':pairs,
            'component_regressions':regressions,'component_improvements':improvements,
            'paired_elapsed_ratio_qwen_over_bonsai':q['totals']['request_seconds']/b['totals']['request_seconds'],
            'independent_semantic_judge':False,'thinking':False,'speculation':False,'training':False,
            'source_sha256':sha(__file__)}
    write(out/'campaign_results.json',result)
    print(classification,quality.get('joint_accuracy'),b['totals']['output_tokens_per_second'])


def register(campaign,report):
    from neural_architecture_lab.word_games_bonsai_hypothesis import BonsaiEvidence
    c=read(campaign)
    for key in ('bonsai_replay','qwen_replay','bonsai_quality','qwen_quality'):
        r=c[key]
        name='campaign_results.json' if r['completed'] else 'quality/interruption.json'
        expected=r['campaign_sha256'] if r['completed'] else r['failure_sha256']
        if sha(Path(r['root'])/name)!=expected: raise ValueError('changed evidence')
        if r.get('quality_campaign_sha256') and sha(Path(r['root'])/'quality/campaign_results.json')!=r['quality_campaign_sha256']:
            raise ValueError('changed quality evidence')
    if c['bonsai_quality']['completed']:
        review=c['bonsai_quality']['manual_review']
        if sha(review['path'])!=review['sha256']: raise ValueError('changed review')
    record=BonsaiEvidence(c['classification'],c['bonsai_quality']['joint_successes'],c['bonsai_quality']['cases'],
        c['bonsai_replay']['totals']['output_tokens_per_second'],str(campaign),sha(campaign),str(report),sha(report)).record()
    from structure_net.logging.standardized_logging import LoggingConfig,StandardizedLogger
    logger=StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',
        queue_dir='data/experiment_queue',sent_dir='data/experiment_sent',rejected_dir='data/experiment_rejected',
        enable_wandb=False,auto_upload=False,enable_chromadb=True,chromadb_path='data/chroma_db'))
    logger.log_hypothesis(record)
    if logger.hypotheses_collection is None: raise RuntimeError('registry unavailable')
    logger.hypotheses_collection.update(ids=[record['id']],metadatas=[record])
    exact=logger.hypotheses_collection.get(ids=[record['id']],include=['metadatas'])
    if exact['ids']!=[record['id']]: raise RuntimeError('missing readback')
    for k in ('tested','confirmed','confirmation_status','joint_successes','cases_completed','output_tokens_per_second'):
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
    p.add_argument('--bonsai-full',type=Path,default=ROOT/'20261031_bonsai_full_v1')
    p.add_argument('--bonsai-live',type=Path,default=ROOT/'20261031_bonsai_live_v1')
    p.add_argument('--qwen-full',type=Path,default=ROOT/'20261031_qwen_control_full_v1')
    p.add_argument('--manual-review',type=Path,default=ROOT/'20261031_bonsai_manual_review_v1/review.json')
    a=p.parse_args()
    if a.output: aggregate(a.output,a.bonsai_full,a.bonsai_live,a.qwen_full,a.manual_review)
    else: register(a.campaign,a.report)
