"""Hash-check and summarize the Qwen model switch; optionally index the report."""
import argparse
from pathlib import Path

from experiments.word_games.common import digest, read, sha, write

ROOT = Path('data/experiments/word_games')


def checked(root):
    p = read(root / 'producer.json')
    fingerprint = p.pop('fingerprint')
    if digest(p) != fingerprint:
        raise ValueError('producer fingerprint changed')
    if not (root / 'campaign_results.json').exists():
        failure = read(root/'quality/interruption.json')
        if failure['status'] != 'metaphor_selection_failed' or failure['completed'] != 0:
            raise ValueError('unexpected incomplete campaign')
        return {'root':str(root),'completed':False,'failure':failure,
                'failure_sha256':sha(root/'quality/interruption.json'),
                'producer_sha256':sha(root/'producer.json'), 'fingerprint':fingerprint,
                'raw_hashes':{str(p):sha(p) for p in root.rglob('response.json')}}
    c = read(root / 'campaign_results.json')
    if c['fingerprint'] != fingerprint:
        raise ValueError('campaign fingerprint mismatch')
    for path, expected in c['artifacts'].items():
        if sha(path) != expected:
            raise ValueError('raw response changed: ' + path)
    for path, expected in p['source_hashes'].items():
        if not any(sha(x) == expected for x in (root/'producer_source').glob('*_' + Path(path).name)):
            raise ValueError('missing frozen source: ' + path)
    return {'root': str(root), 'completed':True, 'campaign_sha256': sha(root/'campaign_results.json'),
            'producer_sha256': sha(root/'producer.json'), 'totals': c['totals'],
            'fingerprint': fingerprint}


def aggregate(out):
    out.mkdir(parents=True, exist_ok=False)
    roots = {'baseline_pilot': ROOT/'20261031_qwen_baseline_pilot_v2',
             'draft_pilot': ROOT/'20261031_qwen_draft_pilot_v1',
             'baseline_quality': ROOT/'20261031_qwen_baseline_live_v1',
             'draft_quality': ROOT/'20261031_qwen_draft_live_v1'}
    result = {k: checked(v) for k,v in roots.items()}
    pilot_ids = ['metaphor','cases/normal_reserve/sender/turn_01','cases/cancel/receiver']
    exact, parsed = 0, 0
    for case in pilot_ids:
        a,b = (read(roots[k]/'runs'/case/'response.json') for k in ('baseline_pilot','draft_pilot'))
        exact += a['response']['content'] == b['response']['content']
        parsed += a['parsed'] == b['parsed']
    components = ('sender_operation_match','boundary_outcome_match','receiver_state_correct')
    arms = {}
    for key in ('baseline_quality','draft_quality'):
        root = roots[key] / 'quality'
        if not result[key]['completed']:
            result[key].update(cases=0,joint_successes=0,metrics={},case_accuracy=None)
            continue
        c = read(root/'campaign_results.json')
        if c['completed'] != 12:
            raise ValueError('twelve completed quality cases required')
        for path, expected in c['artifacts'].items():
            if sha(path) != expected:
                raise ValueError('quality case changed')
        arms[key] = {p.parent.name:read(p) for p in (root/'cases').glob('*/result.json')}
        result[key].update(quality_campaign_sha256=sha(root/'campaign_results.json'),
                           metrics=c['metrics'], metaphor=c['metaphor'],
                           joint_successes=sum(all(r[k] for k in components) for r in arms[key].values()),
                           cases=12)
    regressions, improvements = [], []
    for case, a in arms['baseline_quality'].items():
        if 'draft_quality' not in arms: break
        b = arms['draft_quality'][case]
        left, right = (read(roots[k]/'quality/cases'/case/'context.json') for k in ('baseline_quality','draft_quality'))
        if left != right:
            raise ValueError('paired contexts differ')
        for component in components:
            if a[component] and not b[component]: regressions.append([case,component])
            if not a[component] and b[component]: improvements.append([case,component])
    disclosure = ('canary_in_receiver_input','canary_in_receiver_output','forbidden_source_id_in_packet')
    leak = any(result[k]['metrics'][m] for k in arms for m in disclosure)
    ratio = result['draft_pilot']['totals']['output_tokens_per_second'] / result['baseline_pilot']['totals']['output_tokens_per_second']
    gate = result['draft_quality']['completed'] and ratio > 1 and not regressions and not leak
    classification = ('draft_metaphor_gate_failed' if not result['draft_quality']['completed'] else
                      'disclosure_failure' if leak else 'draft_quality_regressed' if regressions else
                      'draft_pilot_slower' if ratio <= 1 else 'observed_pilot_gain_no_suite_regression')
    check = ROOT/'20261031_qwen_tokenizers_v1/build/campaign_results.json'
    result.update(hypothesis_id='word-games-qwen25-pair-v1', schema='word-games.qwen-pair.v1',
                  classification=classification, confirmed=False, tested=True,
                  tokenizer_check=str(check), tokenizer_check_sha256=sha(check),
                  pilot_pairs=3, pilot_exact_text_matches=exact, pilot_parsed_json_matches=parsed,
                  pilot_speed_ratio=ratio, component_regressions=regressions, component_improvements=improvements,
                  observed_speculative_gate=gate, default_draft_tokens=0,
                  semantic_review_completed=False, trained_parameters=0,
                  startup_failures=[str(ROOT/'20261031_qwen_baseline_pilot_v1/interruption.json')],
                  limitations=['three frozen pilot requests, one run per arm, no full replay speed claim',
                               'twelve synthetic independent cases per quality arm; no semantic judge',
                               'unrelated idle services remain resident; not an isolated hardware benchmark',
                               'native target verification does not establish bitwise float batching equivalence'],
                  source_sha256=sha(__file__))
    write(out/'campaign_results.json', result)
    print(classification, 'pilot_speed_ratio', ratio)


def register(campaign, report):
    from neural_architecture_lab.word_games_qwen_pair_hypothesis import QwenPairEvidence
    c = read(campaign)
    for key in ('baseline_pilot','draft_pilot','baseline_quality','draft_quality'):
        root = Path(c[key]['root'])
        if c[key]['completed']:
            if sha(root/'campaign_results.json') != c[key]['campaign_sha256']:
                raise ValueError('changed campaign')
        else:
            if sha(root/'quality/interruption.json') != c[key]['failure_sha256']:
                raise ValueError('changed failure')
            if any(sha(p)!=h for p,h in c[key]['raw_hashes'].items()):
                raise ValueError('changed failed response')
    record = QwenPairEvidence(c['classification'], c['pilot_speed_ratio'],
        c['baseline_quality']['joint_successes'], c['draft_quality']['joint_successes'],
        str(campaign), sha(campaign), str(report), sha(report), c['draft_quality']['cases']).record()
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',
        queue_dir='data/experiment_queue', sent_dir='data/experiment_sent', rejected_dir='data/experiment_rejected',
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path='data/chroma_db'))
    logger.log_hypothesis(record)
    if logger.hypotheses_collection is None: raise RuntimeError('registry unavailable')
    logger.hypotheses_collection.update(ids=[record['id']], metadatas=[record])
    exact = logger.hypotheses_collection.get(ids=[record['id']], include=['metadatas'])
    if exact['ids'] != [record['id']]: raise RuntimeError('missing readback')
    for k in ('tested','confirmed','confirmation_status','pilot_speed_ratio','baseline_joint_successes','draft_joint_successes'):
        if exact['metadatas'][0][k] != record[k]: raise RuntimeError('incorrect readback: ' + k)
    semantic = logger.hypotheses_collection.query(query_texts=[record['description']], n_results=10)
    if record['id'] not in semantic['ids'][0]: raise RuntimeError('missing semantic readback')
    receipt = {'record': record, 'exact_readback': exact['metadatas'][0], 'semantic_ids': semantic['ids'][0]}
    for dest in (campaign.parent/'registry_readback.json', Path('data/hypothesis_registry')/(record['id']+'.json')):
        write(dest, receipt)
    print('registered_and_read_back', c['classification'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--campaign', type=Path)
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    if a.output: aggregate(a.output)
    else: register(a.campaign, a.report)
