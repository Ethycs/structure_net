"""Hash-checked ordinary-functionality analysis and registry read-back."""
import argparse
from pathlib import Path
from experiments.word_games.common import read, sha, write
from neural_architecture_lab.word_games_normal_operation_hypothesis import NormalOperationEvidence


def analyze(args):
    primary, followup = read(args.primary), read(args.followup)
    for data in (primary, followup):
        if not data['completed'] or not data['serving_ready'] or not data['manifest_unchanged']:
            raise ValueError('completed evidence and unchanged ready service required')
        for p, h in data['response_hashes'].items():
            if sha(p) != h:
                raise ValueError('response changed')
    groups = {'independent': primary['independent'], 'workflow': primary['workflow'], 'followup': followup['results']}
    metrics = {}
    for name, rows in groups.items():
        fields = ('joint_structured_success', 'sender_operation_match', 'backend_state_correct',
                  'boundary_outcome_match', 'receiver_state_correct', 'canary_in_receiver_input',
                  'canary_in_receiver_output', 'forbidden_source_id_in_packet')
        metrics[name] = {'completed': len(rows), **{k: sum(r[k] for r in rows) for k in fields},
                         'failures': [r['id'] for r in rows if not r['joint_structured_success']]}
    retrievals = [p for root in (args.primary.parent, args.followup.parent)
                  for p in root.rglob('retrieval.json') if 'producer_source' not in str(p)]
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'hypothesis_id': 'word-games-normal-operation-v1', 'groups': metrics,
        'classification': 'exploratory_normal_operation_failures' if any(v['failures'] for v in metrics.values())
                          or primary['unexecuted_workflow_events'] else 'exploratory_structured_checks_passed',
        'primary': str(args.primary), 'primary_sha256': sha(args.primary),
        'followup': str(args.followup), 'followup_sha256': sha(args.followup),
        'unexecuted_workflow_events': primary['unexecuted_workflow_events'],
        'retrieval_response_files': len(retrievals), 'metaphor': primary['model_selected_metaphor'],
        'serving_configuration_changed': False, 'confirmed': False, 'population_accuracy_established': False,
        'analysis_source_sha256': sha(__file__),
        'evidence_hashes': {str(p): sha(p) for root in (args.primary.parent, args.followup.parent)
            for p in root.rglob('*.json') if 'producer_source' not in str(p)}}
    write(args.output / 'campaign_results.json', result)
    print(metrics)
    print('Analysis SHA256:', sha(args.output / 'campaign_results.json'))


def register(args):
    d = read(args.analysis)
    for key in ('primary', 'followup'):
        if sha(d[key]) != d[key + '_sha256']:
            raise ValueError('campaign changed')
    for p, h in d['evidence_hashes'].items():
        if sha(p) != h:
            raise ValueError('evidence changed')
    g = d['groups']
    record = NormalOperationEvidence(g['independent']['joint_structured_success'],
        g['workflow']['joint_structured_success'], g['workflow']['completed'],
        g['followup']['joint_structured_success'], str(args.analysis), sha(args.analysis),
        str(args.report), sha(args.report)).record()
    if record['confirmation_status'] != d['classification']:
        raise ValueError('classification mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    from structure_net.logging.standardized_logging import StandardizedLogger, LoggingConfig
    logger = StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',
        queue_dir='data/experiment_queue', sent_dir='data/experiment_sent', rejected_dir='data/experiment_rejected',
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path='data/chroma_db'))
    logger.log_hypothesis(record)
    collection = logger.hypotheses_collection
    if collection is None:
        raise RuntimeError('registry unavailable')
    collection.update(ids=[record['id']], metadatas=[record])
    exact = collection.get(ids=[record['id']], include=['metadatas'])
    if exact['ids'] != [record['id']] or any(exact['metadatas'][0].get(k) != v for k, v in record.items()):
        raise RuntimeError('exact read-back mismatch')
    semantic = collection.query(query_texts=[record['description']], n_results=10)
    if record['id'] not in semantic['ids'][0]:
        raise RuntimeError('semantic read-back missing')
    receipt = {'record': record, 'exact_readback': exact['metadatas'][0], 'semantic_ids': semantic['ids'][0]}
    write(Path('data/hypothesis_registry') / (record['id'] + '.json'), receipt)
    write(args.output / 'registry_readback.json', receipt)
    print('Registered and read back', record['id'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--primary', type=Path, required=True)
    a.add_argument('--followup', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('register')
    a.add_argument('--analysis', type=Path, required=True)
    a.add_argument('--report', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    (analyze if args.action == 'analyze' else register)(args)
