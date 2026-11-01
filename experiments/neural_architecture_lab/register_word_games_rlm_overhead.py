"""Register measured worker overhead with bounded coverage and timing claims."""
import argparse
from pathlib import Path

from experiments.word_games.common import read, write, sha


def run(args):
    result, audit = read(args.result), read(args.audit)
    if audit['status'] != 'independent_audit_passed' or audit['result_sha256'] != sha(args.result):
        raise ValueError('matching independent audit required')
    for name, expected in result['artifacts'].items():
        if sha(args.result.parent/name) != expected:
            raise ValueError('experiment artifact changed: '+name)
    if not result['completed'] or not audit['coverage_gate_passed']:
        raise ValueError('completed coverage-preserving comparison required')
    baseline, persistent = result['totals']['baseline'], result['totals']['persistent']
    record = {'id': result['study'], 'name': 'WORD_GAMES persistent worker environment',
        'description': 'Matched sixteen-question research conversations compare process-per-turn '
        'Claude Code against persistent stream-JSON and tokenizer processes. Estimated time outside '
        'provider calls fell from 16.404 to 1.452 seconds; coverage was 29 versus 31 of 36 facts '
        'and both recalled 11 of 11. Overall cloud-call work also differed; no population speedup claim.',
        'project': 'WORD_GAMES', 'category': 'inference', 'tested': True, 'confirmed': False,
        'confirmation_status': 'persistent_transport_boundary_and_coverage_pass',
        'direct_experiment_count': 1, 'actual_model': ','.join(audit['models']),
        'baseline_seconds': baseline['seconds'], 'persistent_seconds': persistent['seconds'],
        'baseline_non_api_seconds': baseline['non_api_seconds'],
        'persistent_non_api_seconds': persistent['non_api_seconds'],
        'observed_wall_speedup': baseline['seconds']/persistent['seconds'],
        'baseline_unique_facts': baseline['unique_facts'], 'persistent_unique_facts': persistent['unique_facts'],
        'recall_facts_each': persistent['recall_facts'], 'catalog_facts': 36,
        'population_accuracy_established': False, 'causal_cloud_speedup_established': False,
        'result': str(args.result), 'result_sha256': sha(args.result),
        'audit': str(args.audit), 'audit_sha256': sha(args.audit),
        'report': str(args.report), 'report_sha256': sha(args.report)}
    registry = Path('data/hypothesis_registry')/(record['id']+'.json')
    if registry.exists() or args.output.exists():
        raise FileExistsError('registration receipt already exists')
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
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
        raise RuntimeError('registry read-back mismatch')
    semantic = collection.query(query_texts=[record['description']], n_results=10)
    if record['id'] not in semantic['ids'][0]:
        raise RuntimeError('semantic read-back missing')
    receipt = {'record': record, 'exact_readback': exact['metadatas'][0],
        'semantic_ids': semantic['ids'][0], 'status': 'registered_and_read_back'}
    write(registry, receipt)
    write(args.output, receipt)
    print(receipt['status'], record['id'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('result', 'audit', 'report', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
