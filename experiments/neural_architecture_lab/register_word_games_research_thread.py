"""Register the audited bounded research channel without a model-accuracy claim."""
import argparse
from pathlib import Path

from experiments.word_games.common import read, write, sha


def run(args):
    result, audit = read(args.result), read(args.audit)
    if audit['result_sha256'] != sha(args.result) or audit['status'] != 'independently_audited':
        raise ValueError('matching independently audited result required')
    for name, expected in result['artifacts'].items():
        if sha(args.result.parent/name) != expected:
            raise ValueError('source artifact changed: '+name)
    if not result['completed'] or audit['unique_facts'] != result['unique_facts_released']:
        raise ValueError('result/audit mismatch')
    record = {'id': result['study'], 'name': 'Claude Code behind WORD_GAMES research bridge',
        'description': 'Persistent read-only Claude Code conversation through a host-authorized '
        '36-fact StructureNet research channel. Retrieved 30 unique facts in 16 frozen turns; '
        'recalled 11 of 11 requested facts without additional source reads. '
        'Host supplies canonical wording; catalog coverage is not model accuracy.',
        'project': 'WORD_GAMES', 'category': 'inference', 'tested': True, 'confirmed': False,
        'confirmation_status': audit['classification'], 'direct_experiment_count': 1,
        'tested_scope': 'one conversation, bounded canonical fact catalog and exact citation witnesses',
        'actual_model': ','.join(audit['models']), 'frozen_turns': audit['turns'],
        'unique_facts': audit['unique_facts'], 'catalog_facts': audit['authorized_catalog_facts'],
        'recall_facts': audit['recall_facts'], 'recall_expected_facts': audit['recall_expected_facts'],
        'native_output_tokens_per_client_second': audit['native_worker_output_tokens_per_client_second'],
        'public_reference_tokens_per_client_second': audit['public_reference_tokens_per_client_second'],
        'population_accuracy_established': False, 'general_free_text_firewall': False,
        'result': str(args.result), 'result_sha256': sha(args.result),
        'audit': str(args.audit), 'audit_sha256': sha(args.audit),
        'report': str(args.report), 'report_sha256': sha(args.report)}
    receipt_path = Path('data/hypothesis_registry')/(record['id']+'.json')
    if args.output.exists() or receipt_path.exists():
        raise FileExistsError('registry receipt already exists; use an explicit amendment')
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',
        queue_dir='data/experiment_queue', sent_dir='data/experiment_sent',
        rejected_dir='data/experiment_rejected', enable_wandb=False, auto_upload=False,
        enable_chromadb=True, chromadb_path='data/chroma_db'))
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
    write(receipt_path, receipt)
    write(args.output, receipt)
    print(receipt['status'], record['id'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('result', 'audit', 'report', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
