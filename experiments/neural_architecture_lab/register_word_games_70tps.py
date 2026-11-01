"""Register the measured speed target with exact and semantic read-back."""
import argparse
from pathlib import Path

from experiments.word_games.common import read, sha, write
from neural_architecture_lab.word_games_70tps_hypothesis import SeventyTpsEvidence


def run(args):
    data = read(args.analysis)
    if not data['service_restored']:
        raise ValueError('service restoration required')
    for key in ('campaign', 'manual_review'):
        if sha(data[key]) != data[key + '_sha256']:
            raise ValueError('changed ' + key)
    best = data['best_arm']
    c = data['conditions'][best]
    record = SeventyTpsEvidence(best, data['conditions']['control']['tok_s'], c['tok_s'],
        c['receiver_correct'], c['receiver_cases'], str(args.analysis), sha(args.analysis),
        str(args.report), sha(args.report), valid_selectors=c['valid_selectors']).record()
    if record['confirmation_status'] != data['classification']:
        raise ValueError('classification mismatch')
    if args.triton_validation:
        serving = read(args.triton_validation)
        if not serving['completed'] or not serving['unload_release_reload_verified'] or not serving['model_ready']:
            raise ValueError('completed Triton lifecycle validation required')
        for path, expected in serving['response_hashes'].items():
            if sha(path) != expected:
                raise ValueError('changed Triton response')
        if len(serving['response_hashes']) != 72 or any(
                p['quality']['receiver_correct'] != 12 or not p['quality']['selector_valid'] or
                p['different_parsed_outputs_from_native'] for p in serving['passes']):
            raise ValueError('Triton configuration must pass existing checks and reproduce native outputs')
        producer = read(args.triton_validation.parent / 'producer.json')
        record.update({'triton_validation': str(args.triton_validation),
                       'triton_validation_sha256': sha(args.triton_validation),
                       'triton_profile': producer['profile'], 'triton_tok_s': serving['tok_s'],
                       'triton_model': 'word-games-qwen38-native',
                       'triton_load_unload_reload_verified': True,
                       'triton_existing_checks_pass': True})
    args.output.mkdir(parents=True, exist_ok=False)
    from structure_net.logging.standardized_logging import StandardizedLogger, LoggingConfig
    logger = StandardizedLogger(LoggingConfig(
        project_name='structure_net_completed_hypotheses', queue_dir='data/experiment_queue',
        sent_dir='data/experiment_sent', rejected_dir='data/experiment_rejected',
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
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--triton-validation', type=Path)
    run(p.parse_args())
