"""Register a hash-checked CPU expert result and verify exact/search read-back."""
import argparse
from pathlib import Path

from experiments.word_games.common import read, sha, write
from neural_architecture_lab.word_games_cpu_experts_hypothesis import CpuExpertEvidence


def run(args):
    data = read(args.analysis)
    if not data['service_restored']:
        raise ValueError('service restoration required')
    for key in ('campaign', 'preparation', 'manual_review'):
        if sha(data[key]) != data[key + '_sha256']:
            raise ValueError('changed ' + key)
    best = max(data['conditions'], key=lambda name: data['conditions'][name]['tok_s'])
    c = data['conditions'][best]
    record = CpuExpertEvidence(best, data['conditions']['packed38']['tok_s'], c['tok_s'],
        c['cpu_per_token_reduction'], c['receiver_correct'], c['receiver_cases'],
        str(args.analysis), sha(args.analysis), str(args.report), sha(args.report)).record()
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
        raise RuntimeError('searchable read-back missing')
    receipt = {'record': record, 'exact_readback': exact['metadatas'][0], 'semantic_ids': semantic['ids'][0]}
    write(Path('data/hypothesis_registry') / (record['id'] + '.json'), receipt)
    write(args.output / 'registry_readback.json', receipt)
    print('Registered and read back', record['id'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
